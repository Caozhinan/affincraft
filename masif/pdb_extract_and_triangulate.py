import numpy as np  
import os  
import Bio  
import shutil  
from Bio.PDB import *   
import sys  
import importlib  
from IPython.core.debugger import set_trace  
  
# 本地导入  
from source.default_config.masif_opts import masif_opts  
from source.triangulation.computeMSMS import computeMSMS  
from source.triangulation.fixmesh import fix_mesh  
from source.triangulation.ligand_utils import extract_ligand  
import pymesh  
from source.input_output.extractPDB import extractPDB  
from source.input_output.save_ply import save_ply  
from source.input_output.read_ply import read_ply  
from source.input_output.protonate import protonate  
from source.triangulation.computeHydrophobicity import computeHydrophobicity  
from source.triangulation.computeCharges import computeCharges, assignChargesToNewMesh  
from source.triangulation.computeAPBS import computeAPBS  
from source.triangulation.compute_normal import compute_normal  
from sklearn.neighbors import KDTree  
  
def extract_and_triangulate_surface(pdb_file, chain_id, ligand_code=None, ligand_chain=None, sdf_file=None, mol2_patch=None, output_dir=None):  
    """  
    提取PDB并计算表面三角化  
      
    参数:  
    - pdb_file: PDB文件路径  
    - chain_id: 蛋白质链ID  
    - ligand_code: 配体三字母代码 (可选)  
    - ligand_chain: 配体所在链 (可选)  
    - sdf_file: SDF模板文件路径 (可选)  
    - mol2_patch: MOL2补丁文件路径 (可选)  
    - output_dir: 输出目录  
    """  
      
    # 1. 设置临时目录  
    tmp_dir = output_dir + "/tmp/" if output_dir else masif_opts['tmp_dir']  
    os.makedirs(tmp_dir, exist_ok=True)  
      
    # 2. 质子化PDB文件  
    pdb_id = os.path.basename(pdb_file).replace('.pdb', '')  
    protonated_file = tmp_dir + "/" + pdb_id + "_protonated.pdb"  
    protonate(pdb_file, protonated_file)  
      
    # 3. 提取指定链  
    out_filename = tmp_dir + "/" + pdb_id + "_" + chain_id  
    extractPDB(protonated_file, out_filename + ".pdb", chain_id, ligand_code, ligand_chain)  
      
    # 4. 计算MSMS表面  
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(  
        out_filename + ".pdb",   
        protonate=True,   
        ligand_code=ligand_code  
    )  
      
    # 5. 处理配体（如果存在）  
    rdmol = None  
    mol2_file = None  
    if ligand_code is not None and ligand_chain is not None:  
        mol2_file = os.path.join(tmp_dir, "{}_{}.mol2".format(ligand_code, ligand_chain))  
        rdmol = extract_ligand(  
            protonated_file,   
            ligand_code,   
            ligand_chain,   
            mol2_file,   
            sdf_template=sdf_file,   
            patched_mol2_file=mol2_patch  
        )  
        print(f"成功提取配体 {ligand_code}")  
      
    # 6. 计算化学特征  
    vertex_hbond = None  
    vertex_hphobicity = None  
    vertex_charges = None  
      
    # 氢键特征  
    if masif_opts['use_hbond']:  
        vertex_hbond = computeCharges(out_filename, vertices1, names1, ligand_code, rdmol)  
      
    # 疏水性特征  
    if masif_opts['use_hphob']:  
        vertex_hphobicity = computeHydrophobicity(names1, ligand_code, rdmol)  
      
    # 7. 网格修复  
    mesh = pymesh.form_mesh(vertices1, faces1)  
    print("修复网格中...")  
    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])  
    print("网格修复完成!")  
      
    # 8. 计算法向量  
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)  
      
    # 9. 将特征分配到新网格  
    if masif_opts['use_hbond'] and vertex_hbond is not None:  
        vertex_hbond = assignChargesToNewMesh(  
            regular_mesh.vertices, vertices1, vertex_hbond, masif_opts  
        )  
      
    if masif_opts['use_hphob'] and vertex_hphobicity is not None:  
        vertex_hphobicity = assignChargesToNewMesh(  
            regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts  
        )  
      
    # 10. 计算APBS电荷（如果启用）  
    if masif_opts['use_apbs']:  
        print("计算APBS中...")  
        vertex_charges = computeAPBS(  
            regular_mesh.vertices,   
            out_filename + ".pdb",   
            out_filename,   
            mol2_file  
        )  
        print("APBS计算完成!")  
      
    # 11. 计算界面标签（可选）  
    iface = np.zeros(len(regular_mesh.vertices))  
    if 'compute_iface' in masif_opts and masif_opts['compute_iface']:  
        # 计算整个复合物的表面  
        v3, f3, _, _, _ = computeMSMS(protonated_file, protonate=True, ligand_code=ligand_code)  
        full_mesh = pymesh.form_mesh(v3, f3)  
          
        # 找到界面顶点  
        from sklearn.neighbors import KDTree  
        kdt = KDTree(full_mesh.vertices)  
        d, r = kdt.query(regular_mesh.vertices)  
        d = np.square(d)  
        iface_v = np.where(d >= 2.0)[0]  
        iface[iface_v] = 1.0  
      
    # 12. 保存PLY文件  
    output_ply = out_filename + ".ply"  
    save_ply(  
        output_ply,  
        regular_mesh.vertices,  
        regular_mesh.faces,  
        normals=vertex_normal,  
        charges=vertex_charges,  
        normalize_charges=True,  
        hbond=vertex_hbond,  
        hphob=vertex_hphobicity,  
        iface=iface  
    )  
      
    # 13. 复制到输出目录  
    if output_dir:  
        ply_dir = os.path.join(output_dir, "surfaces")  
        pdb_dir = os.path.join(output_dir, "pdbs")  
        os.makedirs(ply_dir, exist_ok=True)  
        os.makedirs(pdb_dir, exist_ok=True)  
          
        shutil.copy(output_ply, ply_dir)  
        shutil.copy(out_filename + '.pdb', pdb_dir)  
      
    return {  
        'ply_file': output_ply,  
        'pdb_file': out_filename + '.pdb',  
        'vertices': regular_mesh.vertices,  
        'faces': regular_mesh.faces,  
        'normals': vertex_normal,  
        'charges': vertex_charges,  
        'hbond': vertex_hbond,  
        'hphobicity': vertex_hphobicity,  
        'interface': iface,  
        'ligand_mol': rdmol  
    }  
  
# 使用示例  
if __name__ == "__main__":  
    # 示例：处理蛋白质-配体复合物  
    result = extract_and_triangulate_surface(  
        pdb_file="example.pdb",  
        chain_id="A",   
        ligand_code="LIG",  
        ligand_chain="A",  
        sdf_file="ligand.sdf",  
        output_dir="./output"  
    )  
      
    print(f"表面三角化完成!")  
    print(f"PLY文件: {result['ply_file']}")  
    print(f"顶点数: {len(result['vertices'])}")  
    print(f"面数: {len(result['faces'])}")