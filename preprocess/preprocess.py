import numpy as np  # 科学计算库
from scipy.spatial.distance import cdist  # 计算原子间距离矩阵
from pathlib import Path  # 跨平台文件路径
from torch_geometric.data import Data  # PyG图数据结构
import torch
from rdkit import Chem  # RDKit分子处理
# from mol2graph import mol2graph, mol2graph_ligand, mol2graph_protein_from_pdb   # 分子转图节点/边特征
from pymol import cmd  # PyMOL命令接口
from ecif import GetECIF  # ECIF特征（蛋白-配体原子对）

SPATIAL_EDGE = [4, 0, 0]  # 空间边的特征编码

def dim2(arr: np.ndarray):
    # 判断数组是否为二维
    return len(arr.shape) == 2

def reindex(atom_idx: list, edge_index: np.ndarray):
    # 对边的节点编号重新映射（因为去除部分原子后索引变化）
    if len(edge_index.shape) != 2:
        return edge_index
    indexmap = {old: new for new, old in enumerate(atom_idx)}
    mapfunc = np.vectorize(indexmap.get)
    edge_index_new = np.array([mapfunc(edge_index[0]), mapfunc(edge_index[1])])
    return edge_index_new

def remove_duplicated_edges(ei: np.ndarray, ea: np.ndarray, ref_ei: np.ndarray):
    # 删除和已有边重复的边（避免冗余）
    if not len(ea):
        return ei, ea
    existing_set = {(i, j) for i, j in zip(*ref_ei)}
    mask = []
    for i, j in zip(*ei):
        if (i, j) in existing_set:
            mask.append(False)  # 已有则删除
        else:
            mask.append(True)  # 否则保留
    mask = np.array(mask, dtype=bool)
    ei_n = np.array([ei[0][mask], ei[1][mask]])
    ea_n = ea[mask]
    return ei_n, ea_n

# def gen_feature(ligand: Chem.Mol, pocket: Chem.Mol, pdbid: str):  
#     ligand_dict = mol2graph_ligand(ligand)  # 使用小分子版本  
#     pocket_dict = mol2graph_protein_from_pdb(pocket)  # 使用蛋白质版本  
      
#     ligand_coords, ligand_features, ligand_edge_index, ligand_edge_attr = ligand_dict['coords'], ligand_dict[  
#         'node_feat'], ligand_dict['edge_index'], ligand_dict['edge_feat']  
#     pocket_coords, pocket_features, pocket_edge_index, pocket_edge_attr = pocket_dict['coords'], pocket_dict[  
#         'node_feat'], pocket_dict['edge_index'], pocket_dict['edge_feat']  
  
#     # 形状检查  
#     if not (dim2(ligand_coords) and dim2(ligand_features) and dim2(ligand_edge_index) and dim2(ligand_edge_attr)):  
#         raise RuntimeError(f"Ligand feature shape error")  
#     if not (dim2(pocket_coords) and dim2(pocket_features) and dim2(pocket_edge_index) and dim2(pocket_edge_attr)):  
#         raise RuntimeError(f"Protein feature shape error")  
  
#     return {'lc': ligand_coords, 'lf': ligand_features, 'lei': ligand_edge_index, 'lea': ligand_edge_attr,  
#             'pc': pocket_coords, 'pf': pocket_features, 'pei': pocket_edge_index, 'pea': pocket_edge_attr,  
#             'pdbid': pdbid,  
#             'ligand_smiles': ligand_dict['smiles'],  # 新增  
#             'protein_atom_names': pocket_dict['pro_name'],  # 新增  
#             'protein_aa_names': pocket_dict['AA_name']}  # 新增

def gen_spatial_edge(dm: np.ndarray, spatial_cutoff: float = 5):
    # 生成所有空间距离<cutoff的节点对，并赋空间边特征
    if spatial_cutoff <= 0.1:
        return np.array([]), np.array([])
    src, dst = np.where((dm <= spatial_cutoff) & (dm > 0.1))
    # 已经对称
    edge_index = [(x, y) for x, y in zip(src, dst)]
    edge_attr = np.array([SPATIAL_EDGE for _ in edge_index])

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.int64)
    return edge_index, edge_attr

def gen_ligpro_edge(dm: np.ndarray, pocket_cutoff: float):
    # 生成配体-蛋白之间的空间边（小于pocket_cutoff）
    lig_num_atom, pro_num_atom = dm.shape
    lig_idx, pro_idx = np.where(dm <= pocket_cutoff)
    # 配体->蛋白 + 蛋白->配体都加
    edge_index = [(x, y + lig_num_atom) for x, y in zip(lig_idx, pro_idx)]
    edge_index += [(y + lig_num_atom, x) for x, y in zip(lig_idx, pro_idx)]
    edge_attr = np.array([SPATIAL_EDGE for _ in edge_index])
    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.int64)
    return edge_index, edge_attr

def gen_graph(ligand: tuple, pocket: tuple, name: str, protein_cutoff: float, pocket_cutoff: float, spatial_cutoff: float):
    # 构建复合物的整体图结构（节点、特征、边、空间边）

    # 解包配体和蛋白 pocket 的输入：坐标、特征、边索引、边特征
    lig_coord, lig_feat, lig_ei, lig_ea = ligand   # 配体的原子坐标、特征、边索引、边特征
    pro_coord, pro_feat, pro_ei, pro_ea = pocket   # 蛋白的原子坐标、特征、边索引、边特征

    # 检查配体和蛋白的坐标和特征长度是否一致
    assert len(lig_coord) == len(lig_feat)
    assert len(pro_coord) == len(pro_feat)

    # 检查 cutoff 参数的合理性
    assert protein_cutoff >= pocket_cutoff, f"Protein cutoff {protein_cutoff} should be larger than pocket cutoff {pocket_cutoff}"
    assert pocket_cutoff >= spatial_cutoff, f"Protein cutoff {protein_cutoff} should be larger than spatial cutoff {spatial_cutoff}"

    # 只保留距离配体小于 protein_cutoff 的蛋白原子
    # 先初始化全为 False 的掩码
    pro_atom_mask = np.zeros(len(pro_coord), dtype=bool)
    # 计算配体和蛋白的距离矩阵，保留距离小于 protein_cutoff 的蛋白原子的索引
    pro_atom_mask[np.where(cdist(lig_coord, pro_coord) <= protein_cutoff)[1]] = 1
    # 只保留蛋白内部边的两端都在掩码范围内的边
    pro_edge_mask = np.array([True if pro_atom_mask[i] and pro_atom_mask[j] else False for i, j in zip(*pro_ei)])

    # 筛选对应的蛋白节点和边
    pro_coord = pro_coord[pro_atom_mask]   # 只保留掩码内的坐标
    pro_feat = pro_feat[pro_atom_mask]     # 只保留掩码内的特征
    # 只保留掩码内的边
    pro_ei = np.array([pro_ei[0, pro_edge_mask], pro_ei[1, pro_edge_mask]])
    pro_ea = pro_ea[pro_edge_mask]
    # 边的原子索引需要重新编号（因为节点删减了）
    pro_ei = reindex(np.where(pro_atom_mask)[0], pro_ei)

    # 配体内部空间边：所有距离小于 spatial_cutoff 的配体原子对
    lig_dm = cdist(lig_coord, lig_coord)   # 配体原子之间的距离矩阵
    lig_sei, lig_sea = gen_spatial_edge(lig_dm, spatial_cutoff=spatial_cutoff)  # 生成空间边
    lig_sei, lig_sea = remove_duplicated_edges(lig_sei, lig_sea, lig_ei)        # 去掉和结构边重复的空间边

    # 蛋白内部空间边：所有距离小于 spatial_cutoff 的蛋白原子对
    pro_dm = cdist(pro_coord, pro_coord)   # 蛋白原子之间的距离矩阵
    pro_sei, pro_sea = gen_spatial_edge(pro_dm, spatial_cutoff=spatial_cutoff)  # 生成空间边
    pro_sei, pro_sea = remove_duplicated_edges(pro_sei, pro_sea, pro_ei)        # 去掉和结构边重复的空间边

    # 配体-蛋白之间的空间边：所有距离小于 pocket_cutoff 的配体-蛋白原子对
    dm_lig_pro = cdist(lig_coord, pro_coord)  # 配体到蛋白的距离矩阵
    lig_pock_ei, lig_pock_ea = gen_ligpro_edge(dm_lig_pro, pocket_cutoff=pocket_cutoff)  # 生成配体-蛋白边

    # 构建整体节点、特征
    comp_coord = np.vstack([lig_coord, pro_coord])   # 合并配体和蛋白的原子坐标
    comp_feat = np.vstack([lig_feat, pro_feat])      # 合并配体和蛋白的原子特征
    comp_ei, comp_ea = lig_ei, lig_ea                # 先只用配体的结构边作为初始边

    # 拼接所有蛋白、配体、空间和配体-蛋白边
    if len(pro_ei.shape) == 2 and len(pro_ei.T) >= 3:   # 如果蛋白有结构边
        comp_ei = np.hstack([comp_ei, pro_ei + len(lig_feat)])   # 蛋白边下标整体平移
        comp_ea = np.vstack([comp_ea, pro_ea])                  # 合并蛋白边特征
    if len(lig_sei.shape) == 2 and len(lig_sei.T) >= 3:         # 如果配体有空间边
        comp_ei = np.hstack([comp_ei, lig_sei])                 # 合并配体空间边
        comp_ea = np.vstack([comp_ea, lig_sea])                 # 合并配体空间边特征
    if len(pro_sei.shape) == 2 and len(pro_sei.T) >= 3:         # 如果蛋白有空间边
        comp_ei = np.hstack([comp_ei, pro_sei + len(lig_feat)]) # 蛋白空间边下标整体平移
        comp_ea = np.vstack([comp_ea, pro_sea])                 # 合并蛋白空间边特征
    if len(lig_pock_ei.shape) == 2 and len(lig_pock_ei.T) >= 3: # 如果有配体-蛋白空间边
        comp_ei = np.hstack([comp_ei, lig_pock_ei])             # 合并配体-蛋白空间边
        comp_ea = np.vstack([comp_ea, lig_pock_ea])             # 合并配体-蛋白空间边特征

    # 记录每类节点、边数
    comp_num_node = np.array([len(lig_feat), len(pro_feat)], dtype=np.int64)   # 配体、蛋白节点数
    comp_num_edge = np.array([
        lig_ei.T.shape[0],               # 配体结构边数
        pro_ei.T.shape[0],               # 蛋白结构边数
        lig_pock_ei.T.shape[0],          # 配体-蛋白边数
        lig_sei.T.shape[0],              # 配体空间边数
        pro_sei.T.shape[0],              # 蛋白空间边数
    ], dtype=np.int64)
    return comp_coord, comp_feat, comp_ei, comp_ea, comp_num_node, comp_num_edge  # 返回所有信息

def load_pk_data(data_path: Path):
    # 从txt读取pdbid与pk值
    pdbid, pk = [], []
    for line in open(data_path):
        if line[0] == '#': continue
        elem = line.split()
        v1, _, _, v2 = elem[:4]
        pdbid.append(v1)
        pk.append(float(v2))
    res = {i: p for i, p in zip(pdbid, pk)}
    return res

def to_pyg_graph(raw: list, **kwargs):  
    comp_coord, comp_feat, comp_ei, comp_ea, comp_num_node, comp_num_edge, rfscore, gbscore, ecif, pk, name = raw  
      
    # 从kwargs中提取rmsd值  
    rmsd = kwargs.pop('rmsd', 0.0)  # 默认值为0.0  
  
    d = Data(x=torch.from_numpy(comp_feat).to(torch.long),   
             edge_index=torch.from_numpy(comp_ei).to(torch.long),   
             edge_attr=torch.from_numpy(comp_ea).to(torch.long),  
             pos=torch.from_numpy(comp_coord).to(torch.float32),   
             y=torch.tensor([pk], dtype=torch.float32),   
             pdbid=name,  
             num_node=torch.from_numpy(comp_num_node).to(torch.long),   
             num_edge=torch.from_numpy(comp_num_edge).to(torch.long),  
             rfscore=torch.from_numpy(rfscore).to(torch.float32),   
             gbscore=torch.from_numpy(gbscore).to(torch.float32),  
             ecif=torch.from_numpy(ecif).to(dtype=torch.float32),  
             rmsd=torch.tensor([rmsd], dtype=torch.float32),  # 新增：rmsd值  
             **kwargs)  
    return d

def get_info(protein_file, ligand_file):
    # 用PyMOL获取蛋白和配体的元素、残基和坐标信息
    cmd.reinitialize()
    cmd.load(protein_file, 'receptor')
    cmd.load(ligand_file, 'ligand')
    cmd.remove('sol.')
    cmd.h_add()
    proinfo = {"elem": [], "resn": [], "coord":[], }
    liginfo = {"elem": [], "resn": [], "coord":[], }
    cmd.iterate_state(1, 'receptor', 'info["elem"].append(elem); info["resn"].append(resn); info["coord"].append(np.array([x, y, z]))',space={"info": proinfo, "np": np})
    cmd.iterate_state(1, 'ligand', 'info["elem"].append(elem); info["resn"].append(resn); info["coord"].append(np.array([x, y, z]))',space={"info": liginfo, "np": np})
    for k in proinfo.keys():
        proinfo[k] = np.array(proinfo[k])
        liginfo[k] = np.array(liginfo[k])
    return proinfo, liginfo

def GB_score(lig_info: dict, pro_info: dict) -> np.ndarray:
    # 生成GBScore特征（配体-蛋白原子对分组距离统计，400维）
    amino_acid_groups = [
        {"ARG", "LYS", "ASP", "GLU"},
        {"GLN", "ASN", "HIS", "SER", "THR", "CYS"},
        {"TRP", "TYR", "MET"},
        {"ILE", "LEU", "PHE", "VAL", "PRO", "GLY", "ALA"},
    ]
    elements = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    distmap = cdist(lig_info['coord'], pro_info['coord'])
    restype = np.zeros(len(pro_info['resn'])) - 1
    elem_mask = {k: pro_info["elem"] == k for k in elements}
    fp = np.zeros([len(elements), len(elements), len(amino_acid_groups)])

    for idx, r in enumerate(pro_info['resn']):
        if r in amino_acid_groups[0]: restype[idx] = 0
        if r in amino_acid_groups[1]: restype[idx] = 1
        if r in amino_acid_groups[2]: restype[idx] = 2
        if r in amino_acid_groups[3]: restype[idx] = 3

    for i, el in enumerate(elements):
        lmask = lig_info["elem"] == el
        if lmask.sum() < 1: continue
        for j, ep in enumerate(elements):
            pmask = elem_mask[ep]
            if pmask.sum() < 1: continue
            for k, rt in enumerate(range(4)):
                rt_mask = restype == rt
                m = pmask & rt_mask
                if m.sum() < 1: continue
                d = distmap[lmask][:, m]
                v = (1 / d[d<=12]).sum()
                fp[i, j, k] = v

    return fp.flatten()

def RF_score(lig_info: dict, pro_info: dict):
    # 生成RFScore特征（配体-蛋白原子对12A内原子对数，100维）
    LIG_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    PRO_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    distmap = cdist(lig_info['coord'], pro_info['coord'])
    fp = np.zeros([10, 10])
    for i, el in enumerate(LIG_TYPES):
        lmask = lig_info['elem'] == el
        if lmask.sum() < 1: continue
        for j, ep in enumerate(PRO_TYPES):
            pmask = pro_info['elem'] == ep
            d = distmap[lmask][:, pmask]
            v = d[d < 12].shape[0]
            fp[i, j] = v
    return fp.flatten()