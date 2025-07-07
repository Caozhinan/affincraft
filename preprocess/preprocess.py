
import numpy as np  
from scipy.spatial.distance import cdist  
from pathlib import Path  
from torch_geometric.data import Data  
import torch  
from rdkit import Chem  
from pymol import cmd  
# 添加PLIP导入  
from plip.structure.preparation import PDBComplex  
import tempfile  
import os  
from plip_analysis import merge_protein_ligand_with_pymol  
# 扩展边特征编码  
SPATIAL_EDGE = [4, 0, 0]  # 原有空间边  
HYDROGEN_BOND_EDGE = [5, 1, 0]  # 氢键  
HYDROPHOBIC_EDGE = [5, 2, 0]    # 疏水相互作用  
PI_STACKING_EDGE = [5, 3, 0]    # π-π堆积  
PI_CATION_EDGE = [5, 4, 0]      # π-阳离子相互作用  
SALT_BRIDGE_EDGE = [5, 5, 0]    # 盐桥  
WATER_BRIDGE_EDGE = [5, 6, 0]   # 水桥  
HALOGEN_BOND_EDGE = [5, 7, 0]   # 卤键  
METAL_COMPLEX_EDGE = [5, 8, 0]  # 金属配位  
OTHERS_EDGE = [5, 9, 0]         # 其他相互作用  
  
# 相互作用类型映射  
INTERACTION_TYPE_MAP = {  
    'hydrogen_bonds': HYDROGEN_BOND_EDGE,        # 改为复数  
    'hydrophobic_contacts': HYDROPHOBIC_EDGE,    # 改为复数  
    'pi_stacking': PI_STACKING_EDGE,  
    'pi_cation': PI_CATION_EDGE,  
    'salt_bridges': SALT_BRIDGE_EDGE,            # 改为复数  
    'water_bridges': WATER_BRIDGE_EDGE,          # 改为复数  
    'halogen_bonds': HALOGEN_BOND_EDGE,          # 改为复数  
    'metal_complexes': METAL_COMPLEX_EDGE,       # 改为复数  
    'others': OTHERS_EDGE  
}

def analyze_plip_interactions(protein_file, ligand_file):  
    """  
    使用PLIP分析蛋白质-配体相互作用  
      
    Args:  
        protein_file: 蛋白质PDB文件路径  
        ligand_file: 配体SDF文件路径  
      
    Returns:  
        dict: 包含原子对相互作用信息的字典  
    """  
    # Create complex file in the same directory as the protein file  
    protein_path = Path(protein_file)  
    complex_file = protein_path.parent / "complex.pdb"  
      
     
    if not complex_file.exists():  
        print(f"Creating complex file: {complex_file}")  
          
        if not merge_protein_ligand_with_pymol(protein_file, ligand_file, str(complex_file)):  
            print(f"Failed to create complex file")  
            return {}  
          
        # 新增：修改HETATM行的链标识符  
        fix_hetatm_chain_ids(str(complex_file), 'X')  
        
    try:  
        my_mol = PDBComplex()  
        my_mol.load_pdb(str(complex_file))  
        my_mol.analyze()  
          
        # 提取原子对信息  
        atom_pairs_dict = {}  
          
        for bsid, interactions in my_mol.interaction_sets.items():  
            atom_pairs_dict[bsid] = {  
                'hydrogen_bonds': [],  
                'hydrophobic_contacts': [],  
                'pi_stacking': [],  
                'pi_cation': [],  
                'salt_bridges': [],  
                'water_bridges': [],  
                'halogen_bonds': [],  
                'metal_complexes': []  
            }  
              
            # 氢键  
            for hbond in interactions.hbonds_ldon + interactions.hbonds_pdon:  
                atom_pairs_dict[bsid]['hydrogen_bonds'].append({  
                    'protein_coords': hbond.d.coords if hbond.protisdon else hbond.a.coords,  
                    'ligand_coords': hbond.a.coords if hbond.protisdon else hbond.d.coords,  
                    'distance': hbond.distance_ad  
                })  
              
            # 疏水相互作用  
            for hydrophobic in interactions.hydrophobic_contacts:  
                atom_pairs_dict[bsid]['hydrophobic_contacts'].append({  
                    'protein_coords': hydrophobic.bsatom.coords,  
                    'ligand_coords': hydrophobic.ligatom.coords,  
                    'distance': hydrophobic.distance  
                })  
              
            # π-π堆积  
            for pistack in interactions.pistacking:  
                atom_pairs_dict[bsid]['pi_stacking'].append({  
                    'protein_coords': pistack.proteinring.center,  
                    'ligand_coords': pistack.ligandring.center,  
                    'distance': pistack.distance  
                })  
              
            # π-阳离子相互作用  
            for pication in interactions.pication_laro + interactions.pication_paro:  
                atom_pairs_dict[bsid]['pi_cation'].append({  
                    'protein_coords': pication.ring.center if pication.protcharged else pication.charge.center,  
                    'ligand_coords': pication.charge.center if pication.protcharged else pication.ring.center,  
                    'distance': pication.distance  
                })  
              
            # 盐桥  
            for saltbridge in interactions.saltbridge_lneg + interactions.saltbridge_pneg:  
                atom_pairs_dict[bsid]['salt_bridges'].append({  
                    'protein_coords': saltbridge.positive.center if saltbridge.protispos else saltbridge.negative.center,  
                    'ligand_coords': saltbridge.negative.center if saltbridge.protispos else saltbridge.positive.center,  
                    'distance': saltbridge.distance  
                })  
              
            # 水桥  
            for wbridge in interactions.water_bridges:  
                atom_pairs_dict[bsid]['water_bridges'].append({  
                    'protein_coords': wbridge.d.coords if wbridge.protisdon else wbridge.a.coords,  
                    'ligand_coords': wbridge.a.coords if wbridge.protisdon else wbridge.d.coords,  
                    'distance': (wbridge.distance_dw + wbridge.distance_aw) / 2  # 平均距离  
                })  
              
            # 卤键  
            for halogen in interactions.halogen_bonds:  
                atom_pairs_dict[bsid]['halogen_bonds'].append({  
                    'protein_coords': halogen.acc.coords,  
                    'ligand_coords': halogen.don.coords,  
                    'distance': halogen.distance  
                })  
              
            # 金属配位  
            for metal in interactions.metal_complexes:  
                atom_pairs_dict[bsid]['metal_complexes'].append({  
                    'protein_coords': metal.target.atom.coords,  
                    'ligand_coords': metal.metal.coords,  
                    'distance': metal.distance  
                })  
          
        return atom_pairs_dict  
          
    except Exception as e:  
        print(f"PLIP分析失败: {e}")  
        return {}  

def fix_hetatm_chain_ids(pdb_file, chain_id='X'):  
    """  
    修改PDB文件中HETATM行的链标识符  
      
    Args:  
        pdb_file: PDB文件路径  
        chain_id: 要设置的链标识符，默认为'X'  
    """  
    import os  
      
    # 读取原文件  
    with open(pdb_file, 'r') as f:  
        lines = f.readlines()  
      
    # 修改HETATM行  
    modified_lines = []  
    for line in lines:  
        if line.startswith('HETATM'):  
            # PDB格式：HETATM行的链标识符在第22位（索引21）  
            # 原格式：HETATM   48  H   UNK     0     105.436...  
            # 目标格式：HETATM   48  H   UNK X   0     105.436...  
            if len(line) > 21:  
                # 将第22位（索引21）设置为链标识符  
                line = line[:21] + chain_id + line[22:]  
            modified_lines.append(line)  
        else:  
            modified_lines.append(line)  
      
    # 写回文件  
    with open(pdb_file, 'w') as f:  
        f.writelines(modified_lines)  
      
    print(f"✓ 已修改PDB文件中的HETATM链标识符为: {chain_id}")

def convert_plip_to_edges(atom_pairs_dict, lig_coord, pro_coord, lig_num_atom):  
    """  
    将PLIP相互作用转换为图边  
      
    Args:  
        atom_pairs_dict: PLIP分析结果  
        lig_coord: 配体原子坐标  
        pro_coord: 蛋白原子坐标  
        lig_num_atom: 配体原子数量  
      
    Returns:  
        tuple: (edge_index, edge_attr) 包含相互作用类型和距离的边  
    """  
    edge_list = []  
    edge_attr_list = []  
      
    for bsid, interactions in atom_pairs_dict.items():  
        for interaction_type, pairs in interactions.items():  
            if not pairs:  
                continue  
                  
            edge_type = INTERACTION_TYPE_MAP.get(interaction_type, OTHERS_EDGE)  
              
            for pair in pairs:  
                # 找到最近的原子索引  
                lig_distances = cdist([pair['ligand_coords']], lig_coord)[0]  
                pro_distances = cdist([pair['protein_coords']], pro_coord)[0]  
                  
                lig_idx = np.argmin(lig_distances)  
                pro_idx = np.argmin(pro_distances)  
                  
                # 实际欧氏距离  
                actual_distance = np.linalg.norm(  
                    np.array(pair['ligand_coords']) - np.array(pair['protein_coords'])  
                )  
                  
                # 创建边特征：[类型编码] + [欧氏距离]  
                edge_feature = edge_type + [actual_distance]  
                  
                # 添加双向边  
                edge_list.append([lig_idx, pro_idx + lig_num_atom])  
                edge_list.append([pro_idx + lig_num_atom, lig_idx])  
                edge_attr_list.append(edge_feature)  
                edge_attr_list.append(edge_feature)  
      
    if edge_list:  
        edge_index = np.array(edge_list, dtype=np.int64).T  
        edge_attr = np.array(edge_attr_list, dtype=np.float32)  
    else:  
        edge_index = np.empty((2, 0), dtype=np.int64)  
        edge_attr = np.empty((0, 4), dtype=np.float32)  # 3个类型编码 + 1个距离  
      
    return edge_index, edge_attr

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
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 4), dtype=np.float32)  
      
    src, dst = np.where((dm <= spatial_cutoff) & (dm > 0.1))  
    edge_index = [(x, y) for x, y in zip(src, dst)]  
      
    # 为每条边添加距离特征  
    edge_attr = []  
    for x, y in edge_index:  
        distance = dm[x, y]  
        edge_feature = SPATIAL_EDGE + [distance]  # [4, 0, 0, distance]  
        edge_attr.append(edge_feature)  
  
    edge_index = np.array(edge_index, dtype=np.int64).T  
    edge_attr = np.array(edge_attr, dtype=np.float32)  # 改为float32  
    return edge_index, edge_attr

def gen_ligpro_edge(dm: np.ndarray, pocket_cutoff: float, plip_interactions=None, lig_coord=None, pro_coord=None):  
    # print(f"DEBUG: plip_interactions = {plip_interactions}")  
    # print(f"DEBUG: plip_interactions is None = {plip_interactions is None}")  
      
    # if plip_interactions:  
    #     print(f"DEBUG: Found {len(plip_interactions)} binding sites")  
    #     for bsid, interactions in plip_interactions.items():  
    #         print(f"DEBUG: Binding site {bsid} has interactions: {list(interactions.keys())}")  
    
#     生成配体-蛋白之间的边，包括PLIP相互作用和空间边  
#       
#     Args:  
#         dm: 距离矩阵  
#         pocket_cutoff: 距离阈值  
#         plip_interactions: PLIP分析结果  
#         lig_coord: 配体坐标  
#         pro_coord: 蛋白坐标   
    lig_num_atom, pro_num_atom = dm.shape  
      
    # 如果有PLIP分析结果，使用你之前写的convert_plip_to_edges函数  
    if plip_interactions and lig_coord is not None and pro_coord is not None:  
        return convert_plip_to_edges(plip_interactions, lig_coord, pro_coord, lig_num_atom)  
      
    # 否则生成普通空间边，但要包含距离特征  
    lig_idx, pro_idx = np.where(dm <= pocket_cutoff)  
    edge_list = []  
    edge_attr_list = []  
      
    for x, y in zip(lig_idx, pro_idx):  
        distance = dm[x, y]  
        edge_feature = SPATIAL_EDGE + [distance]  # [4, 0, 0, distance]  
          
        # 双向边  
        edge_list.extend([(x, y + lig_num_atom), (y + lig_num_atom, x)])  
        edge_attr_list.extend([edge_feature, edge_feature])  
      
    edge_index = np.array(edge_list, dtype=np.int64).T if edge_list else np.empty((2, 0), dtype=np.int64)  
    edge_attr = np.array(edge_attr_list, dtype=np.float32) if edge_attr_list else np.empty((0, 4), dtype=np.float32)  
      
    return edge_index, edge_attr

def gen_graph(ligand: tuple, pocket: tuple, name: str, protein_cutoff: float,   
              pocket_cutoff: float, spatial_cutoff: float, protein_file=None,   
              ligand_file=None, plip_interactions=None):  
    # 构建复合物的整体图结构（节点、特征、边、空间边）  
  
    # 解包配体和蛋白 pocket 的输入：坐标、特征、边索引、边特征  
    lig_coord, lig_feat, lig_ei, lig_ea = ligand  
    pro_coord, pro_feat, pro_ei, pro_ea = pocket  
  
    # 检查配体和蛋白的坐标和特征长度是否一致  
    assert len(lig_coord) == len(lig_feat)  
    assert len(pro_coord) == len(pro_feat)  
  
    # 检查 cutoff 参数的合理性  
    assert protein_cutoff >= pocket_cutoff, f"Protein cutoff {protein_cutoff} should be larger than pocket cutoff {pocket_cutoff}"  
    assert pocket_cutoff >= spatial_cutoff, f"Pocket cutoff {pocket_cutoff} should be larger than spatial cutoff {spatial_cutoff}"  
  
    # 如果没有提供PLIP分析结果且有文件路径，进行PLIP分析  
    if plip_interactions is None and protein_file and ligand_file:  
        plip_interactions = analyze_plip_interactions(protein_file, ligand_file)  
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
    lig_sei = np.empty((2, 0), dtype=np.int64)  
    lig_sea = np.empty((0, 4), dtype=np.float32)
    pro_sei = np.empty((2, 0), dtype=np.int64)  
    pro_sea = np.empty((0, 4), dtype=np.float32)
    # 配体内部空间边：所有距离小于 spatial_cutoff 的配体原子对
    lig_dm = cdist(lig_coord, lig_coord)   # 配体原子之间的距离矩阵
    lig_sei, lig_sea = gen_spatial_edge(lig_dm, spatial_cutoff=spatial_cutoff)  # 生成空间边
    lig_sei, lig_sea = remove_duplicated_edges(lig_sei, lig_sea, lig_ei)        # 去掉和结构边重复的空间边

    # 蛋白内部空间边：所有距离小于 spatial_cutoff 的蛋白原子对
    pro_dm = cdist(pro_coord, pro_coord)   # 蛋白原子之间的距离矩阵
    pro_sei, pro_sea = gen_spatial_edge(pro_dm, spatial_cutoff=spatial_cutoff)  # 生成空间边
    pro_sei, pro_sea = remove_duplicated_edges(pro_sei, pro_sea, pro_ei)        # 去掉和结构边重复的空间边

    # 配体-蛋白之间的空间边：所有距离小于 pocket_cutoff 的配体-蛋白原子对
    dm_lig_pro = cdist(lig_coord, pro_coord)  
    lig_pock_ei, lig_pock_ea = gen_ligpro_edge(  
        dm_lig_pro,   
        pocket_cutoff=pocket_cutoff,  
        plip_interactions=plip_interactions,  
        lig_coord=lig_coord,  
        pro_coord=pro_coord  
    )  
    def pad_edge_features_with_distance(edge_attr, edge_index, coords):  
        """为3维边特征添加距离信息，使其变为4维"""  
        if edge_attr.shape[1] == 3:  # 如果是3维特征  
            distances = []  
            for i in range(edge_index.shape[1]):  
                src_idx, dst_idx = edge_index[0, i], edge_index[1, i]  
                dist = np.linalg.norm(coords[src_idx] - coords[dst_idx])  
                distances.append(dist)  

            # 添加距离作为第4维  
            distances = np.array(distances).reshape(-1, 1)  
            edge_attr = np.hstack([edge_attr, distances]).astype(np.float32)  

        return edge_attr  
  
# 在拼接前处理所有边特征  
    lig_ea = pad_edge_features_with_distance(lig_ea, lig_ei, lig_coord)  
    pro_ea = pad_edge_features_with_distance(pro_ea, pro_ei, pro_coord)
    # 构建整体节点、特征
    comp_coord = np.vstack([lig_coord, pro_coord])   # 合并配体和蛋白的原子坐标
    comp_feat = np.vstack([lig_feat, pro_feat])      # 合并配体和蛋白的原子特征
    comp_ei, comp_ea = lig_ei, lig_ea                # 先只用配体的结构边作为初始边
    print(f"lig_ea shape: {lig_ea.shape}")  
    print(f"pro_ea shape: {pro_ea.shape}")  
    print(f"lig_sea shape: {lig_sea.shape}")  
    print(f"pro_sea shape: {pro_sea.shape}")  
    print(f"lig_pock_ea shape: {lig_pock_ea.shape}")

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
    lig_ei.shape[1],           # 配体结构边数  
    pro_ei.shape[1],           # 蛋白结构边数  
    lig_pock_ei.shape[1],      # 配体-蛋白边数  
    lig_sei.shape[1],          # 配体空间边数
    pro_sei.shape[1],          # 蛋白空间边数
], dtype=np.int64)
    return comp_coord, comp_feat, comp_ei, comp_ea, comp_num_node, comp_num_edge, lig_sei, lig_sea, pro_sei, pro_sea

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
         edge_attr=torch.from_numpy(comp_ea).to(torch.float32),  # 改为float32  
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