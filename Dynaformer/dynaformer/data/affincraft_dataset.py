import pickle  
import torch  
import numpy as np  
from pathlib import Path  
from typing import List, Dict, Any, Optional  
from torch.utils.data import Dataset, DataLoader  
  
# 如果需要使用现有的数据处理工具  
# from .wrapper import preprocess_item  
# from .collator import collator

class AffinCraftDataset(torch.utils.data.Dataset):  
    def __init__(self, pkl_files):  
        self.pkl_files = pkl_files  
          
    def __len__(self):  
        return len(self.pkl_files)  
      
    def __getitem__(self, idx):  
        pkl_file = self.pkl_files[idx]  
        with open(pkl_file, 'rb') as f:  
            pkl_data = pickle.load(f)[0]  # 假设每个PKL文件包含一个复合物  
          
        return preprocess_affincraft_item(pkl_data)
    
def preprocess_affincraft_item(pkl_data):  
    """专门处理AffinCraft PKL文件的预处理函数"""  
      
    # 直接使用PKL中的特征  
    node_feat = torch.from_numpy(pkl_data['node_feat'])  
    edge_index = torch.from_numpy(pkl_data['edge_index'])  
    edge_feat = torch.from_numpy(pkl_data['edge_feat'])  
    coords = torch.from_numpy(pkl_data['coords'])  
      
    # 处理分离的空间边信息  
    lig_spatial_edges = {  
        'index': torch.from_numpy(pkl_data['lig_spatial_edge_index']),  
        'attr': torch.from_numpy(pkl_data['lig_spatial_edge_attr'])  
    }  
      
    pro_spatial_edges = {  
        'index': torch.from_numpy(pkl_data['pro_spatial_edge_index']),  
        'attr': torch.from_numpy(pkl_data['pro_spatial_edge_attr'])  
    }  
      
    # 保留MaSIF特征  
    masif_features = {  
        'input_feat': torch.from_numpy(pkl_data['masif_input_feat']),  
        'desc_straight': torch.from_numpy(pkl_data['masif_desc_straight']),  
        'desc_flipped': torch.from_numpy(pkl_data['masif_desc_flipped']),  
        'rho_wrt_center': torch.from_numpy(pkl_data['masif_rho_wrt_center']),  
        'theta_wrt_center': torch.from_numpy(pkl_data['masif_theta_wrt_center']),  
        'mask': torch.from_numpy(pkl_data['masif_mask'])  
    }  
      
    # 添加embedding层需要的额外字段  
    N = node_feat.shape[0]  
      
    # 创建基础注意力偏置矩阵  
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  
      
    # 计算度数信息（从边索引计算）  
    adj = torch.zeros([N, N], dtype=torch.bool)  
    adj[edge_index[0, :], edge_index[1, :]] = True  
    in_degree = adj.long().sum(dim=1).view(-1)  
    out_degree = in_degree  # 对于无向图  
      
    return {  
        'node_feat': node_feat,  
        'edge_index': edge_index,  
        'edge_feat': edge_feat,  
        'coords': coords,  
        'lig_spatial_edges': lig_spatial_edges,  
        'pro_spatial_edges': pro_spatial_edges,  
        'masif_features': masif_features,  
        'num_ligand_atoms': pkl_data['num_node'][0],  
        'attn_bias': attn_bias,  
        'in_degree': in_degree,  
        'out_degree': out_degree,  
        'gbscore': torch.from_numpy(pkl_data['gbscore']),  
        'pdbid': pkl_data['pdbid'],  
        'pk': pkl_data['pk'],  
        'smiles': pkl_data['smiles']  
    }

def affincraft_collator(items, max_node=512):  
    """AffinCraft数据的批处理函数"""  
      
    # 过滤无效数据  
    items = [item for item in items if item is not None and item['node_feat'].size(0) <= max_node]  
      
    if not items:  
        return None  
      
    max_node_num = max(item['node_feat'].size(0) for item in items)  
    max_edge_num = max(item['edge_feat'].size(0) for item in items)  # 新增：计算最大边数  
      
    # 批处理特征  
    node_feats = []  
    edge_feats = []  
    edge_indices = []  
    edge_masks = []  # 新增：边掩码  
    coords_list = []  
    attn_biases = []  
    in_degrees = []  
    out_degrees = []  
      
    for item in items:  
        n_node = item['node_feat'].size(0)  
        n_edge = item['edge_feat'].size(0)  
          
        # 填充节点特征  
        padded_node_feat = torch.zeros(max_node_num, item['node_feat'].size(1))  
        padded_node_feat[:n_node] = item['node_feat']  
        node_feats.append(padded_node_feat)  
          
        # 填充边特征 - 关键修改  
        padded_edge_feat = torch.zeros(max_edge_num, item['edge_feat'].size(1))  
        padded_edge_feat[:n_edge] = item['edge_feat']  
        edge_feats.append(padded_edge_feat)  
          
        # 填充边索引  
        padded_edge_index = torch.zeros(2, max_edge_num, dtype=torch.long)  
        padded_edge_index[:, :n_edge] = item['edge_index']  
        edge_indices.append(padded_edge_index)  
          
        # 创建边掩码  
        edge_mask = torch.zeros(max_edge_num, dtype=torch.bool)  
        edge_mask[:n_edge] = True  
        edge_masks.append(edge_mask)  
          
        # 其他特征处理保持不变  
        padded_coords = torch.zeros(max_node_num, 3)  
        padded_coords[:n_node] = item['coords']  
        coords_list.append(padded_coords)  
          
        padded_attn_bias = torch.zeros(max_node_num + 1, max_node_num + 1)  
        padded_attn_bias[:n_node+1, :n_node+1] = item['attn_bias']  
        attn_biases.append(padded_attn_bias)  
          
        padded_in_degree = torch.zeros(max_node_num, dtype=torch.long)  
        padded_in_degree[:n_node] = item['in_degree']  
        in_degrees.append(padded_in_degree)  
          
        padded_out_degree = torch.zeros(max_node_num, dtype=torch.long)  
        padded_out_degree[:n_node] = item['out_degree']  
        out_degrees.append(padded_out_degree)  
      
    return {  
        'node_feat': torch.stack(node_feats),  
        'edge_feat': torch.stack(edge_feats),  # 改为tensor  
        'edge_index': torch.stack(edge_indices),  # 改为tensor  
        'edge_mask': torch.stack(edge_masks),  # 新增边掩码  
        'coords': torch.stack(coords_list),  
        'attn_bias': torch.stack(attn_biases),  
        'in_degree': torch.stack(in_degrees),  
        'out_degree': torch.stack(out_degrees),  
        'num_ligand_atoms': torch.tensor([item['num_ligand_atoms'] for item in items]),  
        'gbscore': torch.stack([item['gbscore'] for item in items]),  
        'masif_desc_straight': torch.stack([item['masif_features']['desc_straight'] for item in items]),  
        'pdbid': [item['pdbid'] for item in items],  
        'pk': torch.tensor([item['pk'] for item in items]),  
        'smiles': [item['smiles'] for item in items]  
    }


def create_affincraft_dataloader(pkl_files, batch_size=32, shuffle=True):  
    """创建AffinCraft数据加载器"""  
    dataset = AffinCraftDataset(pkl_files)  
      
    return torch.utils.data.DataLoader(  
        dataset,  
        batch_size=batch_size,  
        shuffle=shuffle,  
        collate_fn=affincraft_collator,  
        num_workers=4  
    )