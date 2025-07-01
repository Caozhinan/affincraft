#!/usr/bin/env python3  
"""  
PKL文件分析脚本 - 读取并输出所有类型的边特征信息（字典格式）  
"""  
  
import pickle  
import torch  
import numpy as np  
from pathlib import Path  
import sys  
  
def analyze_edge_types_detailed(edge_attr):  
    """详细分析边的类型分布，包括所有PLIP相互作用类型"""  
    edge_types = {}  
    edge_type_stats = {}  
      
    # 处理numpy数组或tensor  
    if isinstance(edge_attr, torch.Tensor):  
        edge_attr = edge_attr.numpy()  
      
    for i, edge_feature in enumerate(edge_attr):  
        if len(edge_feature) >= 3:  
            # 前3维是边类型编码  
            edge_type_code = tuple(edge_feature[:3].astype(int))  
              
            # 根据你定义的边类型映射  
            if edge_type_code == (4, 0, 0):  
                edge_type = "SPATIAL_EDGE"  
                edge_category = "空间边"  
            elif edge_type_code == (5, 1, 0):  
                edge_type = "HYDROGEN_BOND"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 2, 0):  
                edge_type = "HYDROPHOBIC_CONTACT"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 3, 0):  
                edge_type = "PI_STACKING"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 4, 0):  
                edge_type = "PI_CATION"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 5, 0):  
                edge_type = "SALT_BRIDGE"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 6, 0):  
                edge_type = "WATER_BRIDGE"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 7, 0):  
                edge_type = "HALOGEN_BOND"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 8, 0):  
                edge_type = "METAL_COMPLEX"  
                edge_category = "PLIP相互作用边"  
            elif edge_type_code == (5, 9, 0):  
                edge_type = "OTHERS"  
                edge_category = "PLIP相互作用边"  
            else:  
                # 可能是结构边（化学键）  
                edge_type = f"STRUCTURAL_BOND_{edge_type_code}"  
                edge_category = "结构边"  
              
            if edge_type not in edge_types:  
                edge_types[edge_type] = []  
                edge_type_stats[edge_type] = {  
                    'category': edge_category,  
                    'code': edge_type_code,  
                    'count': 0,  
                    'distances': []  
                }  
              
            edge_type_stats[edge_type]['count'] += 1  
              
            # 如果有第4维，那是距离信息  
            if len(edge_feature) > 3:  
                distance = edge_feature[3]  
                edge_types[edge_type].append(distance)  
                edge_type_stats[edge_type]['distances'].append(distance)  
            else:  
                edge_types[edge_type].append("N/A")  
      
    return edge_types, edge_type_stats  
  
def print_edge_summary(edge_type_stats):  
    """打印边类型摘要统计"""  
    print("\n=== 边类型摘要统计 ===")  
      
    # 按类别分组  
    categories = {}  
    for edge_type, stats in edge_type_stats.items():  
        category = stats['category']  
        if category not in categories:  
            categories[category] = []  
        categories[category].append((edge_type, stats))  
      
    total_edges = sum(stats['count'] for stats in edge_type_stats.values())  
    print(f"总边数: {total_edges}")  
      
    for category, edges in categories.items():  
        category_total = sum(stats['count'] for _, stats in edges)  
        print(f"\n{category}: {category_total} 条边 ({category_total/total_edges*100:.1f}%)")  
          
        for edge_type, stats in sorted(edges, key=lambda x: x[1]['count'], reverse=True):  
            count = stats['count']  
            percentage = count / total_edges * 100  
            print(f"  {edge_type}: {count} 条 ({percentage:.1f}%)")  
              
            if stats['distances'] and stats['distances'][0] != "N/A":  
                distances = [d for d in stats['distances'] if d != "N/A"]  
                if distances:  
                    print(f"    距离范围: {min(distances):.3f} - {max(distances):.3f} Å")  
                    print(f"    平均距离: {np.mean(distances):.3f} Å")  
  
def write_detailed_analysis_to_file(graphs, output_file):  
    """将详细分析结果写入文件"""  
    with open(output_file, 'w', encoding='utf-8') as f:  
        f.write("=== PKL文件分子图详细分析报告 ===\n\n")  
        f.write(f"总复合物数量: {len(graphs)}\n\n")  
          
        for idx, graph in enumerate(graphs):  
            f.write(f"{'='*60}\n")  
            f.write(f"复合物 {idx + 1}: {graph.get('pdbid', 'Unknown')}\n")  
            f.write(f"{'='*60}\n\n")  
              
            # 基本信息  
            f.write("=== 基本信息 ===\n")  
            f.write(f"PDB ID: {graph.get('pdbid', 'N/A')}\n")  
            f.write(f"结合亲和力 (pK): {graph.get('pk', 'N/A')}\n")  
            f.write(f"SMILES: {graph.get('smiles', 'N/A')}\n")  
            f.write(f"RMSD: {graph.get('rmsd', 'N/A')}\n\n")  
              
            # 图结构信息  
            f.write("=== 图结构信息 ===\n")  
            node_feat = graph.get('node_feat', np.array([]))  
            edge_index = graph.get('edge_index', np.array([]))  
            edge_feat = graph.get('edge_feat', np.array([]))  
            coords = graph.get('coords', np.array([]))  
              
            f.write(f"节点数量: {node_feat.shape[0] if len(node_feat.shape) > 0 else 0}\n")  
            f.write(f"边数量: {edge_index.shape[1] if len(edge_index.shape) > 1 else 0}\n")  
            f.write(f"节点特征维度: {node_feat.shape[1] if len(node_feat.shape) > 1 else 0}\n")  
            f.write(f"边特征维度: {edge_feat.shape[1] if len(edge_feat.shape) > 1 else 0}\n\n")  
              
            # 分子组成统计  
            if 'num_node' in graph and 'num_edge' in graph:  
                f.write("=== 分子组成统计 ===\n")  
                num_node = graph['num_node']  
                num_edge = graph['num_edge']  
                f.write(f"配体节点数: {num_node[0]}\n")  
                f.write(f"蛋白质节点数: {num_node[1]}\n")  
                if len(num_edge) >= 5:  
                    f.write(f"配体结构边数: {num_edge[0]}\n")  
                    f.write(f"蛋白质结构边数: {num_edge[1]}\n")  
                    f.write(f"配体-蛋白相互作用边数: {num_edge[2]}\n")  
                    f.write(f"配体空间边数: {num_edge[3]}\n")  
                    f.write(f"蛋白质空间边数: {num_edge[4]}\n")  
                f.write("\n")  
              
            # 详细边类型分析  
            if len(edge_feat.shape) > 1:  
                f.write("=== 详细边类型分析 ===\n")  
                edge_types, edge_type_stats = analyze_edge_types_detailed(edge_feat)  
                  
                # 总体统计  
                total_edges = sum(stats['count'] for stats in edge_type_stats.values())  
                f.write(f"总边数: {total_edges}\n\n")  
                  
                # 按类别分组显示  
                categories = {}  
                for edge_type, stats in edge_type_stats.items():  
                    category = stats['category']  
                    if category not in categories:  
                        categories[category] = []  
                    categories[category].append((edge_type, stats))  
                  
                for category, edges in categories.items():  
                    category_total = sum(stats['count'] for _, stats in edges)  
                    f.write(f"{category}: {category_total} 条边 ({category_total/total_edges*100:.1f}%)\n")  
                      
                    for edge_type, stats in sorted(edges, key=lambda x: x[1]['count'], reverse=True):  
                        count = stats['count']  
                        percentage = count / total_edges * 100  
                        f.write(f"  {edge_type}:\n")  
                        f.write(f"    数量: {count} 条 ({percentage:.1f}%)\n")  
                        f.write(f"    编码: {stats['code']}\n")  
                          
                        if stats['distances'] and len([d for d in stats['distances'] if d != "N/A"]) > 0:  
                            distances = [d for d in stats['distances'] if d != "N/A"]  
                            f.write(f"    距离范围: {min(distances):.3f} - {max(distances):.3f} Å\n")  
                            f.write(f"    平均距离: {np.mean(distances):.3f} Å\n")  
                            f.write(f"    距离标准差: {np.std(distances):.3f} Å\n")  
                        f.write("\n")  
                    f.write("\n")  
              
            # 3D坐标信息  
            if len(coords.shape) > 1:  
                f.write("=== 3D坐标信息 ===\n")  
                f.write(f"坐标形状: {coords.shape}\n")  
                f.write(f"坐标范围:\n")  
                f.write(f"  X: {coords[:, 0].min():.3f} - {coords[:, 0].max():.3f}\n")  
                f.write(f"  Y: {coords[:, 1].min():.3f} - {coords[:, 1].max():.3f}\n")  
                f.write(f"  Z: {coords[:, 2].min():.3f} - {coords[:, 2].max():.3f}\n\n")  
              
            # 分子相互作用评分  
            f.write("=== 分子相互作用评分 ===\n")  
            if 'rfscore' in graph:  
                rfscore = graph['rfscore']  
                f.write(f"RF-Score 维度: {len(rfscore)}\n")  
                f.write(f"RF-Score 前10个值: {rfscore[:10].tolist()}\n")  
            if 'gbscore' in graph:  
                gbscore = graph['gbscore']  
                f.write(f"GB-Score 维度: {len(gbscore)}\n")  
                f.write(f"GB-Score 前10个值: {gbscore[:10].tolist()}\n")  
            if 'ecif' in graph:  
                ecif = graph['ecif']  
                f.write(f"ECIF 维度: {len(ecif)}\n")  
                f.write(f"ECIF 非零元素数: {np.count_nonzero(ecif)}\n")  
            f.write("\n")  
  
def main():  
    if len(sys.argv) != 3:  
        print("用法: python analyze_pkl_detailed.py <pkl_file_path> <output_txt_path>")  
        sys.exit(1)  
      
    pkl_file_path = sys.argv[1]  
    output_file_path = sys.argv[2]  
      
    # 检查文件是否存在  
    if not Path(pkl_file_path).exists():  
        print(f"错误: PKL文件不存在: {pkl_file_path}")  
        sys.exit(1)  
      
    try:  
        # 加载pkl文件  
        print(f"正在加载PKL文件: {pkl_file_path}")  
        with open(pkl_file_path, 'rb') as f:  
            graphs = pickle.load(f)  
          
        print(f"成功加载 {len(graphs)} 个复合物")  
        print(f"数据类型: {type(graphs[0])}")  
          
        # 显示第一个复合物的数据结构  
        if graphs:  
            graph = graphs[0]  
            print("\n复合物数据结构:")  
            for key, value in graph.items():  
                if isinstance(value, (np.ndarray, torch.Tensor)):  
                    print(f"  {key}: {type(value).__name__} shape {value.shape}")  
                else:  
                    print(f"  {key}: {type(value).__name__} - {value}")  
              
            # 分析第一个复合物的边类型  
            edge_feat = graph.get('edge_feat', np.array([]))  
            if len(edge_feat.shape) > 1:  
                _, edge_type_stats = analyze_edge_types_detailed(edge_feat)  
                print_edge_summary(edge_type_stats)  
          
        # 分析并写入文件  
        print(f"\n正在分析并写入详细结果到: {output_file_path}")  
        write_detailed_analysis_to_file(graphs, output_file_path)  
          
        print("分析完成！")  
          
    except Exception as e:  
        print(f"处理过程中出现错误: {e}")  
        sys.exit(1)  
  
if __name__ == "__main__":  
    main()