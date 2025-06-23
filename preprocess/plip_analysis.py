#!/usr/bin/env python3  
"""  
PLIP相互作用分析脚本  
处理蛋白质口袋文件和小分子SDF文件  
"""  
  
import sys  
import os  
from plip.structure.preparation import PDBComplex  
  
def merge_protein_ligand_with_pymol(protein_file, ligand_file, output_file="complex.pdb"):  
    """  
    使用PyMOL合并蛋白质和配体文件  
      
    Args:  
        protein_file: 蛋白质PDB文件路径  
        ligand_file: 配体SDF文件路径  
        output_file: 输出复合物文件名  
      
    Returns:  
        bool: 是否成功合并  
    """  
    try:  
        import pymol  
        from pymol import cmd  
          
        # 初始化PyMOL  
        pymol.finish_launching(['pymol', '-c'])  # -c表示命令行模式  
        cmd.reinitialize()  
          
        print(f"使用PyMOL合并文件...")  
        print(f"  蛋白质文件: {protein_file}")  
        print(f"  配体文件: {ligand_file}")  
          
        # 加载蛋白质文件  
        cmd.load(protein_file, "protein")  
        print("  ✓ 蛋白质文件已加载")  
          
        # 加载配体文件  
        cmd.load(ligand_file, "ligand")  
        print("  ✓ 配体文件已加载")  
          
        # 创建复合物选择  
        cmd.create("complex", "protein or ligand")  
          
        # 保存复合物  
        cmd.save(output_file, "complex")  
        print(f"  ✓ 复合物已保存为: {output_file}")  
          
        # 清理PyMOL会话  
        cmd.reinitialize()  
          
        return True  
          
    except ImportError:  
        print("错误: 未找到PyMOL模块")  
        print("请安装PyMOL: pip install pymol-open-source")  
        return False  
    except Exception as e:  
        print(f"PyMOL合并过程中出现错误: {e}")  
        return False  
  
def analyze_protein_ligand_interactions(protein_file, ligand_sdf_file):  
    """  
    分析蛋白质-配体相互作用  
      
    Args:  
        protein_file: 蛋白质口袋PDB文件路径  
        ligand_sdf_file: 小分子SDF文件路径  
      
    Returns:  
        interaction_sets字典  
    """  
      
    # 检查文件是否存在  
    if not os.path.exists(protein_file):  
        raise FileNotFoundError(f"蛋白质文件不存在: {protein_file}")  
    if not os.path.exists(ligand_sdf_file):  
        raise FileNotFoundError(f"配体文件不存在: {ligand_sdf_file}")  
      
    complex_file = "complex.pdb"  
      
    # 检查是否已有复合物文件  
    if os.path.exists(complex_file):  
        print(f"发现已存在的复合物文件: {complex_file}")  
        use_existing = input("是否使用现有文件？(y/n): ").lower().strip()  
        if use_existing != 'y':  
            print("重新生成复合物文件...")  
            if not merge_protein_ligand_with_pymol(protein_file, ligand_sdf_file, complex_file):  
                return {}  
    else:  
        print("未找到复合物文件，使用PyMOL自动生成...")  
        if not merge_protein_ligand_with_pymol(protein_file, ligand_sdf_file, complex_file):  
            return {}  
      
    try:  
        # 创建PDBComplex实例  
        my_mol = PDBComplex()  
          
        # 加载复合物PDB文件  
        my_mol.load_pdb(complex_file)  
          
        # 显示结构信息  
        print(f"\n加载的结构: {my_mol}")  
        print(f"发现的配体数量: {len(my_mol.ligands)}")  
          
        if len(my_mol.ligands) == 0:  
            print("错误: 在复合物文件中未发现配体")  
            print("请检查配体文件格式或PyMOL合并过程")  
            return {}  
          
        # 分析所有相互作用  
        print("开始分析相互作用...")  
        my_mol.analyze()  
          
        # 返回interaction_sets字典  
        return my_mol.interaction_sets  
          
    except Exception as e:  
        print(f"分析过程中出现错误: {e}")  
        return {}  
  
def print_interaction_summary(interaction_sets):  
    """打印相互作用摘要"""  
      
    if not interaction_sets:  
        print("没有找到相互作用数据")  
        return  
      
    print("\n=== 相互作用分析结果 ===")  
      
    for bsid, interactions in interaction_sets.items():  
        print(f"\n结合位点 ID: {bsid}")  
          
        # 统计各种相互作用  
        hbonds = interactions.hbonds_ldon + interactions.hbonds_pdon  
        hydrophobic = interactions.hydrophobic_contacts  
        pistacking = interactions.pistacking  
        pication = interactions.pication_laro + interactions.pication_paro  
        saltbridge = interactions.saltbridge_lneg + interactions.saltbridge_pneg  
        waterbridges = interactions.water_bridges  
        halogenbonds = interactions.halogen_bonds  
        metalcomplexes = interactions.metal_complexes  
          
        print(f"  氢键: {len(hbonds)}")  
        print(f"  疏水相互作用: {len(hydrophobic)}")  
        print(f"  π-π堆积: {len(pistacking)}")  
        print(f"  π-阳离子相互作用: {len(pication)}")  
        print(f"  盐桥: {len(saltbridge)}")  
        print(f"  水桥: {len(waterbridges)}")  
        print(f"  卤键: {len(halogenbonds)}")  
        print(f"  金属配位: {len(metalcomplexes)}")  
          
        # 显示参与相互作用的残基  
        if hasattr(interactions, 'interacting_res'):  
            print(f"  参与相互作用的残基数: {len(interactions.interacting_res)}")  
  
def main():  
    """主函数"""  
      
    # 如果没有提供命令行参数，使用默认文件名  
    if len(sys.argv) == 1:  
        protein_file = "protein_pocket.pdb"  
        ligand_sdf_file = "ligand.sdf"  
        print("使用默认文件名:")  
        print(f"  蛋白质文件: {protein_file}")  
        print(f"  配体文件: {ligand_sdf_file}")  
    elif len(sys.argv) == 3:  
        protein_file = sys.argv[1]  
        ligand_sdf_file = sys.argv[2]  
    else:  
        print("用法: python plip.py [<蛋白质PDB文件> <配体SDF文件>]")  
        print("如果不提供参数，将使用当前目录下的 protein_pocket.pdb 和 ligand.sdf")  
        sys.exit(1)  
      
    print(f"蛋白质文件: {protein_file}")  
    print(f"配体文件: {ligand_sdf_file}")  
      
    # 分析相互作用  
    interaction_sets = analyze_protein_ligand_interactions(protein_file, ligand_sdf_file)  
      
    # 打印结果摘要  
    print_interaction_summary(interaction_sets)  
      
    # 返回interaction_sets供进一步使用  
    return interaction_sets  
  
if __name__ == "__main__":  
    interaction_sets = main()  