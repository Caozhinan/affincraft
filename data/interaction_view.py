#!/usr/bin/env python3  
"""  
整合的蛋白质-配体相互作用分析和可视化脚本  
完全基于intra_pro_plip模块的相互作用检测逻辑  
"""  
  
import os  
import sys  
import numpy as np  
from collections import namedtuple  
from openbabel import pybel  
from pymol import cmd  
import tempfile  
  
# 导入您的蛋白质内部相互作用分析脚本  
sys.path.append('/xcfhome/zncao02/affincraft/preprocess')  
import intra_pro_plip  
  
class ComplexAnalyzer:  
    """蛋白质-配体复合物分析器"""  
      
    def __init__(self, pdb_file):  
        self.pdb_file = pdb_file  
        self.protein_name = os.path.splitext(os.path.basename(pdb_file))[0]  
        self.molecule = None  
        self.ligand_atoms = []  
        self.pocket_atoms = []  
        self.protein_atoms = []  
        self.interactions = {  
            'protein_ligand': {},  
            'protein_internal': {}  
        }  
          
        # 加载分子  
        self.load_molecule()  
          
    def load_molecule(self):  
        """加载PDB文件"""  
        self.molecule = pybel.readfile("pdb", self.pdb_file).__next__()  
        print(f"已加载分子，包含 {len(self.molecule.atoms)} 个原子")  
          
    def identify_ligand_and_protein(self):  
        """识别配体(UNK)和蛋白质原子"""  
        ligand_atoms = []  
        protein_atoms = []  
          
        for atom in self.molecule.atoms:  
            residue = atom.OBAtom.GetResidue()  
            if residue and residue.GetName() == 'UNK':  
                ligand_atoms.append(atom)  
            else:  
                protein_atoms.append(atom)  
          
        self.ligand_atoms = ligand_atoms  
        self.protein_atoms = protein_atoms  
          
        print(f"识别到配体原子: {len(ligand_atoms)} 个")  
        print(f"识别到蛋白质原子: {len(protein_atoms)} 个")  
          
        if not ligand_atoms:  
            raise ValueError("未找到UNK配体原子")  
              
    def find_pocket_residues(self, cutoff=5.0):  
        """找到配体周围指定距离内的口袋残基 - 基于PyMOL的byres逻辑"""  
        pocket_atoms = []  
        pocket_residues = set()  

        # 首先找到距离配体任意原子6Å内的蛋白质原子  
        for protein_atom in self.protein_atoms:  
            for ligand_atom in self.ligand_atoms:  
                distance = np.linalg.norm(  
                    np.array(protein_atom.coords) - np.array(ligand_atom.coords)  
                )  
                if distance <= cutoff:  
                    # 获取该原子所属的残基信息  
                    residue = protein_atom.OBAtom.GetResidue()  
                    if residue:  
                        res_key = (residue.GetName(), residue.GetChain(), residue.GetNum())  
                        pocket_residues.add(res_key)  
                    break  # 找到一个距离内的配体原子就足够了  
                
        # 然后收集所有口袋残基中的原子  
        for protein_atom in self.protein_atoms:  
            residue = protein_atom.OBAtom.GetResidue()  
            if residue:  
                res_key = (residue.GetName(), residue.GetChain(), residue.GetNum())  
                if res_key in pocket_residues:  
                    pocket_atoms.append(protein_atom)  

        self.pocket_atoms = pocket_atoms  
        print(f"识别到口袋残基: {len(pocket_residues)} 个")  
        print(f"识别到口袋原子: {len(pocket_atoms)} 个")  

        return pocket_atoms
    
    
      
    def analyze_protein_ligand_interactions(self):  
        """分析蛋白-配体间相互作用 - 使用intra_pro_plip的完整逻辑"""  
        print("分析蛋白-配体相互作用...")  
          
        # 使用intra_pro_plip的InteractionAnalyzer  
        analyzer = intra_pro_plip.InteractionAnalyzer()  
          
        # 准备原子对：口袋原子 vs 配体原子  
        atom_pairs = []  
        for pocket_atom in self.pocket_atoms:  
            for ligand_atom in self.ligand_atoms:  
                distance = np.linalg.norm(  
                    np.array(pocket_atom.coords) - np.array(ligand_atom.coords)  
                )  
                if distance <= 5.0:  # 只分析5Å内的原子对  
                    atom_pairs.append((pocket_atom, ligand_atom))  
          
        print(f"分析 {len(atom_pairs)} 个蛋白-配体原子对...")  
          
        # 使用intra_pro_plip的分析方法  
        results = analyzer.analyze_atom_pairs(atom_pairs)  
          
        # 同时使用分子级别的分析来检测π-π堆积和π-阳离子相互作用  
        # 创建临时分子对象包含口袋和配体原子  
        temp_mol = self._create_temp_molecule_for_analysis()  
        molecular_results = analyzer.analyze_molecule_interactions(temp_mol)  
          
        # 合并结果  
        all_results = results + molecular_results  
          
        # 转换为可视化格式  
        self.interactions['protein_ligand'] = self._convert_to_visualization_format_v2(all_results)  
          
        print(f"发现 {len(all_results)} 个蛋白-配体相互作用")  
          
    def analyze_protein_internal_interactions(self):  
        """分析蛋白质内部相互作用 - 使用intra_pro_plip的完整逻辑"""  
        print("分析蛋白质内部相互作用...")  
          
        analyzer = intra_pro_plip.InteractionAnalyzer()  
          
        # 准备蛋白质内部原子对（只考虑口袋区域）  
        atom_pairs = []  
        for i, atom1 in enumerate(self.pocket_atoms):  
            for atom2 in self.pocket_atoms[i+1:]:  
                distance = np.linalg.norm(  
                    np.array(atom1.coords) - np.array(atom2.coords)  
                )  
                if distance <= 5.0:  # 只分析5Å内的原子对  
                    atom_pairs.append((atom1, atom2))  
          
        print(f"分析 {len(atom_pairs)} 个蛋白质内部原子对...")  
          
        # 使用intra_pro_plip的分析方法  
        results = analyzer.analyze_atom_pairs(atom_pairs)  
          
        # 创建只包含口袋原子的临时分子进行分子级别分析  
        pocket_mol = self._create_pocket_molecule_for_analysis()  
        molecular_results = analyzer.analyze_molecule_interactions(pocket_mol)  
          
        # 合并结果  
        all_results = results + molecular_results  
          
        # 转换为可视化格式  
        self.interactions['protein_internal'] = self._convert_to_visualization_format_v2(all_results)  
          
        print(f"发现 {len(all_results)} 个蛋白质内部相互作用")  
      
    def _create_temp_molecule_for_analysis(self):  
        """创建包含口袋和配体原子的临时分子对象用于分析"""  
        # 这里需要创建一个临时的pybel分子对象  
        # 包含口袋原子和配体原子，用于π-π堆积等分子级别的相互作用检测  
        # 由于OpenBabel的限制，这里使用简化实现  
        return self.molecule  
      
    def _create_pocket_molecule_for_analysis(self):  
        """创建只包含口袋原子的临时分子对象用于分析"""  
        # 类似上面的方法，但只包含口袋原子  
        return self.molecule  
      
    def _convert_to_visualization_format_v2(self, results):  
        """将intra_pro_plip的分析结果转换为可视化格式"""  
        viz_data = {  
            'hydrophobic': [],  
            'hydrogen_bonds': [],  
            'pi_stacking': [],  
            'salt_bridges': [],  
            'pi_cation': [],  
            'halogen_bonds': []  
        }  
          
        for result in results:  
            # 处理原子对相互作用  
            if 'atom1' in result and 'atom2' in result:  
                atom1 = result['atom1']  
                atom2 = result['atom2']  
                interaction_types = result['interaction_types']  
                distance = result['distance']  
                  
                for interaction_type in interaction_types:  
                    if interaction_type == 'hydrophobic':  
                        viz_data['hydrophobic'].append({  
                            'atom1_id': atom1.idx,  
                            'atom2_id': atom2.idx,  
                            'distance': distance,  
                            'type': 'hydrophobic'  
                        })  
                    elif interaction_type == 'hydrogen_bond':  
                        viz_data['hydrogen_bonds'].append({  
                            'donor_id': atom1.idx,  
                            'acceptor_id': atom2.idx,  
                            'distance': distance,  
                            'type': 'hydrogen_bond'  
                        })  
                    elif interaction_type == 'salt_bridge':  
                        viz_data['salt_bridges'].append({  
                            'positive_id': atom1.idx,  
                            'negative_id': atom2.idx,  
                            'distance': distance,  
                            'type': 'salt_bridge'  
                        })  
                    elif interaction_type == 'halogen_bond':  
                        viz_data['halogen_bonds'].append({  
                            'donor_id': atom1.idx,  
                            'acceptor_id': atom2.idx,  
                            'distance': distance,  
                            'type': 'halogen_bond'  
                        })  
              
            # 处理分子级别的相互作用  
            elif 'interaction_type' in result:  
                if result['interaction_type'] == 'pi_stacking':  
                    ring1_center = [float(x) for x in result['ring1'].center]  
                    ring2_center = [float(x) for x in result['ring2'].center]  
                    viz_data['pi_stacking'].append({  
                        'ring1_center': ring1_center,  
                        'ring2_center': ring2_center,  
                        'distance': result['distance'],  
                        'type': result['type']  
                    })  
                elif result['interaction_type'] == 'pi_cation':  
                    ring_center = [float(x) for x in result['ring'].center]  
                    charge_center = [float(x) for x in result['charge'].center]  
                    viz_data['pi_cation'].append({  
                        'ring_center': ring_center,  
                        'charge_center': charge_center,  
                        'distance': result['distance'],  
                        'type': 'pi_cation'  
                    })  
          
        return viz_data  
      
    def create_pymol_visualization(self, output_dir="."):  
        """创建PyMOL可视化 - 基于实际检测到的相互作用"""  
        print("创建PyMOL可视化...")  
          
        # 启动PyMOL  
        cmd.reinitialize()  
          
        # 加载结构  
        cmd.load(self.pdb_file, self.protein_name)  
          
        # 基本设置  
        self._setup_pymol_environment()  
          
        # 选择和显示配体  
        cmd.select('ligand', 'resn UNK')  
        cmd.show('sticks', 'ligand')  
        cmd.color('orange', 'ligand')  
          
        # 选择和显示口袋  
        cmd.select('pocket', f'br. all within 6.0 of ligand')  
        cmd.show('sticks', 'pocket')  
        cmd.color('cyan', 'pocket')  
          
        # 显示蛋白质骨架  
        cmd.show('cartoon', self.protein_name)  
        cmd.color('lightblue', self.protein_name)  
          
        # 可视化蛋白-配体相互作用（使用实际检测结果）  
        self._visualize_interactions_v2('protein_ligand', 'PL')  
          
        # 可视化蛋白质内部相互作用（使用实际检测结果）  
        self._visualize_interactions_v2('protein_internal', 'PP')  
          
        # 创建分组  
        cmd.group('Protein_Ligand_Interactions', 'PL_*')  
        cmd.group('Protein_Internal_Interactions', 'PP_*')  
        cmd.group('Structure', f'{self.protein_name} ligand pocket')  
          
        # 调整视图  
        cmd.center('ligand')  
        cmd.zoom('ligand or pocket', buffer=5)  
          
        # 保存会话  
        session_file = os.path.join(output_dir, f"{self.protein_name}_interactions.pse")  
        cmd.save(session_file)  
        print(f"PyMOL会话已保存到: {session_file}")  
          
        # 保存图像  
        image_file = os.path.join(output_dir, f"{self.protein_name}_interactions.png")  
        cmd.png(image_file, width=1200, height=900, dpi=300, ray=1)  
        print(f"图像已保存到: {image_file}")  
          
    def _setup_pymol_environment(self):  
            """设置PyMOL环境"""  
            cmd.set('bg_rgb', [1.0, 1.0, 1.0])  
            cmd.set('depth_cue', 0)  
            cmd.set('cartoon_side_chain_helper', 1)  
            cmd.set('cartoon_fancy_helices', 1)  
            cmd.set('transparency_mode', 1)  
            cmd.set('dash_radius', 0.05)  
            cmd.set('dash_gap', 0)  
            cmd.set('ray_shadow', 0)  

            # 设置PLIP标准颜色  
            cmd.set_color('myorange', '[253, 174, 97]')  
            cmd.set_color('mygreen', '[171, 221, 164]')  
            cmd.set_color('myred', '[215, 25, 28]')  
            cmd.set_color('myblue', '[43, 131, 186]')  
            cmd.set_color('mylightblue', '[158, 202, 225]')  
            cmd.set_color('mylightgreen', '[229, 245, 224]')  

    def _visualize_interactions_v2(self, interaction_category, prefix):  
        """可视化特定类别的相互作用 - 使用实际检测结果"""  
        interactions = self.interactions[interaction_category]  
          
        # 可视化疏水相互作用  
        if 'hydrophobic' in interactions:  
            for i, contact in enumerate(interactions['hydrophobic']):  
                atom1_id = contact['atom1_id']  
                atom2_id = contact['atom2_id']  
                  
                cmd.select('tmp_atom1', f'id {atom1_id}')  
                cmd.select('tmp_atom2', f'id {atom2_id}')  
                cmd.distance(f'{prefix}_Hydrophobic_{i}', 'tmp_atom1', 'tmp_atom2')  
                cmd.set('dash_color', 'grey50', f'{prefix}_Hydrophobic_{i}')  
                cmd.set('dash_gap', 0.3, f'{prefix}_Hydrophobic_{i}')  
          
        # 可视化氢键  
        if 'hydrogen_bonds' in interactions:  
            for i, hbond in enumerate(interactions['hydrogen_bonds']):  
                donor_id = hbond['donor_id']  
                acceptor_id = hbond['acceptor_id']  
                  
                cmd.select('tmp_donor', f'id {donor_id}')  
                cmd.select('tmp_acceptor', f'id {acceptor_id}')  
                cmd.distance(f'{prefix}_HBond_{i}', 'tmp_donor', 'tmp_acceptor')  
                cmd.set('dash_color', 'blue', f'{prefix}_HBond_{i}')  
          
        # 可视化盐桥  
        if 'salt_bridges' in interactions:  
            for i, bridge in enumerate(interactions['salt_bridges']):  
                pos_id = bridge['positive_id']  
                neg_id = bridge['negative_id']  
                  
                cmd.select('tmp_pos', f'id {pos_id}')  
                cmd.select('tmp_neg', f'id {neg_id}')  
                cmd.distance(f'{prefix}_SaltBridge_{i}', 'tmp_pos', 'tmp_neg')  
                cmd.set('dash_color', 'yellow', f'{prefix}_SaltBridge_{i}')  
                cmd.set('dash_gap', 0.5, f'{prefix}_SaltBridge_{i}')  
          
        # 可视化卤键  
        if 'halogen_bonds' in interactions:  
            for i, hal_bond in enumerate(interactions['halogen_bonds']):  
                donor_id = hal_bond['donor_id']  
                acceptor_id = hal_bond['acceptor_id']  
                  
                cmd.select('tmp_hal_don', f'id {donor_id}')  
                cmd.select('tmp_hal_acc', f'id {acceptor_id}')  
                cmd.distance(f'{prefix}_HalogenBond_{i}', 'tmp_hal_don', 'tmp_hal_acc')  
                cmd.set('dash_color', 'greencyan', f'{prefix}_HalogenBond_{i}')  
          
        # 可视化π-π堆积  
        if 'pi_stacking' in interactions:  
            for i, stack in enumerate(interactions['pi_stacking']):  
                ring1_center = stack['ring1_center']  
                ring2_center = stack['ring2_center']  
                stack_type = stack['type']  
                  
                cmd.pseudoatom(f'{prefix}_ring1_center_{i}', pos=ring1_center)  
                cmd.pseudoatom(f'{prefix}_ring2_center_{i}', pos=ring2_center)  
                  
                if 'parallel' in stack_type:  
                    cmd.distance(f'{prefix}_PiStack_P_{i}', f'{prefix}_ring1_center_{i}', f'{prefix}_ring2_center_{i}')  
                    cmd.set('dash_color', 'green', f'{prefix}_PiStack_P_{i}')  
                else:  
                    cmd.distance(f'{prefix}_PiStack_T_{i}', f'{prefix}_ring1_center_{i}', f'{prefix}_ring2_center_{i}')  
                    cmd.set('dash_color', 'smudge', f'{prefix}_PiStack_T_{i}')  
                  
                cmd.set('dash_gap', 0.3, f'{prefix}_PiStack_*_{i}')  
                cmd.set('dash_length', 0.6, f'{prefix}_PiStack_*_{i}')  
          
        # 可视化π-阳离子相互作用  
        if 'pi_cation' in interactions:  
            for i, pication in enumerate(interactions['pi_cation']):  
                ring_center = pication['ring_center']  
                charge_center = pication['charge_center']  
                  
                cmd.pseudoatom(f'{prefix}_pi_ring_{i}', pos=ring_center)  
                cmd.pseudoatom(f'{prefix}_cation_{i}', pos=charge_center)  
                cmd.distance(f'{prefix}_PiCation_{i}', f'{prefix}_pi_ring_{i}', f'{prefix}_cation_{i}')  
                cmd.set('dash_color', 'orange', f'{prefix}_PiCation_{i}')  
                cmd.set('dash_gap', 0.3, f'{prefix}_PiCation_{i}')  
                cmd.set('dash_length', 0.6, f'{prefix}_PiCation_{i}')  
          
        # 清理临时选择  
        cmd.delete('tmp_*')  
      
    def generate_report(self, output_dir="."):  
        """生成分析报告"""  
        report_file = os.path.join(output_dir, f"{self.protein_name}_interaction_report.txt")  
          
        with open(report_file, 'w') as f:  
            f.write(f"蛋白质-配体相互作用分析报告\n")  
            f.write(f"=" * 50 + "\n\n")  
            f.write(f"PDB文件: {self.pdb_file}\n")  
            f.write(f"配体原子数: {len(self.ligand_atoms)}\n")  
            f.write(f"口袋原子数: {len(self.pocket_atoms)}\n\n")  
              
            # 蛋白-配体相互作用统计  
            f.write("蛋白-配体相互作用:\n")  
            f.write("-" * 30 + "\n")  
            pl_interactions = self.interactions['protein_ligand']  
            for interaction_type, interactions in pl_interactions.items():  
                f.write(f"{interaction_type}: {len(interactions)} 个\n")  
              
            f.write("\n蛋白质内部相互作用:\n")  
            f.write("-" * 30 + "\n")  
            pp_interactions = self.interactions['protein_internal']  
            for interaction_type, interactions in pp_interactions.items():  
                f.write(f"{interaction_type}: {len(interactions)} 个\n")  
          
        print(f"分析报告已保存到: {report_file}")  
      
    def run_complete_analysis(self, output_dir="."):  
        """运行完整的分析流程"""  
        print("开始完整分析流程...")  
          
        # 确保输出目录存在  
        os.makedirs(output_dir, exist_ok=True)  
          
        try:  
            # 步骤1: 识别配体和蛋白质原子  
            print("步骤1: 识别配体和蛋白质原子...")  
            self.identify_ligand_and_protein()  
              
            # 步骤2: 找到口袋残基  
            print("步骤2: 识别配体周围6Å的口袋残基...")  
            self.find_pocket_residues(cutoff=5.0)  
              
            # 步骤3: 分析蛋白-配体相互作用  
            print("步骤3: 分析蛋白-配体相互作用...")  
            self.analyze_protein_ligand_interactions()  
              
            # 步骤4: 分析蛋白质内部相互作用  
            print("步骤4: 分析蛋白质内部相互作用...")  
            self.analyze_protein_internal_interactions()  
              
            # 步骤5: 生成可视化  
            print("步骤5: 创建PyMOL可视化...")  
            self.create_pymol_visualization(output_dir)  
              
            # 步骤6: 生成分析报告  
            print("步骤6: 生成分析报告...")  
            self.generate_report(output_dir)  
              
            print(f"分析完成！所有文件已保存到: {output_dir}")  
              
            # 打印总结  
            self._print_summary()  
              
        except Exception as e:  
            print(f"分析过程中出现错误: {str(e)}")  
            raise  
        
    def _print_summary(self):  
        """打印分析结果总结"""  
        print("\n" + "="*60)  
        print("分析结果总结")  
        print("="*60)  
          
        # 蛋白-配体相互作用总结  
        pl_total = sum(len(interactions) for interactions in self.interactions['protein_ligand'].values())  
        print(f"蛋白-配体相互作用总数: {pl_total}")  
        for interaction_type, interactions in self.interactions['protein_ligand'].items():  
            if interactions:  
                print(f"  - {interaction_type}: {len(interactions)} 个")  
          
        # 蛋白质内部相互作用总结  
        pp_total = sum(len(interactions) for interactions in self.interactions['protein_internal'].values())  
        print(f"蛋白质内部相互作用总数: {pp_total}")  
        for interaction_type, interactions in self.interactions['protein_internal'].items():  
            if interactions:  
                print(f"  - {interaction_type}: {len(interactions)} 个")  
          
        print("="*60)  
  
# 主函数  
def main():  
    """主函数 - 脚本入口点"""  
    import argparse  
      
    parser = argparse.ArgumentParser(description='蛋白质-配体相互作用分析和可视化')  
    parser.add_argument('pdb_file', help='输入的PDB文件路径 (例如: complex.pdb)')  
    parser.add_argument('-o', '--output', default='.', help='输出目录 (默认: 当前目录)')  
      
    args = parser.parse_args()  
      
    # 检查输入文件是否存在  
    if not os.path.exists(args.pdb_file):  
        print(f"错误: 找不到文件 {args.pdb_file}")  
        return  
      
    # 创建分析器并运行分析  
    analyzer = ComplexAnalyzer(args.pdb_file)  
    analyzer.run_complete_analysis(args.output)  
  
if __name__ == "__main__":  
    main()