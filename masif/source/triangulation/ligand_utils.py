# 导入Python的shutil模块，用于文件操作（如复制文件）
import shutil
# 导入StringIO和BytesIO，用于字符串和字节流的内存操作
from io import StringIO, BytesIO
# 导入RDKit的化学分子处理库
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
# 导入OpenBabel库用于分子格式转换
from openbabel import openbabel
# 导入ProDy用于PDB文件生物分子结构处理
import prody
# 设置ProDy的输出级别为最低（不显示日志信息）
import os  
from rdkit import Chem 
prody.confProDy(verbosity='none')


def neutralize_atoms(mol):
    # 定义SMARTS模式，匹配带正或负电荷的原子（排除某些情况）
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    # 在分子中查找所有符合SMARTS模式的原子
    at_matches = mol.GetSubstructMatches(pattern)
    # 提取所有匹配原子的索引
    at_matches_list = [y[0] for y in at_matches]
    # 如果有匹配，进行中和处理
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()    # 获取原子的形式电荷
            hcount = atom.GetTotalNumHs()   # 获取原子的氢原子数
            atom.SetFormalCharge(0)         # 电荷归零
            atom.SetNumExplicitHs(hcount - chg)  # 调整氢原子数
            atom.UpdatePropertyCache()      # 更新原子属性缓存
    # 返回一个新的分子对象（副本）
    mol_out = Chem.Mol(mol)
    return mol_out


def amide_to_single_bond(mol2_outfile):
    # 读取mol2文件，将所有amide类型的键修改为单键
    mol2_new = []
    bond_record = False
    with open(mol2_outfile, 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n')

            # 检查TRIPOS区块标记
            if line.startswith('@<TRIPOS>'):
                # 判断是否进入BOND区块
                if line.startswith('@<TRIPOS>BOND'):
                    bond_record = True
                else:
                    bond_record = False
                mol2_new.append(line)
                continue

            if bond_record:
                # 解析BOND行，格式: bond_id origin_atom_id target_atom_id bond_type
                bond_id, origin_atom_id, target_atom_id, bond_type = line.split()
                # 如果是amide键，将其类型设为单键（1）
                bond_type = '1' if bond_type == 'am' else bond_type
                # 重新拼接行
                line = '\t'.join([bond_id, origin_atom_id, target_atom_id, bond_type])

            mol2_new.append(line)

    # 覆盖写回修改后的mol2文件
    with open(mol2_outfile, 'w') as f:
        f.write("\n".join(mol2_new))

def sdf_to_mol2(sdf_file, mol2_outfile):  
    """  
    Convert SDF file to MOL2 format using OpenBabel and add required SUBSTRUCTURE section  
      
    Args:  
        sdf_file: Path to input SDF file  
        mol2_outfile: Path to output MOL2 file  
      
    Returns:  
        bool: True if conversion successful, False otherwise  
    """  
    try:  
        from openbabel import openbabel  
          
        obConversion = openbabel.OBConversion()  
        obConversion.SetInAndOutFormats("sdf", "mol2")  
          
        obmol = openbabel.OBMol()  
        if not obConversion.ReadFile(obmol, sdf_file):  
            print(f"Failed to read SDF file: {sdf_file}")  
            return False  
              
        if not obConversion.WriteFile(obmol, mol2_outfile):  
            print(f"Failed to write MOL2 file: {mol2_outfile}")  
            return False  
          
        # Add the missing @<TRIPOS>SUBSTRUCTURE section  
        add_substructure_section(mol2_outfile)  
          
        # Apply the same amide bond fix as in the original code  
        from triangulation.ligand_utils import amide_to_single_bond  
        amide_to_single_bond(mol2_outfile)  
          
        return True  
          
    except Exception as e:  
        print(f"Error during SDF to MOL2 conversion: {e}")  
        return False  
  
def add_substructure_section(mol2_file):  
    """  
    Add @<TRIPOS>SUBSTRUCTURE section to MOL2 file if missing  
    """  
    try:  
        with open(mol2_file, 'r') as f:  
            content = f.read()  
          
        # Check if SUBSTRUCTURE section already exists  
        if '@<TRIPOS>SUBSTRUCTURE' in content:  
            return  
          
        # Add the SUBSTRUCTURE section at the end  
        substructure_section = "\n@<TRIPOS>SUBSTRUCTURE\n1\tUNK\t1\tGROUP\t1 X\tUNK\n"  
          
        with open(mol2_file, 'w') as f:  
            f.write(content + substructure_section)  
              
        print(f"Added @<TRIPOS>SUBSTRUCTURE section to {mol2_file}")  
          
    except Exception as e:  
        print(f"Error adding SUBSTRUCTURE section: {e}")
    
def extract_ligand(ligand_sdf_file="ligand.sdf"):  
    """  
    直接从SDF文件中提取配体分子  
  
    Args:  
        ligand_sdf_file: 配体SDF文件路径 (默认: "ligand.sdf")  
  
    Returns:  
        RDKit分子对象，如果失败则返回None  
    """  
  
    # 检查SDF文件是否存在  
    if not os.path.exists(ligand_sdf_file):  
        print(f"DEBUG: ERROR - SDF file does not exist: {ligand_sdf_file}")  
        return None  
  
    print(f"DEBUG: Reading ligand from SDF file: {ligand_sdf_file}")  
  
    try:  
        # 直接从SDF文件读取分子  
        rdmol = Chem.MolFromMolFile(ligand_sdf_file, sanitize=True, removeHs=False)  
  
        if rdmol is not None:  
        # 为原子添加PDB残基信息，使用与MSMS输出匹配的命名  
            element_counts = {}  

            for i, atom in enumerate(rdmol.GetAtoms()):  
                info = Chem.AtomPDBResidueInfo()  

                # 使用简单的元素符号作为原子名称  
                element = atom.GetSymbol()  
                if element not in element_counts:  
                    element_counts[element] = 0  
                element_counts[element] += 1  

                # 第一个原子使用纯元素符号，后续添加数字  
                if element_counts[element] == 1:  
                    atom_name = element  # 'C', 'N', 'O' 等  
                else:  
                    atom_name = f"{element}{element_counts[element]}"  # 'C2', 'N2' 等  

                info.SetName(atom_name)  
                info.SetResidueName("UNK")  
                info.SetResidueNumber(1)  
                info.SetChainId("X")  
                atom.SetPDBResidueInfo(info)  
      
        return rdmol  
  
    except Exception as e:  
        print(f"DEBUG: Exception during SDF processing: {e}")  
        return None