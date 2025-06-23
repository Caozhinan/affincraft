#!/usr/bin/env python
# coding: utf-8

import numpy as np  # 科学计算库，主要用于数组、距离计算
import pandas as pd  # 用于表格数据处理
from os import listdir  # 用于目录遍历（本脚本未用到）
from rdkit import Chem  # RDKit 化学分子处理库
from scipy.spatial.distance import cdist  # 计算欧氏距离
from itertools import product  # 笛卡尔积，用于生成所有原子对类型
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator  # RDKit 分子描述符计算器

# === ECIF 原子类型定义 ===
# 蛋白原子的所有可能类型（精细区分化学环境）
ECIF_ProteinAtoms = [
    # 元素;显式价;重原子邻居数;氢邻居数;芳香性;是否在环
    'C;4;1;3;0;0', 'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1',
    'C;4;3;0;0;0', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
    'C;5;3;0;0;0', 'C;6;3;0;0;0', 'N;3;1;2;0;0', 'N;3;2;0;1;1',
    'N;3;2;1;0;0', 'N;3;2;1;1;1', 'N;3;3;0;0;1', 'N;4;1;2;0;0',
    'N;4;1;3;0;0', 'N;4;2;1;0;0', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
    'S;2;1;1;0;0', 'S;2;2;0;0;0'
]

# 配体分子的所有可能 ECIF 原子类型（根据 PDBbind refined set 统计）
ECIF_LigandAtoms = [
    # 省略，内容同上，详见原代码
    # ...
]

# 所有可能的蛋白-配体原子对类型，格式: 蛋白类型-配体类型
PossibleECIF = [i[0] + "-" + i[1] for i in product(ECIF_ProteinAtoms, ECIF_LigandAtoms)]

# === ELEMENTS 粗略原子类型定义（只区分元素）===
ELEMENTS_ProteinAtoms = ["C", "N", "O", "S"]
ELEMENTS_LigandAtoms = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
PossibleELEMENTS = [i[0] + "-" + i[1] for i in product(ELEMENTS_ProteinAtoms, ELEMENTS_LigandAtoms)]

# === 配体描述符列表（RDKit 支持的分子属性，适合机器学习建模）===
LigandDescriptors = [
    # 省略，内容同上，详见原代码
    # ...
]
DescCalc = MolecularDescriptorCalculator(LigandDescriptors)  # 初始化描述符计算器

# === ECIF 原子类型判定函数 ===
def GetAtomType(atom):
    """
    输入: RDKit 原子对象
    输出: ECIF 格式原子类型字符串
    """
    AtomType = [
        atom.GetSymbol(),  # 元素符号
        str(atom.GetExplicitValence()),  # 显式价
        str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),  # 重原子邻居数
        str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),  # 氢原子邻居数
        str(int(atom.GetIsAromatic())),  # 芳香性（1/0）
        str(int(atom.IsInRing())),  # 是否在环（1/0）
    ]
    return (";".join(AtomType))

# === 读取配体 SDF，输出 DataFrame，包含所有原子空间坐标与 ECIF 类型 ===
def LoadSDFasDF(SDF):
    """
    输入: 配体 SDF 文件路径
    输出: DataFrame，每行一个非氢原子，包含索引、ECIF 类型、坐标
    """
    m = Chem.MolFromMolFile(SDF, sanitize=False)  # 不消毒，直接读入
    m.UpdatePropertyCache(strict=False)  # 更新原子属性
    ECIF_atoms = []
    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H":  # 只保留非氢原子
            entry = [int(atom.GetIdx())]
            entry.append(GetAtomType(atom))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            # 保留4位小数的坐标
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            ECIF_atoms.append(entry)
    df = pd.DataFrame(ECIF_atoms)
    df.columns = ["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]
    return (df)

# === 读取蛋白质 PDB 文件，输出 DataFrame，包含所有原子空间坐标与 ECIF 类型 ===
Atom_Keys = pd.read_csv("PDB_Atom_Keys.csv", sep=",")  # PDB 原子名到 ECIF 类型的映射表

def LoadPDBasDF(PDB):
    """
    输入: 蛋白 PDB 文件路径
    输出: DataFrame，每行一个非氢原子，包含索引、ECIF 类型、坐标
    """
    ECIF_atoms = []
    with open(PDB) as f:
        for i in f:
            if i[:4] == "ATOM":
                # 只保留非氢原子，PDB 原子名有特殊规则
                atom_name = i[12:16].replace(" ", "")
                try:
                    # 判断方式参考 ECIF 脚本（非氢原子名）
                    if (len(atom_name) < 4 and atom_name[0] != "H") or (
                            len(atom_name) == 4 and atom_name[1] != "H" and atom_name[0] != "H"):
                        atom_index_str = i[6:11].strip()
                        try:
                            atom_index = int(atom_index_str)
                        except ValueError:
                            print(f"Warning: skip invalid atom serial '{atom_index_str}' in file {PDB}")
                            continue
                        ECIF_atoms.append([
                            atom_index,
                            i[17:20] + "-" + atom_name,  # 残基名-原子名
                            i[30:38].strip(),  # X 坐标
                            i[38:46].strip(),  # Y 坐标
                            i[46:54].strip()   # Z 坐标
                        ])
                except Exception as e:
                    print(f"Warning: skip line due to error: {e}")
                    continue
    df = pd.DataFrame(ECIF_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    # 坐标全部转为 float
    for col in ["X", "Y", "Z"]:
        df[col] = df[col].astype(float)
    # 合并 ECIF 类型
    df = df.merge(Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    for col in ["X", "Y", "Z"]:
        df[col] = df[col].astype(float)
    if list(df["ECIF_ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
    return df

# === 获取蛋白-配体所有空间邻近原子对 ===
def GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=6.0):
    """
    输入: 蛋白 pdb 路径、配体 sdf 路径、距离阈值
    输出: DataFrame, 每行为一个蛋白-配体原子对（含 ECIF 类型、ELEMENTS 类型、距离）
    """
    # 读取原子信息
    Target = LoadPDBasDF(PDB_protein)
    Ligand = LoadSDFasDF(SDF_ligand)

    # 只保留距离配体区域最近的蛋白原子（先用立方体过滤以加速）
    for i in ["X", "Y", "Z"]:
        Target = Target[Target[i] < float(Ligand[i].max()) + distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min()) - distance_cutoff]

    # 确保全部为 float 类型
    Target[["X", "Y", "Z"]] = Target[["X", "Y", "Z"]].astype(float)
    Ligand[["X", "Y", "Z"]] = Ligand[["X", "Y", "Z"]].astype(float)

    # 所有可能的原子类型组合
    Pairs = list(product(Target["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
    Pairs = [x[0] + "-" + x[1] for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])

    # 计算所有蛋白原子和配体原子的空间距离，展平成一维
    Distances = cdist(Target[["X", "Y", "Z"]], Ligand[["X", "Y", "Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0] * Distances.shape[1], 1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

    Pairs = pd.concat([Pairs, Distances], axis=1)
    # 只保留距离小于指定 cutoff 的原子对
    Pairs = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
    # ECIF pair -> elements pair（只保留元素符号）
    Pairs["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0] + "-" + x.split("-")[1].split(";")[0] for x in Pairs["ECIF_PAIR"]]
    return Pairs

# === 计算 ECIF 特征向量 ===
def GetECIF(PDB_protein, SDF_ligand, distance_cutoff=6.0):
    """
    输入: 蛋白 pdb 路径、配体 sdf 路径、距离阈值
    输出: ECIF 特征向量（每种原子对类型出现次数）
    """
    Pairs = GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=distance_cutoff)
    ECIF = [list(Pairs["ECIF_PAIR"]).count(x) for x in PossibleECIF]
    return ECIF

# === 计算 ELEMENTS 特征向量（元素级别）===
def GetELEMENTS(PDB_protein, SDF_ligand, distance_cutoff=6.0):
    """
    输入: 蛋白 pdb 路径、配体 sdf 路径、距离阈值
    输出: ELEMENTS 特征向量（每种元素对类型出现次数）
    """
    Pairs = GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=distance_cutoff)
    ELEMENTS = [list(Pairs["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS]
    return ELEMENTS

# === 计算配体 RDKit 分子描述符 ===
def GetRDKitDescriptors(SDF):
    """
    输入: 配体 SDF 路径
    输出: RDKit 分子描述符向量
    """
    mol = Chem.MolFromMolFile(SDF, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)  # 计算环信息（部分描述符依赖）
    return DescCalc.CalcDescriptors(mol)