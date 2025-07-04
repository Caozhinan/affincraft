import gemmi
import sys

if len(sys.argv) != 3:
    print("用法: python test.py input.pdb output.cif")
    sys.exit(1)

pdb_path, cif_path = sys.argv[1], sys.argv[2]

st = gemmi.read_structure(pdb_path)
doc = st.make_mmcif_document()   # 生成mmCIF文档对象
doc.write_file(cif_path)         # 写入文件

print(f"已将 {pdb_path} 转换为 {cif_path}")