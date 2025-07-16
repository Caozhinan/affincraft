#!/bin/bash

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <csv_file>"
    echo "CSV format: receptor,ligand,name,pk,rmsd"
    exit 1
fi

CSV_FILE="$1"

# 检查CSV文件是否存在
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file $CSV_FILE not found"
    exit 1
fi

# 检测蛋白质链的函数
detect_protein_chain() {
    local pdb_file="$1"
    # 查找标准氨基酸的链（排除配体链X），按原子数排序取最多的
    local protein_chain
    protein_chain=$(grep "^ATOM" "$pdb_file" | awk '{print $5}' | grep -v "X" | sort | uniq -c | sort -nr | head -1 | awk '{print $2}')
    # 如果没有找到，尝试查找所有ATOM记录的链
    if [ -z "$protein_chain" ]; then
        protein_chain=$(grep "^ATOM" "$pdb_file" | awk '{print $5}' | sort | uniq -c | sort -nr | head -1 | awk '{print $2}')
    fi
    echo "$protein_chain"
}

# 跳过CSV头部，处理每一行
tail -n +2 "$CSV_FILE" | while IFS=',' read -r receptor ligand name pk rmsd; do
    echo "Processing: $name"

    # 获取目标目录
    target_dir=$(dirname "$receptor")
    output_dir="$target_dir/output"
    mkdir -p "$output_dir"

    echo "Target directory: $target_dir"
    echo "Output directory: $output_dir"

    # 创建临时CSV文件用于单个复合物
    temp_csv="$output_dir/temp_input.csv"
    echo "receptor,ligand,name,pk,rmsd" > "$temp_csv"
    echo "$receptor,$ligand,$name,$pk,$rmsd" >> "$temp_csv"

    # 指定PKL输出文件路径
    pkl_output="$output_dir/${name}_features.pkl"

    # Step 1: 预处理 - 生成基础PKL文件
    echo "Step 1: Running custom_input.py..."
    python /xcfhome/zncao02/affincraft/preprocess/custom_input.py \
        "$temp_csv" \
        "$pkl_output"
    if [ $? -ne 0 ]; then
        echo "Error in custom_input.py for $name"
        rm -f "$temp_csv"
        continue
    fi

    rm -f "$temp_csv"

    # 检查PKL文件是否生成
    if [ ! -f "$pkl_output" ]; then
        echo "Error: PKL file not generated: $pkl_output"
        continue
    fi
    echo "Generated PKL file: $pkl_output"

    # 自动检测蛋白质链
    complex_pdb="$target_dir/complex.pdb"
    if [ ! -f "$complex_pdb" ]; then
        echo "Error: Complex PDB file not found: $complex_pdb"
        continue
    fi

    protein_chain=$(detect_protein_chain "$complex_pdb")
    if [ -z "$protein_chain" ]; then
        echo "Error: Could not detect protein chain in $complex_pdb"
        continue
    fi

    echo "Detected protein chain: $protein_chain"

    # Step 2: 网格和表面特征生成
    echo "Step 2: Running meshfeatureGen.py..."
    python /xcfhome/zncao02/affincraft/masif/meshfeatureGen.py \
        --pdb_file "$complex_pdb" \
        --chain_id "$protein_chain" \
        --ligand_code "UNK" \
        --ligand_chain "X" \
        --sdf_file "$ligand" \
        --output_dir "$output_dir"
    if [ $? -ne 0 ]; then
        echo "Error in meshfeatureGen.py for $name"
        continue
    fi

    # 动态构建文件路径（基于检测到的链ID）
    ply_file="$output_dir/surfaces/complex_${protein_chain}.ply"
    precomputed_dir="$output_dir/precomputed/complex_${protein_chain}"
    ppi_pair_id="complex_${protein_chain}"

    # 检查PLY文件是否生成
    if [ ! -f "$ply_file" ]; then
        echo "Error: PLY file not generated: $ply_file"
        continue
    fi

    # Step 3: 特征预计算
    echo "Step 3: Running feature_precompute.py..."
    python /xcfhome/zncao02/affincraft/masif/feature_precompute.py \
        --ply_file "$ply_file" \
        --output_dir "$output_dir" \
        --masif_app "masif_site"
    if [ $? -ne 0 ]; then
        echo "Error in feature_precompute.py for $name"
        continue
    fi

    if [ ! -d "$precomputed_dir" ]; then
        echo "Error: Precomputed directory not generated: $precomputed_dir"
        continue
    fi

    # Step 4: 指纹生成
    echo "Step 4: Running fingerprint_gen.py..."
    python /xcfhome/zncao02/affincraft/masif/fingerprint_gen.py \
        --precomputed_dir "$precomputed_dir" \
        --output_dir "$output_dir" \
        --ppi_pair_id "$ppi_pair_id" \
        --custom_params_file "masif.source.masif_ppi_search.nn_models.sc05.all_feat.custom_params"
    if [ $? -ne 0 ]; then
        echo "Error in fingerprint_gen.py for $name"
        continue
    fi

    # Step 5: 整合MaSIF特征到现有PKL文件
# Step 5: 整合MaSIF特征到现有PKL文件
    echo "Step 5: Merging MaSIF features to existing PKL..."
    python /xcfhome/zncao02/affincraft/preprocess/merge_pkl.py \
        --pkl_file "$pkl_output" \
        --output_dir "$output_dir" \
        --name "$name" \
        --pk "$pk" \
        --rmsd "$rmsd" \
        --chain_id "$protein_chain"
    if [ $? -ne 0 ]; then
        echo "Error in merge_pkl.py for $name"
        continue
    fi

    # echo "Completed processing for $name (protein chain: $protein_chain)"
    echo "----------------------------------------"
done

echo "All processing completed!"