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

# 跳过CSV头部，处理每一行  
tail -n +2 "$CSV_FILE" | while IFS=',' read -r receptor ligand name pk rmsd; do  
    echo "Processing: $name"  
    
    # 获取目标目录  
    target_dir=$(dirname "$receptor")  
    output_dir="$target_dir/output"  
    
    echo "Target directory: $target_dir"  
    echo "Output directory: $output_dir"  
    
    # 创建输出目录  
    mkdir -p "$output_dir"  
    
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
        "$pkl_output" || {  
        echo "Error in custom_input.py for $name"  
        rm -f "$temp_csv"  
        continue  
    }  
    
    # 清理临时文件  
    rm -f "$temp_csv"  
    
    # 检查PKL文件是否生成  
    if [ ! -f "$pkl_output" ]; then  
        echo "Error: PKL file not generated: $pkl_output"  
        continue  
    fi  
    echo "Generated PKL file: $pkl_output"  
    
    # Step 2: 网格和表面特征生成  
    echo "Step 2: Running meshfeatureGen.py..."  
    python /xcfhome/zncao02/affincraft/masif/meshfeatureGen.py \
        --pdb_file "$target_dir/complex.pdb" \
        --chain_id "A" \
        --ligand_code "UNK" \
        --ligand_chain "X" \
        --sdf_file "$ligand" \
        --output_dir "$output_dir" || {  
        echo "Error in meshfeatureGen.py for $name"  
        continue  
    }  
    
    # Step 3: 特征预计算  
    echo "Step 3: Running feature_precompute.py..."  
    python /xcfhome/zncao02/affincraft/masif/feature_precompute.py \
        --ply_file "$output_dir/surfaces/complex_A.ply" \
        --output_dir "$output_dir" \
        --masif_app "masif_site" || {  
        echo "Error in feature_precompute.py for $name"  
        continue  
    }  
    
    # Step 4: 指纹生成
    echo "Step 4: Running fingerprint_gen.py..."
    python /xcfhome/zncao02/affincraft/masif/fingerprint_gen.py \
        --precomputed_dir "$output_dir/precomputed/complex_A" \
        --output_dir "$output_dir" \
        --ppi_pair_id "complex_A" \
        --custom_params_file "masif.source.masif_ppi_search.nn_models.sc05.all_feat.custom_params" || {
        echo "Error in fingerprint_gen.py for $name"
        continue
    }
    
    # Step 5: 整合MaSIF特征到现有PKL文件  
    echo "Step 5: Merging MaSIF features to existing PKL..."  
    python /xcfhome/zncao02/affincraft/preprocess/merge_pkl.py \
        --pkl_file "$pkl_output" \
        --output_dir "$output_dir" \
        --name "$name" \
        --pk "$pk" \
        --rmsd "$rmsd" || {  
        echo "Error in merge_pkl.py for $name"  
        continue  
    }  
    
    echo "Completed processing for $name"  
    echo "----------------------------------------"  
done  

echo "All processing completed!"