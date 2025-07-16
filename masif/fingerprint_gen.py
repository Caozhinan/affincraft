import numpy as np    
import os    
import sys    
import time    
import importlib    
    
# 导入必要的模块    
from default_config.masif_opts import masif_opts    
from masif_modules.MaSIF_ppi_search import MaSIF_ppi_search    
from masif_modules.train_ppi_search import compute_val_test_desc    
    
def mask_input_feat(input_feat, mask):    
    """    
    根据特征掩码过滤输入特征。    
    """    
    mymask = np.where(np.array(mask) == 0.0)[0]    
    return np.delete(input_feat, mymask, axis=2)    
    
def generate_surface_fingerprints(precomputed_features_dir, output_dir, ppi_pair_id, custom_params_file=None):    
    """    
    Step 4: 基于神经网络的指纹生成    
        
    参数:    
    - precomputed_features_dir: Step 3生成的预计算特征的目录    
    - output_dir: 最终指纹的输出目录    
    - ppi_pair_id: 蛋白质-配对ID (例如 'complex_A')    
    - custom_params_file: 包含自定义参数的Python文件路径 (例如 'nn_models.sc05.all_feat.custom_params')    
        
    返回:    
    - 包含指纹数据的字典    
    """    
        
    print(f"开始Step 4: 基于神经网络的指纹生成...")    
    print(f"从目录加载预计算特征: {precomputed_features_dir}")    
        
    # 1. 设置参数    
    params = masif_opts["ppi_search"].copy()  # 创建副本避免修改原始配置  
    if custom_params_file:    
        try:    
            custom_params_module = importlib.import_module(custom_params_file, package=None)    
            custom_params = custom_params_module.custom_params    
            for key in custom_params:    
                params[key] = custom_params[key]    
            print(f"加载自定义参数文件: {custom_params_file}")    
        except Exception as e:    
            print(f"警告: 无法加载自定义参数文件 {custom_params_file}: {e}")    
            print("将使用默认参数。")    
    
    # 2. 加载预训练的神经网络模型    
    print("加载预训练的神经网络模型...")    
    learning_obj = MaSIF_ppi_search(    
        params["max_distance"],    
        n_thetas=16,    
        n_rhos=5,    
        n_rotations=16,    
        idx_gpu="/gpu:0",    
        feat_mask=params["feat_mask"],    
    )    
      
    # 恢复模型权重 - 改进的路径处理  
    model_path = os.path.join(params["model_dir"], "model")  
      
    # 首先尝试使用配置的路径  
    if not os.path.exists(model_path + ".meta"):  
        # 如果配置路径不存在，尝试使用绝对路径  
        abs_model_dir = "/xcfhome/zncao02/affincraft/masif/data/masif_ppi_search/nn_models/sc05/all_feat/model_data/"  
        abs_model_path = os.path.join(abs_model_dir, "model")  
          
        if os.path.exists(abs_model_path + ".meta"):  
            model_path = abs_model_path  
            print(f"使用绝对路径加载模型: {abs_model_dir}")  
        else:  
            print(f"错误: 模型文件未找到。")  
            print(f"尝试的路径:")  
            print(f"  - {model_path}.meta")  
            print(f"  - {abs_model_path}.meta")  
            print("请确保模型文件存在于正确位置。")  
            return None  
    else:  
        print(f"使用配置路径加载模型: {params['model_dir']}")  
    
    try:  
        learning_obj.saver.restore(learning_obj.session, model_path)    
        print("神经网络模型加载成功。")  
    except Exception as e:  
        print(f"模型加载失败: {e}")  
        return None  
    
    # 3. 加载预计算的特征数据    
    in_dir = precomputed_features_dir    
        
    try:    
        p1_rho_wrt_center = np.load(os.path.join(in_dir, "p1_rho_wrt_center.npy"))    
        p1_theta_wrt_center = np.load(os.path.join(in_dir, "p1_theta_wrt_center.npy"))    
        p1_input_feat = np.load(os.path.join(in_dir, "p1_input_feat.npy"))    
        p1_mask = np.load(os.path.join(in_dir, "p1_mask.npy"))    
            
        # 过滤输入特征 (如果feat_mask有定义)    
        p1_input_feat = mask_input_feat(p1_input_feat, params["feat_mask"])    
            
        idx1 = np.array(range(len(p1_rho_wrt_center)))    
        print("预计算特征数据加载成功。")    
    except Exception as e:    
        print(f"错误: 无法加载预计算特征数据: {e}")    
        print(f"请确保文件位于 {in_dir} 且命名正确。")    
        return None    
    
    # 4. 计算表面描述符 (指纹)    
    print("计算表面描述符 (指纹)...")    
    tic = time.time()    
        
    # 计算标准方向的描述符    
    desc1_str = compute_val_test_desc(    
        learning_obj,    
        idx1,    
        p1_rho_wrt_center,    
        p1_theta_wrt_center,    
        p1_input_feat,    
        p1_mask,    
        batch_size=24, # 可以根据GPU内存调整    
        flip=False,    
    )    
        
    # 计算翻转方向的描述符 (用于互补性分析)    
    desc1_flip = compute_val_test_desc(    
        learning_obj,    
        idx1,    
        p1_rho_wrt_center,    
        p1_theta_wrt_center,    
        p1_input_feat,    
        p1_mask,    
        batch_size=24, # 可以根据GPU内存调整    
        flip=True,    
    )    
    print(f"描述符计算完成，耗时: {time.time() - tic:.2f}s")    
    
    # 5. 保存描述符    
    out_desc_dir = os.path.join(output_dir, "descriptors", ppi_pair_id)    
    os.makedirs(out_desc_dir, exist_ok=True)    
        
    np.save(os.path.join(out_desc_dir, "p1_desc_straight.npy"), desc1_str)    
    np.save(os.path.join(out_desc_dir, "p1_desc_flipped.npy"), desc1_flip)    
    print(f"表面指纹已保存到: {out_desc_dir}")    
    
    # 6. 构建返回的指纹字典    
    fingerprints = {    
        'desc_straight': desc1_str,    
        'desc_flipped': desc1_flip,    
        'output_dir': out_desc_dir    
    }    
        
    print(f"Step 4完成! 生成了 {desc1_str.shape[0]} 个指纹，每个指纹维度为 {desc1_str.shape[1]}。")    
        
    return fingerprints    
    
# 使用示例    
if __name__ == "__main__":    
    import argparse    
    parser = argparse.ArgumentParser(description='生成MaSIF表面指纹')  
    parser.add_argument('--precomputed_dir', required=True, help='预计算特征目录')  
    parser.add_argument('--output_dir', required=True, help='输出目录')  
    parser.add_argument('--ppi_pair_id', required=True, help='蛋白质对ID')  
    parser.add_argument('--custom_params_file', default="masif.source.masif_ppi_search.nn_models.sc05.all_feat.custom_params",   
                       help='自定义参数文件')  
        
    args = parser.parse_args()    
        
    fingerprints = generate_surface_fingerprints(    
        precomputed_features_dir=args.precomputed_dir,    
        output_dir=args.output_dir,    
        ppi_pair_id=args.ppi_pair_id,    
        custom_params_file=args.custom_params_file    
    )  
      
    if fingerprints is not None:  
        print("\\n=== Step 4 输出指纹总结 ===")  
        print(f"标准方向指纹形状: {fingerprints['desc_straight'].shape}")  
        print(f"翻转方向指纹形状: {fingerprints['desc_flipped'].shape}")  
        print(f"指纹文件保存在: {fingerprints['output_dir']}")  
    else:  
        print("指纹生成失败!")  
        sys.exit(1)