import pickle  
import torch  
import numpy as np  
from pathlib import Path  
import sys  
  
def print_pkl_contents(pkl_file_path):  
    """打印PKL文件的所有内容和形状"""  
      
    if not Path(pkl_file_path).exists():  
        print(f"错误: PKL文件不存在: {pkl_file_path}")  
        return  
      
    try:  
        print(f"正在加载PKL文件: {pkl_file_path}")  
        with open(pkl_file_path, 'rb') as f:  
            graphs = pickle.load(f)  
          
        print(f"成功加载 {len(graphs)} 个复合物")  
          
        for idx, graph in enumerate(graphs):  
            print(f"\n{'='*60}")  
            print(f"复合物 {idx + 1}: {graph.get('pdbid', 'Unknown')}")  
            print(f"{'='*60}")  
              
            print("\n=== 所有数据字段和形状 ===")  
            for key, value in graph.items():  
                if isinstance(value, (np.ndarray, torch.Tensor)):  
                    print(f"  {key}: {type(value).__name__} shape {value.shape}")  
                    if hasattr(value, 'dtype'):  
                        print(f"    数据类型: {value.dtype}")  
                else:  
                    print(f"  {key}: {type(value).__name__} - {value}")  
          
        print(f"\n分析完成！")  
          
    except Exception as e:  
        print(f"处理过程中出现错误: {e}")  
        import traceback  
        traceback.print_exc()  
  
if __name__ == "__main__":  
    if len(sys.argv) != 2:  
        print("用法: python script.py <pkl_file_path>")  
        sys.exit(1)  
      
    pkl_file_path = sys.argv[1]  
    print_pkl_contents(pkl_file_path)