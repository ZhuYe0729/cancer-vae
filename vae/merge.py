"""
Docstring for vae.merge
datasets.py中可能产生多个分片，这个函数就是把他们打包起来得到最终的一个文件。
比如我data_pretrained_1000这个完整的数据集我是分成两次产生的:
第一次是data_original_pretrained_1000_1
第二到九次是一个完整的文件data_original_pretrained_1000_2to9
我基于这个得到了当前/root/wja/wja/project/CHESS.cpp/data/pretrained_1000中的分片packed_train_data_1.npz和packed_train_data_2to9.npz
现在我需要把他们合并成一个完整的packed_train_data.npz文件
"""
import os
import numpy as np
from typing import List

def merge_npz_files(file_list: List[str], output_file: str) -> None:
    """
    Merge multiple .npz files containing dataset samples into a single .npz file.
    
    The input .npz files are expected to have keys like 'x', 'y', 'names', 'num_types'.
    """
    all_x = []
    all_y = []
    all_names = []
    all_num_types = []
    all_type_files = []

    print(f"Starting merge of {len(file_list)} files...")

    for fpath in file_list:
        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue
        
        print(f"Loading {fpath} ...")
        try:
            data = np.load(fpath, allow_pickle=True)
            
            # 'x' and 'y' are required
            if 'x' not in data or 'y' not in data:
                print(f"Warning: File {fpath} missing 'x' or 'y'. Skipping.")
                continue
            
            all_x.append(data['x'])
            all_y.append(data['y'])
            
            # Optional metadata
            if 'names' in data:
                all_names.append(data['names'])
            if 'num_types' in data:
                all_num_types.append(data['num_types'])
            if 'type_files' in data:
                all_type_files.append(data['type_files'])
                
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not all_x:
        print("No valid data found to merge.")
        return

    # Concatenate
    print("Concatenating arrays...")
    merged_x = np.concatenate(all_x, axis=0)
    merged_y = np.concatenate(all_y, axis=0)
    
    merged_names = np.concatenate(all_names, axis=0) if all_names else np.array([], dtype=object)
    merged_num_types = np.concatenate(all_num_types, axis=0) if all_num_types else np.array([], dtype=object)
    merged_type_files = np.concatenate(all_type_files, axis=0) if all_type_files else np.array([], dtype=object)

    print(f"Merged shapes: x={merged_x.shape}, y={merged_y.shape}")

    # Save
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Saving to {output_file} ...")
    np.savez_compressed(
        output_file,
        x=merged_x,
        y=merged_y,
        names=merged_names,
        num_types=merged_num_types,
        type_files=merged_type_files
    )
    print("Done.")

if __name__ == "__main__":
    # Example usage based on docstring
    base_dir = '/root/wja/wja/project/CHESS.cpp/data/pretrained_1000'
    
    # List of files to merge
    files_to_merge = [
        os.path.join(base_dir, 'packed_train_data_1.npz'),
        os.path.join(base_dir, 'packed_train_data_2to9.npz')
    ]
    
    output_path = os.path.join(base_dir, 'packed_train_data.npz')
    
    merge_npz_files(files_to_merge, output_path)


