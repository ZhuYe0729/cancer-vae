"""
    加载训练完成的模型，输入参数后得到结果，然后进行采样得到最终的vaf图。
"""
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import TumorEvolutionModel
from datasets import _parse_sim_params,_parse_vaf_distribution
data_dir = Path('/root/data/wja/project/CHESS.cpp/data_original/data')
result_dir = Path('/root/data/wja/project/CHESS.cpp/vae/vae_out')
type_number = 2
index = 0
second_indexs = range(64)

visual = True

if __name__ == "__main__":
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TumorEvolutionModel(input_dim=7).to(device)
    ckpt_path = '/root/data/wja/project/CHESS.cpp/vae/ckpts/vae_checkpoint.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    model.eval()

    # 先计算baseline的平均欧氏距离，以及最大最小距离
    baseline_distances = []
    for i in second_indexs:
        for j in range(i+1,len(second_indexs)):
            final_data_dir_i = data_dir / f'{type_number}' / f'{index}' / f'{i}'
            final_data_dir_j = data_dir / f'{type_number}' / f'{index}' / f'{j}'
            y_file_i = os.path.join(final_data_dir_i, 'vaf_distribution.csv')
            y_file_j = os.path.join(final_data_dir_j, 'vaf_distribution.csv')
            vaf_distribution_i = _parse_vaf_distribution(y_file_i)
            vaf_distribution_j = _parse_vaf_distribution(y_file_j)
            euc_distance = np.linalg.norm(vaf_distribution_i - vaf_distribution_j)
            baseline_distances.append(euc_distance)
    baseline_distance = np.mean(baseline_distances)
    baseline_distance_max = np.max(baseline_distances)
    baseline_distance_min = np.min(baseline_distances)
    print(f'Baseline max Euclidean distance between VAF distributions: {baseline_distance_max:.6f}')
    print(f'Baseline min Euclidean distance between VAF distributions: {baseline_distance_min:.6f}')
    print(f'Baseline average Euclidean distance between VAF distributions: {baseline_distance:.6f}')

    # 预测并计算平均欧氏距离，以及最大最小距离

    pred_distances = []
    for second_index in second_indexs:
        final_data_dir = data_dir / f'{type_number}' / f'{index}' / f'{second_index}'
        x_file = os.path.join(final_data_dir, 'sim_params_0.csv')
        y_file = os.path.join(final_data_dir, 'vaf_distribution.csv')
        # 从x_file和y_file中加载数据
        sim_params = _parse_sim_params(x_file)
        vaf_distribution = _parse_vaf_distribution(y_file)

        # 将sim_params输入到模型中，得到结果
        with torch.no_grad():
            x_tensor = torch.tensor(sim_params, dtype=torch.float32).to(device)  
            out = model(x_tensor)  # （100,2）
            mu = out[..., 0]      
            sigma = out[..., 1]
            sampled = torch.normal(mu, sigma).cpu().numpy()

        if visual:
            base_save_dir = result_dir / f'{type_number}' / f'{index}' / 'base'
            model_save_dir = result_dir / f'{type_number}' / f'{index}' / 'model'
            base_save_dir.mkdir(parents=True, exist_ok=True)
            model_save_dir.mkdir(parents=True, exist_ok=True)
            x_axis = np.arange(len(vaf_distribution))

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x_axis, vaf_distribution, color='tab:blue')
            ax.set_title(f'VAF Distribution #{second_index}')
            ax.set_xlabel('Bin Index')
            ax.set_ylabel('Frequency')
            fig.tight_layout()
            fig.savefig(str(base_save_dir / f'{second_index}.png'), dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x_axis, sampled, color='tab:orange')
            ax.set_title(f'Sampled Distribution #{second_index}')
            ax.set_xlabel('Bin Index')
            ax.set_ylabel('Frequency')
            fig.tight_layout()
            fig.savefig(str(model_save_dir / f'{second_index}.png'), dpi=200)
            plt.close(fig)

        euc_distance = np.linalg.norm(sampled - vaf_distribution)
        pred_distances.append(euc_distance)
    pred_distance = np.mean(pred_distances)
    print(f'Predicted max Euclidean distance between VAF distributions: {np.max(pred_distances):.6f}')
    print(f'Predicted min Euclidean distance between VAF distributions: {np.min(pred_distances):.6f}')
    print(f'Predicted average Euclidean distance between VAF distributions: {pred_distance:.6f}')
