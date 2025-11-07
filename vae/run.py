"""
    加载训练完成的模型，输入参数后得到结果，然后进行采样得到最终的vaf图。
"""
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import TumorEvolutionModel
from datasets import _parse_sim_params, _parse_vaf_distribution

data_dir = Path('/root/data/wja/project/CHESS.cpp/data_original/data')
result_dir = Path('/root/data/wja/project/CHESS.cpp/vae/vae_out')
type_number = 1
index = 0

second_indexs = range(64)

try:
    from scipy import linalg  # type: ignore[import]
except ImportError:  # SciPy might be unavailable in some environments
    linalg = None

visual = False


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute the matrix square root with optional SciPy support."""
    sym_matrix = (matrix + matrix.T) / 2.0
    if linalg is not None:
        sqrt_mat, _ = linalg.sqrtm(sym_matrix, disp=False)
        return np.real_if_close(sqrt_mat)
    eigvals, eigvecs = np.linalg.eigh(sym_matrix)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet Inception Distance between two collections of feature vectors."""
    real_features = np.asarray(real_features, dtype=np.float64)
    generated_features = np.asarray(generated_features, dtype=np.float64)

    if real_features.ndim == 1:
        real_features = real_features[None, :]
    if generated_features.ndim == 1:
        generated_features = generated_features[None, :]

    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError('Feature dimensionality mismatch between real and generated data.')

    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(generated_features, rowvar=False)

    if cov_real.ndim == 0:
        cov_real = np.array([[cov_real]])
    if cov_gen.ndim == 0:
        cov_gen = np.array([[cov_gen]])

    cov_real += np.eye(cov_real.shape[0]) * eps
    cov_gen += np.eye(cov_gen.shape[0]) * eps

    mean_diff = mu_real - mu_gen
    cov_prod = cov_real @ cov_gen
    covmean = _matrix_sqrt(cov_prod)

    fid_score = mean_diff @ mean_diff + np.trace(cov_real + cov_gen - 2.0 * covmean)
    return float(np.real(fid_score))


if __name__ == "__main__":
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TumorEvolutionModel(input_dim=7).to(device)
    ckpt_path = '/root/data/wja/project/CHESS.cpp/vae/ckpts/vae_checkpoint.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    model.eval()

    second_indices = list(second_indexs)
    vaf_cache = {}
    for second_index in second_indices:
        final_data_dir = data_dir / f'{type_number}' / f'{index}' / f'{second_index}'
        y_file = os.path.join(final_data_dir, 'vaf_distribution.csv')
        vaf_cache[second_index] = _parse_vaf_distribution(y_file)

    # 先计算baseline的平均欧氏距离，以及最大最小距离
    baseline_distances = []
    for idx_i, i in enumerate(second_indices):
        for j in second_indices[idx_i + 1:]:
            vaf_distribution_i = vaf_cache[i]
            vaf_distribution_j = vaf_cache[j]
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
    generated_samples = []
    for second_index in second_indices:
        final_data_dir = data_dir / f'{type_number}' / f'{index}' / f'{second_index}'
        x_file = os.path.join(final_data_dir, 'sim_params_0.csv')
        sim_params = _parse_sim_params(x_file)
        vaf_distribution = vaf_cache[second_index]

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
        generated_samples.append(sampled)

    pred_distance = np.mean(pred_distances)
    print(f'Predicted max Euclidean distance between VAF distributions: {np.max(pred_distances):.6f}')
    print(f'Predicted min Euclidean distance between VAF distributions: {np.min(pred_distances):.6f}')
    print(f'Predicted average Euclidean distance between VAF distributions: {pred_distance:.6f}')

    if len(generated_samples) > 1:
        fid_score = calculate_fid(np.stack(list(vaf_cache.values())), np.stack(generated_samples))
        print(f'FID between ground truth and generated VAF distributions: {fid_score:.6f}')
    else:
        print('FID between ground truth and generated VAF distributions was not computed (insufficient samples).')
