"""
    用于测试评估：目前的评估方法是欧式距离。
"""
import numpy as np
import torch

def eval(y1, y2):
    return np.linalg.norm(y1 - y2)

def sample(model_out):
    """
        从模型中输出中采样生成具体的vaf。
        [100, 2] -> [100]
    """
    mu = model_out[:, 0]
    sigma = model_out[:, 1]
    y = mu + sigma * torch.randn_like(mu)
    return y


def eval_original(file='/root/data/wja/project/CHESS.cpp/data/chess/train/packed_train_data.npz', 
                  num=1000):
    """
        评估原始的chess.cpp中对于同一输入，在多次运行下生成的输出之前的差异，作为baseline
    """
    data = np.load(file, allow_pickle=True)
    y_samples = data['y'][:num]
    # 计算所有样本之间的pairwise距离
    distances = []
    for i in range(len(y_samples)):
        for j in range(i + 1, len(y_samples)):
            dist = eval(y_samples[i], y_samples[j])
            distances.append(dist)
    return np.mean(distances) if distances else 0.0


def eval_model(y_pred, y_true):
    """
        评估自己的模型生成的输出与labels的差异
    """
    sampled_y = sample(y_pred)
    return eval(sampled_y, y_true)


if __name__ == "__main__":
    baseline_distance = eval_original()
    print(f"Baseline distance from original chess.cpp: {baseline_distance}")