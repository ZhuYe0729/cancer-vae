import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TumorEvolutionModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=1):
        """
        参数:
        input_dim (int): 输入特征的维度 (您的数据是 7)。
        hidden_dim (int): LSTM 隐藏层和后续全连接层的维度。
        num_layers (int): LSTM 的层数，对于初步模型，1 或 2 层通常足够。
        """
        super(TumorEvolutionModel, self).__init__()
        
        # --- 核心：LSTM 层 ---
        # input_size: 每个时间步输入的特征维度
        # hidden_size: 隐藏状态的维度
        # num_layers: LSTM 堆叠的层数
        # batch_first=True: 让输入张量的维度变为 (batch, seq_len, features)，这更直观
        self.lstm = nn.LSTM(input_size=input_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=num_layers, 
                              batch_first=True)
        
        # --- 后续的全连接层 ---
        # 输入维度是 LSTM 的 hidden_dim
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2_mu = nn.Linear(128, 100)      # 预测均值
        self.fc2_sigma = nn.Linear(128, 100)   # 预测标准差

    def forward(self, x):
        """
        参数:
        x (torch.Tensor): 单个样本的输入，维度为 (m, 7)。
                           m 是序列长度 (type_number 的数量)。
        """
        # --- 1. 准备 LSTM 输入 ---
        # PyTorch LSTM/RNN 层期望的输入是 3D 张量: (batch_size, sequence_length, input_size)
        # যেহেতু我们一次只处理一个样本，所以 batch_size 是 1。
        # 我们需要给 x 增加一个 batch 维度。 (m, 7) -> (1, m, 7)
        x = x.unsqueeze(0) 

        # --- 2. 通过 LSTM 层 ---
        # h_0 和 c_0 是初始隐藏状态和细胞状态，默认是全零，所以我们不用显式传入
        # lstm_out: 包含序列中每个时间步的输出隐藏状态。维度 (1, m, hidden_dim)
        # (h_n, c_n): 最后一个时间步的隐藏状态和细胞状态。h_n 维度 (num_layers, 1, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # --- 3. 提取序列的最终表示 ---
        # 我们使用最后一个时间步的隐藏状态 h_n 作为整个演化序列的摘要。
        # h_n[-1] 获取最后一层 LSTM 的最终隐藏状态，维度为 (1, hidden_dim)
        sequence_embedding = h_n[-1]
        
        # --- 4. 通过全连接层 ---
        h = F.relu(self.fc1(sequence_embedding))
        
        mu = self.fc2_mu(h)          # 维度 (1, 100)
        sigma = F.softplus(self.fc2_sigma(h)) # 维度 (1, 100)

        # --- 5. 格式化输出 ---
        # 移除批次维度，以匹配损失函数期望的输入
        mu = mu.squeeze(0)      # (1, 100) -> (100)
        sigma = sigma.squeeze(0)  # (1, 100) -> (100)

        # 最终输出维度为 (100, 2)
        return torch.stack([mu, sigma], dim=-1)


import torch
import torch.nn as nn

class GaussianNLLLoss(nn.Module):
    """
    计算高斯分布的负对数似然损失。
    """
    def __init__(self, eps=1e-6):
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        参数:
        y_pred (torch.Tensor): 模型的预测输出，维度为 (batch_size, 100, 2)，
                               其中 y_pred[..., 0] 是均值 mu，y_pred[..., 1] 是标准差 sigma。
        y_true (torch.Tensor): 真实标签，维度为 (batch_size, 100)。
        """
        # 从预测中分离均值和标准差
        mu = y_pred[..., 0]
        raw_sigma = y_pred[..., 1]

        # 为防止 sigma 过小导致数值不稳定，可以给它加一个很小的 epsilon
        # sigma = sigma.clamp(min=1e-8)
        sigma = F.softplus(raw_sigma ) + self.eps

        # 计算损失
        # L = log(σ) + (y - μ)² / (2 * σ²)
        log_sigma = torch.log(sigma)
        squared_error = (y_true - mu) ** 2
        
        loss = log_sigma + (squared_error / (2 * sigma ** 2))

        # 返回所有样本损失的均值
        return loss.mean()
    

