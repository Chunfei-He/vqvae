import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e       # embedding 向量的个数（即代码簿的大小）
        self.e_dim = e_dim   # 每个 embedding 向量的维度
        self.beta = beta     # commitment cost 系数（用于损失函数）

        # 定义一个 Embedding 层，包含 n_e 个 embedding，每个的维度是 e_dim
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # 对 embedding 权重初始化，范围设为 [-1/n_e, 1/n_e]
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # 1) 先将维度从 (B,C,H,W) -> (B,H,W,C)，然后再展平到 (B*H*W, C)
        # batch, channel, height, width -> batch, height, width, channel
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # 2) 计算 z_flattened 与代码簿中每个 embedding 之间的距离
        #    距离公式: ||z - e||^2 = (z^2 + e^2 - 2 z·e)
        #    这里 d 的维度会是 (B*H*W, n_e)，表示每个 z 对所有 embedding 的距离。
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # 3) 找到距离最小的 embedding 索引（即最接近的代码向量）
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # 4) 将这些索引转换为 one-hot 编码。比如如果最近的是索引 k，就在对应位置上放 1
        #    min_encodings 的形状是 (B*H*W, n_e)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # 5) 根据 one-hot 再和 embedding 权重相乘，得到量化后的向量 z_q
        #    这里 (B*H*W, n_e) x (n_e, e_dim) => (B*H*W, e_dim)
        #    再 reshape 回 (B,H,W,e_dim)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
