from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class bihalfHash(Function):
    @staticmethod
    def forward(ctx, U: torch.Tensor) -> torch.Tensor:
        assert U.shape[0] % 2 == 0, f"双半哈希要求批次大小为偶数，当前为{U.shape[0]}"
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        half_n = int(N / 2)

        # 动态设备：跟随输入U（兼容原生Gme的设备）
        B_creat = torch.cat((
            torch.ones([half_n, D], device=U.device, dtype=U.dtype),
            -torch.ones([N - half_n, D], device=U.device, dtype=U.dtype)
        ), dim=0)
        B = torch.zeros_like(U).scatter_(0, index, B_creat)

        ctx.save_for_backward(U, B)
        return B

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        U, B = ctx.saved_tensors
        add_grad = (U - B) / B.numel()  # 归一化梯度，避免量级问题
        total_grad = grad_output + 6 * add_grad  # 梯度=上游梯度+正则化梯度
        return total_grad


def bihalfHash_layer(input: torch.Tensor) -> torch.Tensor:
    return bihalfHash.apply(input)

class Hashnet(nn.Module):

    def __init__(self, hash_dim,gme_dim):
        super().__init__()
        self.hash_dim = hash_dim
        self.gme_dim = gme_dim
        self.fc_layer = nn.Linear(self.gme_dim, self.hash_dim)
        self.fc_bn = nn.BatchNorm1d(self.hash_dim)

    def forward(self,x):

        fc_emb = self.fc_layer(x)
        fc_emb = self.fc_bn(fc_emb)

        # 3. 双半哈希生成二值码
        hash_code = bihalfHash_layer(fc_emb)

        return fc_emb, hash_code


