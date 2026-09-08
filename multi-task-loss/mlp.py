# -*- coding: utf-8 -*-
"""
多任务精排模型 A：共享 MLP 骨干输出共享表征，再输入各任务塔，共 3 个二分类任务。

结构：
    x ──> shared MLP ──> 共享表征 rep ──> tower_0 ──> logit_0
                                      ├> tower_1 ──> logit_1
                                      └> tower_2 ──> logit_2

模型本身与 GradNorm 完全解耦：它只负责前向计算，不感知任何损失加权逻辑。
"""

from typing import List, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """简单的多层感知机，最后一层为线性输出（不带激活）。"""

    def __init__(self, dims: Sequence[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 最后一层不加激活/dropout
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


