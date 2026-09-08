# -*- coding: utf-8 -*-
"""
GradNorm: Gradient Normalization for Adaptive Loss Balancing
in Deep Multitask Networks (ICML 2018)

即插即用的 GradNorm 实现，仅依赖 PyTorch，与具体模型完全解耦。

典型用法：
    model = MyMultiTaskModel()
    gradnorm = GradNorm(num_tasks=3, alpha=1.5)

    opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_grad  = torch.optim.Adam(gradnorm.parameters(), lr=2.5e-2)

    for x, ys in loader:
        opt_model.zero_grad()
        opt_grad.zero_grad()

        logits = model(x)                       # list[Tensor]，每个任务一个输出
        losses = [bce(logits[i], ys[i]) for i in range(3)]

        # 一行替代 total_loss.backward()：
        weights = gradnorm.backward(losses, shared_params=model.shared)

        opt_model.step()                        # 更新网络参数
        opt_grad.step()                         # 更新任务权重 w_i
"""

from typing import Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradNorm(nn.Module):
    """GradNorm 自适应多任务损失平衡模块。

    Args:
        num_tasks: 任务数 T。
        alpha: 不对称度超参数。alpha 越大，强制各任务训练速率拉齐的“恢复力”越强；
            任务差异大时取大些（论文 NYUv2 用 1.5），任务对称时取小些（如 0.12）。
        initial_losses: 理论初始损失 L_i(0)，长度 T 的序列，可选。
            不传则在第一次 backward 时自动记录首个 batch 的各任务损失（论文默认做法）。
            若初始损失对初始化/batch 敏感，可传入理论值，例如：
            - 二分类交叉熵的随机猜测水平：ln(2) ≈ 0.693；
            - C 类多分类交叉熵：ln(C)；
            - 平方损失回归：标签方差 Var(y)。
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        initial_losses: Sequence[float] = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha

        # 任务权重 w_i 通过 softmax 参数化：w_i = T * softmax(z)_i，
        # 天然保证 w_i > 0 且 sum_i w_i = T（对应论文每步重归一化）。
        self.weight_logits = nn.Parameter(torch.zeros(num_tasks))

        if initial_losses is None:
            self.register_buffer("initial_losses", torch.zeros(num_tasks))
            self._has_initial = False
        else:
            assert len(initial_losses) == num_tasks
            self.register_buffer(
                "initial_losses",
                torch.as_tensor(initial_losses, dtype=torch.float32).clone(),
            )
            self._has_initial = True

        self._step = 0

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_weights(self) -> torch.Tensor:
        """返回当前任务权重 w（已 detach），形状 [T]，满足 sum(w) = T。"""
        return self._weights().detach()

    def backward(
        self,
        losses: Sequence[torch.Tensor],
        shared_params: Union[nn.Module, Iterable[nn.Parameter]],
    ) -> torch.Tensor:
        """执行 GradNorm 的一步反传。

        内部完成两件事（对应论文 Algorithm 1 的双重更新）：
          1. 用 GradNorm 损失 L_grad 对任务权重 w_i 反传（梯度累积到本模块参数）；
          2. 用加权总损失 sum_i w_i L_i 对网络反传（梯度累积到模型参数）。

        调用后，用户只需分别 step 网络优化器和本模块优化器即可。

        Args:
            losses: 长度 T 的标量张量序列（每个任务一个 loss，需保持计算图）。
            shared_params: 共享骨干部分。传 nn.Module（如 model.shared）或
                共享参数的可迭代对象均可——GradNorm 的梯度范数只在这些参数上度量，
                对应论文中选“最后一个共享层”的做法；此处默认在整个共享骨干上度量。

        Returns:
            当前步实际使用的任务权重 w（detach 后的 [T] 张量）。
        """
        losses = list(losses)
        assert len(losses) == self.num_tasks, (
            f"losses 数量 {len(losses)} 与任务数 {self.num_tasks} 不一致"
        )
        params = self._normalize_params(shared_params)
        device = losses[0].device

        # 第 0 步：记录 L_i(0)，等权重预热（论文 w_i(0) = 1）
        if not self._has_initial:
            self.initial_losses = torch.tensor(
                [float(l.detach().item()) for l in losses], device=device
            )
            self._has_initial = True
            torch.stack(losses).sum().backward()
            self._step += 1
            return torch.ones(self.num_tasks, device=device)

        # 1) 当前任务权重 w_i = T * softmax(z)_i
        weights = self._weights()  # [T]，requires_grad=True

        # 2) 每个任务损失对共享参数的梯度。
        #    create_graph=False：梯度本身脱离计算图，使 L_grad 只对 w_i 可微。
        per_task_grads: List[Tuple[torch.Tensor, ...]] = []
        for i in range(self.num_tasks):
            gi = torch.autograd.grad(
                losses[i], params, retain_graph=True, allow_unused=True
            )
            gi = tuple(
                torch.zeros_like(p) if g is None else g
                for g, p in zip(gi, params)
            )
            per_task_grads.append(gi)

        # 3) 各任务加权梯度范数 G_W^(i)(t) = ||w_i * grad_W L_i||_2
        g_norms = []
        for i in range(self.num_tasks):
            flat = torch.cat([g.reshape(-1) for g in per_task_grads[i]])
            # g_norms.append(torch.linalg.vector_norm(weights[i] * flat, ord=2))
            g_norms.append(torch.norm(weights[i] * flat, p=2)) # 各个任务的梯度范数
        g_norms = torch.stack(g_norms)  # [T]，仅通过 weights 连计算图
        g_bar = g_norms.mean()          # 平均梯度范数 G_bar_W(t)

        # 4) 相对逆训练速率 r_i(t) = (L_i/L_i(0)) / mean_task(L_i/L_i(0))
        with torch.no_grad():
            cur_losses = torch.tensor(
                [float(l.detach().item()) for l in losses], device=device
            )
            loss_ratio = cur_losses / self.initial_losses.to(device)
            r_i = loss_ratio / loss_ratio.mean()  # r_i 越大 = 训练越慢

        # 5) 目标梯度范数 G_bar * r_i^alpha，detach 为常数
        #    （论文：treat the target gradient norm as a fixed constant，
        #      防止 w_i 全部漂向 0 的平凡解）
        target = (g_bar * r_i.pow(self.alpha)).detach()

        # 6) GradNorm 损失（L1），反传只更新任务权重 w_i
        l_grad = (g_norms - target).abs().sum()
        l_grad.backward()

        # 7) 用当前权重加权各任务损失做标准反传，
        #    更新共享骨干 + 各任务塔的全部网络参数
        w = weights.detach()
        total_loss = sum(w[i] * losses[i] for i in range(self.num_tasks))
        total_loss.backward()

        self._step += 1
        return w

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _weights(self) -> torch.Tensor:
        """w_i = T * softmax(z)_i，保证 w_i > 0 且 sum w_i = T。"""
        return self.num_tasks * F.softmax(self.weight_logits, dim=0)

    @staticmethod
    def _normalize_params(
        shared_params: Union[nn.Module, Iterable[nn.Parameter]]
    ) -> List[nn.Parameter]:
        """把 nn.Module 或参数可迭代对象统一成需要梯度的参数列表。"""
        if isinstance(shared_params, nn.Module):
            return [p for p in shared_params.parameters() if p.requires_grad]
        return [p for p in shared_params if p.requires_grad]


if __name__ == "__main__":
    import numpy as np
    import torch.nn as nn
    from mlp import MLP
    net = MLP([10, 5, 3])
    print(net)

    gradnorm = GradNorm(num_tasks=3, alpha=1.5,
                        initial_losses=[float(np.log(2))] * 3)
    opt_net = torch.optim.Adam(net.parameters(), lr=1e-3)
    opt_w = torch.optim.Adam(gradnorm.parameters(), lr=2.5e-2)
    bce = nn.BCEWithLogitsLoss()

    # 提取共享 trunk 参数：去掉 Sequential 最后一层（Linear(5,3)）
    shared_params = list(net.net[0].parameters())

    for step in range(10):
        opt_net.zero_grad()
        opt_w.zero_grad()

        x = torch.randn(64, 10)
        y = (torch.rand(64, 3) > 0.5).float()

        logits = net(x)                                 # [B, 3]
        losses = [bce(logits[:, i], y[:, i]) for i in range(3)]

        # 一行替代 sum(losses).backward()：在共享层上度量梯度范数
        w = gradnorm.backward(losses, shared_params=shared_params)

        opt_net.step()
        opt_w.step()

    print("训练后任务权重 w:", [round(v, 3) for v in w.tolist()],
          " (sum =", round(float(w.sum()), 3), ")")
