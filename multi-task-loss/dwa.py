# -*- coding: utf-8 -*-
"""
DWA: Dynamic Weight Averaging（动态权重平均）
End-to-End Multi-Task Learning with Attention (Liu et al., CVPR 2019, Sec 4.1.3)

即插即用的 DWA 实现，仅依赖 PyTorch，与具体模型完全解耦。

与 GradNorm 的区别：
- DWA **没有可学习参数**，不需要额外的优化器，也不需要度量梯度范数；
- 任务权重完全由“各任务损失的下降速度”这一历史统计量决定：
  损失下降越快（学得越快）的任务权重越小，下降越慢（学得越慢）的任务权重越大；
- 因此 DWA 极其廉价，每个训练步只多一次 softmax，且不依赖共享层参数。

典型用法：
    model = MyMultiTaskModel()
    dwa = DWA(num_tasks=3, temperature=2.0)

    opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)

    for x, ys in loader:
        opt_model.zero_grad()

        logits = model(x)                       # list[Tensor]，每个任务一个输出
        losses = [bce(logits[i], ys[i]) for i in range(3)]

        # 一行替代 total_loss.backward()：
        weights = dwa.backward(losses)

        opt_model.step()                        # 只需更新网络参数，无权重优化器

论文口径说明：
    论文中 L_k(t) 是第 t 个 **epoch** 的平均损失，权重每个 epoch 更新一次，
    前两个 epoch 用等权重（lambda_k = 1）。本实现按“每次调用”记录损失，
    每个 step 调用一次即为逐 step 版本（batch 损失噪声较大，但更省改动）；
    若要严格对齐论文，可每个 epoch 用该 epoch 的平均损失调用一次 update()。
"""

from collections import deque
from typing import Deque, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWA(nn.Module):
    """DWA 自适应多任务损失平衡模块（基于损失下降速度，无可学习参数）。

    权重公式（论文 Eq. 见 Sec 4.1.3）：
        学习速度  w_k(t-1) = L_k(t-1) / L_k(t-2)   # 越小 = 学得越快
        任务权重  lambda_k(t) = K * softmax(w_k(t-1) / tau)_k
    天然满足 lambda_k > 0 且 sum_k lambda_k = K（平均权重为 1）。

    Args:
        num_tasks: 任务数 K。
        temperature: 温度系数 tau（论文取 2.0）。tau 越大权重越趋近等权 1，
            tau 越小权重在任务间越尖锐。
        clip: 对损失比值 w_k 的截断范围 [0, clip]，防止某个任务损失反弹
            （比值 >> 1）时 softmax 数值爆炸（官方 MTAN 实现默认 clip 到 10）。
            设为 None 可关闭截断。
        eps: 分母保护项，避免 L_k(t-2) 接近 0 时除零。
    """

    def __init__(
        self,
        num_tasks: int,
        temperature: float = 2.0,
        clip: Optional[float] = 10.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.clip = clip
        self.eps = eps

        # 损失历史：只保留最近两次损失 L(t-1)、L(t-2)（DWA 权重仅依赖它们），
        # 用 maxlen=2 的 deque 自动淘汰最旧记录，避免随训练步数无限增长。
        # 不注册为 buffer，避免 state_dict 混入训练过程中的临时状态。
        self.loss_history: Deque[torch.Tensor] = deque(maxlen=2)
        self._cur_weights = torch.ones(num_tasks)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_weights(self) -> torch.Tensor:
        """返回当前任务权重（已 detach），形状 [K]，满足 sum(w) = K。"""
        return self._cur_weights.detach()

    def update(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """根据损失历史计算当前步权重，并把当前损失记入历史。

        权重只由 **历史** 损失 L(t-1)/L(t-2) 决定，与当前步损失无关
        （当前步损失在权重确定后才用于反传，与论文一致）。

        Args:
            losses: 长度 K 的标量张量序列（每个任务一个 loss）。

        Returns:
            当前步实际使用的任务权重 w（[K] 张量，无梯度）。
        """
        losses = list(losses)
        assert len(losses) == self.num_tasks, (
            f"losses 数量 {len(losses)} 与任务数 {self.num_tasks} 不一致"
        )
        device = losses[0].device

        cur_losses = torch.stack(
            [l.detach().reshape(()) for l in losses]
        ).float().to(device)  # [K]

        # 需要至少两条历史记录（L(t-1)、L(t-2)）才能估计下降速度
        if len(self.loss_history) >= 2:
            l_prev = self.loss_history[-1].to(device)   # L_k(t-1)
            l_prev2 = self.loss_history[-2].to(device)  # L_k(t-2)

            # w_k = L_k(t-1) / L_k(t-2)：损失下降越快，比值越小
            speed = l_prev / (l_prev2 + self.eps)
            if self.clip is not None:
                speed = torch.clamp(speed, min=0.0, max=self.clip)

            # lambda_k = K * softmax(w_k / tau)，max 减法保证数值稳定
            logits = speed / self.temperature
            logits = logits - logits.max()
            weights = self.num_tasks * F.softmax(logits, dim=0)
        else:
            # 前两个 step/epoch：等权重预热（论文默认做法，lambda_k = 1）
            weights = torch.ones(self.num_tasks, device=device)

        self.loss_history.append(cur_losses)
        self._cur_weights = weights
        return weights

    def backward(
        self,
        losses: Sequence[torch.Tensor],
        shared_params: Optional[Union[nn.Module, object]] = None,
    ) -> torch.Tensor:
        """计算 DWA 权重并执行加权反传，一行替代 total_loss.backward()。

        Args:
            losses: 长度 K 的标量张量序列（每个任务一个 loss，需保持计算图）。
            shared_params: 仅为与 GradNorm.backward 接口对齐而保留，
                DWA 基于损失历史、不涉及梯度度量，传入后会被忽略。

        Returns:
            当前步实际使用的任务权重 w（[K] 张量，无梯度）。
        """
        weights = self.update(losses)

        # 用当前权重加权各任务损失做标准反传，更新全部网络参数
        total_loss = sum(
            weights[i] * losses[i] for i in range(self.num_tasks)
        )
        total_loss.backward()
        return weights


if __name__ == "__main__":
    import torch.nn as nn
    from mlp import MLP

    torch.manual_seed(0)
    net = MLP([10, 5, 3])
    print(net)

    dwa = DWA(num_tasks=3, temperature=2.0)
    opt_net = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    # DWA 无需共享参数、无需权重优化器：直接反传 + step 网络即可
    for step in range(15):
        opt_net.zero_grad()

        x = torch.randn(64, 10)
        y = (torch.rand(64, 3) > 0.5).float()

        logits = net(x)                                 # [B, 3]
        losses = [bce(logits[:, i], y[:, i]) for i in range(3)]
        print(losses)
        # 一行替代 sum(losses).backward()
        w = dwa.backward(losses)

        opt_net.step()

        loss_str = ", ".join(f"{float(l):.4f}" for l in losses)
        print(
            f"step {step:02d} | 损失: [{loss_str}] | "
            f"权重: {[round(v, 3) for v in w.tolist()]} "
            f"(sum = {round(float(w.sum()), 3)})"
        )
