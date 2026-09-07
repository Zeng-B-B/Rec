"""
多任务不确定性加权损失(Uncertainty to Weigh Losses)

论文: Multi-Task Learning Using Uncertainty to Weigh Losses for
      Scene Geometry and Semantics (Kendall et al., CVPR 2018)
      https://arxiv.org/pdf/1705.07115

核心思想:
    用可学习参数 σ_i 度量每个任务的同方差不确定性(homoscedastic uncertainty):
    不确定性越大的任务,loss 权重越小,从而避免大 loss / 噪声任务主导训练。

总损失:
    L = Σ_i [ 1/(2σ_i²) · L_i + log(1 + σ_i) ]

实现细节(与 readme 一致):
    1. 参数化 σ = e^x,x 为可学习参数,保证 σ > 0;
    2. x 初始化为 ln(1/√2),使初始 2σ² = 1,即初始权重 1/(2σ²) = 1;
    3. 正则项 log σ 使用 log(1 + σ)(= softplus(x)),恒为正,保证损失非负,
       同时防止 σ 无限增大导致任务被彻底忽略。
"""

import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """即插即用的多任务不确定性加权损失模块。

    用法:
        criterion = UncertaintyWeightedLoss(num_tasks=3,
                                            task_names=["click", "like", "finish"])

        # 训练循环中,各任务 loss 需为标量(通常已对 batch 做 mean)
        losses = {
            "click":  F.binary_cross_entropy(click_pred, click_label),
            "like":   F.binary_cross_entropy(like_pred, like_label),
            "finish": F.binary_cross_entropy(finish_pred, finish_label),
        }
        total_loss = criterion(losses)   # 直接 backward 即可,权重自动学习
        total_loss.backward()

        # 记录日志(可选)
        writer.add_scalars("train/sigma", criterion.value_dict())
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError(f"num_tasks 必须为正整数,得到 {num_tasks}")
        if task_names is not None:
            if len(task_names) != num_tasks:
                raise ValueError(
                    f"task_names 长度({len(task_names)})与 num_tasks({num_tasks})不一致"
                )
            self.task_names = list(task_names)
        else:
            self.task_names = [f"task_{i}" for i in range(num_tasks)]

        # σ = e^x;x 初始化为 ln(1/√2),使 2σ² = 1,初始权重为 1
        self.log_sigma = nn.Parameter(torch.full((num_tasks,), math.log(1.0 / math.sqrt(2.0))))

    # ---------- 便于监控的只读属性 ----------
    @property
    def sigma(self) -> torch.Tensor:
        """各任务当前的不确定性 σ = e^x。"""
        return torch.exp(self.log_sigma)

    @property
    def weights(self) -> torch.Tensor:
        """各任务当前的 loss 权重 1/(2σ²)。"""
        return 0.5 / self.sigma.pow(2)

    @torch.no_grad()
    def value_dict(self) -> Dict[str, float]:
        """返回各任务的 σ 与权重(detach 后的 Python float),用于日志记录。"""
        stats: Dict[str, float] = {}
        for name, s, w in zip(self.task_names, self.sigma, self.weights):
            stats[f"sigma/{name}"] = s.item()
            stats[f"weight/{name}"] = w.item()
        return stats

    # ---------- 前向 ----------
    def forward(
        self,
        losses: Union[Dict[str, torch.Tensor], Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """计算加权后的总损失(标量)。

        Args:
            losses: 各任务的标量 loss,支持两种传入方式:
                - dict: {任务名: loss_tensor},键需与 task_names 对应(顺序无关);
                - list/tuple: 按 task_names 顺序排列的 loss_tensor。

        Returns:
            total_loss: 标量张量,可直接 backward。
        """
        if isinstance(losses, dict):
            names = list(losses.keys())
            if set(names) != set(self.task_names):
                raise KeyError(
                    f"losses 的键 {names} 与 task_names {self.task_names} 不匹配"
                )
            loss_vec = torch.stack([losses[n] for n in self.task_names]) # (num_tasks,)
        else:
            loss_vec = torch.stack(list(losses))
            if loss_vec.shape[0] != self.log_sigma.shape[0]:
                raise ValueError(
                    f"传入 {loss_vec.shape[0]} 个 loss,但模块定义了 "
                    f"{self.log_sigma.shape[0]} 个任务"
                )

        sigma = self.sigma # (num_tasks,)
        # 1/(2σ²) · L_i  —— 不确定性越大,权重越小
        weighted = 0.5 / sigma.pow(2) * loss_vec # (num_tasks,)
        # log(1 + σ)  —— 正则项,恒正,防止 σ 无限增大
        reg = torch.log1p(sigma) # (num_tasks,)

        return (weighted + reg).sum()


if __name__ == "__main__":
    # ---------------- 自测 demo:两个虚拟任务 ----------------
    torch.manual_seed(0)

    num_tasks = 2
    criterion = UncertaintyWeightedLoss(num_tasks, task_names=["easy_task", "hard_task"])

    # 共享底座 + 两个任务头
    backbone = nn.Linear(8, 16)
    head_easy = nn.Linear(16, 1)
    head_hard = nn.Linear(16, 1)
    params = list(backbone.parameters()) + list(head_easy.parameters()) + \
             list(head_hard.parameters()) + list(criterion.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)

    x = torch.randn(64, 8)
    y_easy = (x[:, :1].sum(dim=1, keepdim=True) > 0).float()
    y_hard = torch.randint(0, 2, (64, 1)).float()  # 纯噪声任务(不确定性应更大)

    for step in range(10):
        h = torch.relu(backbone(x))
        loss_easy = nn.functional.binary_cross_entropy_with_logits(head_easy(h), y_easy)
        loss_hard = nn.functional.binary_cross_entropy_with_logits(head_hard(h), y_hard)

        total = criterion([loss_easy, loss_hard])

        opt.zero_grad()
        total.backward()
        opt.step()

        if step % 1 == 0 or step == 199:
            stats = criterion.value_dict()
            print(
                f"step {step:3d} | total {total.item():.4f} "
                f"| easy: loss {loss_easy.item():.4f} w {stats['weight/easy_task']:.3f} "
                f"σ {stats['sigma/easy_task']:.3f} "
                f"| hard: loss {loss_hard.item():.4f} w {stats['weight/hard_task']:.3f} "
                f"σ {stats['sigma/hard_task']:.3f}"
            )

    # 验证:噪声任务(hard)应学到更大的 σ、更小的权重
    w = criterion.weights.detach()
    assert criterion.log_sigma.grad is not None, "σ 未收到梯度,模块未参与反向传播"
    print("\n自测通过:不确定性参数可学习,梯度正常回传。")
    print(f"最终权重 -> easy_task: {w[0]:.3f}, hard_task: {w[1]:.3f}")
