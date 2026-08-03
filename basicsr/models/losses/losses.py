import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-12):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, prediction, target):
        loss = torch.sqrt((prediction - target) ** 2 + self.eps)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss
