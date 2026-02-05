import torch
import torch.nn as nn


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, ignore=None):
        loss = self.bce(pred, target)
        if ignore is not None:
            loss = loss * (1.0 - ignore)
        return loss.mean()
