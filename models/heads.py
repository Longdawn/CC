from __future__ import annotations
import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat):
        return self.proj(feat)  # logits [B,1,H,W]
