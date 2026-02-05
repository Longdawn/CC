import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
