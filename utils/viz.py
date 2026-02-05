from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch


def _to_uint8_img(x: torch.Tensor) -> np.ndarray:
    # x: [3,H,W] 0..1
    x = (x.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return x


def _to_uint8_mask(x: torch.Tensor) -> np.ndarray:
    # x: [1,H,W] or [H,W]
    if x.ndim == 3:
        x = x[0]
    x = (x.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return x


def save_vis(out_dir: str, step: int, image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = _to_uint8_img(image)
    gt = _to_uint8_mask(mask)
    pr = _to_uint8_mask(pred)

    # heatmap for pred
    heat = cv2.applyColorMap(pr, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    gt3 = np.stack([gt, gt, gt], axis=2)
    canvas = np.concatenate([img, gt3, heat], axis=1)  # [H, 3W, 3]

    save_path = Path(out_dir) / f"step_{step:08d}.png"
    cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))[1].tofile(str(save_path))
