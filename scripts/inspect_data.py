import torch
import cv2
import numpy as np
from pathlib import Path
from datasets.tw25_dataset import TW25SegDataset, PatchCfg


def to_u8_img(x):
    """[3,H,W] 0..1 -> HxWx3 uint8 RGB"""
    return (x.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def to_u8_m(x):
    """[1,H,W] -> HxW uint8"""
    x = x[0].cpu().numpy()
    return (x * 255).clip(0, 255).astype(np.uint8)


cfg = PatchCfg(size=768)
ds = TW25SegDataset(data_root="data", json_path="data/tw25/tw25_train.json", patch_cfg=cfg, is_train=True)

sample = ds[0]
img = to_u8_img(sample["image"])
img_bg = to_u8_img(sample["image_bg"])
gt = to_u8_m(sample["mask"])
ig = to_u8_m(sample["ignore"])

# Convert masks to 3-channel for concatenation
gt3 = np.stack([gt, gt, gt], 2)
ig_heat = cv2.applyColorMap(ig, cv2.COLORMAP_JET)
ig_heat = cv2.cvtColor(ig_heat, cv2.COLOR_BGR2RGB)

# Concatenate: [img | gt | img_bg | ignore]
canvas = np.concatenate([img, gt3, img_bg, ig_heat], axis=1)

# Save
out = Path("experiments/_debug_vis")
out.mkdir(parents=True, exist_ok=True)
save_path = out / "bg_check.png"
cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))[1].tofile(str(save_path))
print(f"Saved to: {save_path}")
print(f"Image shape: {img.shape}, BG shape: {img_bg.shape}, GT shape: {gt.shape}, Ignore shape: {ig.shape}")
