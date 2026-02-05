import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.unet import UNet
from models.head import SegHead
from datasets.tw25_dataset import TW25SegDataset, PatchCfg


@torch.no_grad()
def main():
    # ====== 你需要按自己项目改这两个路径 ======
    ckpt_path = r"experiments/min_dual/checkpoints/epoch_01.pt"
    data_root = "data"
    val_json = r"data/TW25/tw25_val.json"  # 你的 val json 路径
    # =======================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 和训练一致的 patch size
    patch_cfg = PatchCfg(size=768)

    ds = TW25SegDataset(
        data_root=data_root,
        json_path=val_json,
        patch_cfg=patch_cfg,
        is_train=False,  # 评估模式
    )

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_ch=3, base_ch=32).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval()
    head.eval()

    thr = 0.5

    # 全局累积（更严谨）
    TP = 0.0
    FP = 0.0
    FN = 0.0

    # 也顺便算 mean（可选）
    f1_list = []
    iou_list = []

    for i, batch in enumerate(loader):
        img = batch["image"].to(device)          # [B,3,H,W]
        gt = batch["mask"].to(device)            # [B,1,H,W] or [B,H,W]
        ignore = batch.get("ignore", None)       # [B,1,H,W] or [B,H,W], 1=ignore

        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if ignore is not None and ignore.dim() == 3:
            ignore = ignore.unsqueeze(1)

        feat = model(img)
        logit = head(feat)
        prob = torch.sigmoid(logit)

        pred = (prob > thr).float()

        # valid mask：ignore=1 的地方不计入
        if ignore is None:
            valid = torch.ones_like(gt)
        else:
            ignore = ignore.to(device)
            valid = (ignore < 0.5).float()

        # 二值化 GT（防止是 0-255）
        gt_bin = (gt > 0.5).float()

        tp = (pred * gt_bin * valid).sum().item()
        fp = (pred * (1.0 - gt_bin) * valid).sum().item()
        fn = ((1.0 - pred) * gt_bin * valid).sum().item()

        TP += tp
        FP += fp
        FN += fn

        # per-batch 指标（看趋势用）
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1_list.append(f1)
        iou_list.append(iou)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(loader)}] batch_F1={f1:.4f} batch_IoU={iou:.4f}")

    # 全局指标（推荐写论文用这个）
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_global = 2 * precision * recall / (precision + recall + 1e-8)
    iou_global = TP / (TP + FP + FN + 1e-8)

    print("\n================= TW25 Patch-level Eval =================")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Threshold:  {thr}")
    print(f"Precision (global): {precision:.4f}")
    print(f"Recall    (global): {recall:.4f}")
    print(f"Pixel-F1  (global): {f1_global:.4f}")
    print(f"IoU       (global): {iou_global:.4f}")
    print("---------------------------------------------------------")
    print(f"Pixel-F1 (mean over batches): {float(np.mean(f1_list)):.4f}")
    print(f"IoU      (mean over batches): {float(np.mean(iou_list)):.4f}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
