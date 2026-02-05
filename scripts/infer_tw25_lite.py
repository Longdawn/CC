"""
简化版推理脚本：用 checkpoint 在验证集上快速测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from datasets.tw25_dataset import TW25SegDataset, PatchCfg
from models.unet import UNet
from models.head import SegHead
import numpy as np
import cv2
import time

def compute_metrics(pred_mask, gt_mask, ignore_mask=None):
    """计算 Pixel-level F1"""
    if ignore_mask is not None:
        pred_mask = pred_mask * (1 - ignore_mask)
        gt_mask = gt_mask * (1 - ignore_mask)
    
    tp = (pred_mask * gt_mask).sum()
    fp = (pred_mask * (1 - gt_mask)).sum()
    fn = ((1 - pred_mask) * gt_mask).sum()
    
    if tp + fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0
    
    if tp + fn > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0
    
    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0
    
    return {"f1": f1, "prec": prec, "rec": rec}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint
    ckpt_path = Path("experiments/min_dual/checkpoints/epoch_01.pt")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        raise FileNotFoundError(f"Required checkpoint missing: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Load model
    model = UNet(in_ch=3, base_ch=32).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)
    
    # Load state dict
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    
    model.eval()
    head.eval()
    print(f"✓ Checkpoint loaded from: {ckpt_path}")
    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    
    # Load TW25 validation set
    print("\nLoading TW25 validation set...")
    val_set = TW25SegDataset(
        data_root="data",
        json_path="data/tw25/tw25_train.json",
        patch_cfg=PatchCfg(size=768),
        is_train=False,
    )
    print(f"Validation set size: {len(val_set)}")
    
    # Inference loop (with timeout)
    loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)
    
    all_metrics = []
    max_batches = 10  # Just test first 10 batches
    
    print(f"\nRunning inference on {max_batches * 4} samples...")
    start = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            ignore = batch.get("ignore", torch.zeros_like(mask)).to(device)
            
            # Forward
            feat = model(img)
            logit = head(feat)
            
            # Post-process
            pred = torch.sigmoid(logit) > 0.5
            pred_np = pred.cpu().numpy().astype(np.uint8)
            mask_np = mask.cpu().numpy().astype(np.uint8)
            ignore_np = ignore.cpu().numpy().astype(np.uint8)
            
            # Compute metrics
            for i in range(pred_np.shape[0]):
                metrics = compute_metrics(pred_np[i, 0], mask_np[i, 0], ignore_np[i, 0])
                all_metrics.append(metrics)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}: Processed {(batch_idx + 1) * 4} samples")
    
    elapsed = time.time() - start
    
    # Summary
    avg_f1 = np.mean([m["f1"] for m in all_metrics])
    avg_prec = np.mean([m["prec"] for m in all_metrics])
    avg_rec = np.mean([m["rec"] for m in all_metrics])
    
    print(f"\n{'='*50}")
    print(f"Results on {len(all_metrics)} validation patches:")
    print(f"  Pixel-F1:   {avg_f1:.4f}")
    print(f"  Precision:  {avg_prec:.4f}")
    print(f"  Recall:     {avg_rec:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
