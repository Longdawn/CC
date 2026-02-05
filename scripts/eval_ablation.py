"""
Ablation 评估脚本：同时测试 baseline 和 ours 的 checkpoint
输出：对比表格（Precision / Recall / F1 / IoU）
"""
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
def eval_checkpoint(ckpt_path, device, val_loader, model, head, thr=0.5):
    """评估单个 checkpoint"""
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval()
    head.eval()
    
    TP = 0.0
    FP = 0.0
    FN = 0.0
    
    for batch in val_loader:
        img = batch["image"].to(device)
        gt = batch["mask"].to(device)
        ignore = batch.get("ignore", None)
        
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if ignore is not None and ignore.dim() == 3:
            ignore = ignore.unsqueeze(1)
        
        feat = model(img)
        logit = head(feat)
        prob = torch.sigmoid(logit)
        pred = (prob > thr).float()
        
        if ignore is None:
            valid = torch.ones_like(gt)
        else:
            ignore = ignore.to(device)
            valid = (ignore < 0.5).float()
        
        gt_bin = (gt > 0.5).float()
        
        tp = (pred * gt_bin * valid).sum().item()
        fp = (pred * (1.0 - gt_bin) * valid).sum().item()
        fn = ((1.0 - pred) * gt_bin * valid).sum().item()
        
        TP += tp
        FP += fp
        FN += fn
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备 val set
    patch_cfg = PatchCfg(size=768)
    val_set = TW25SegDataset(
        data_root="data",
        json_path="data/TW25/tw25_val.json",
        patch_cfg=patch_cfg,
        is_train=False,
    )
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # 模型
    model = UNet(in_ch=3, base_ch=32).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)
    
    # 评估配置
    experiments = [
        ("Baseline (λ_ortho=0)", "experiments/baseline/checkpoints/epoch_01.pt"),
        ("Ours (λ_ortho=0.01)", "experiments/ours/checkpoints/epoch_01.pt"),
    ]
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY: Orthogonality Loss Impact")
    print("=" * 80)
    print(f"Dataset: TW25 Validation (patch-level, 768×768)")
    print(f"Device: {device}")
    print("-" * 80)
    
    results = []
    for name, ckpt_path in experiments:
        ckpt_path = Path(ckpt_path)
        
        if not ckpt_path.exists():
            print(f"\n❌ {name}: Checkpoint not found at {ckpt_path}")
            continue
        
        print(f"\n▶ Evaluating: {name}")
        print(f"  Checkpoint: {ckpt_path}")
        
        metrics = eval_checkpoint(ckpt_path, device, val_loader, model, head)
        results.append((name, metrics))
        
        print(f"  ✓ Completed")
    
    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'Pixel-F1':<12} {'IoU':<12}")
    print("-" * 80)
    
    for name, metrics in results:
        print(
            f"{name:<25} "
            f"{metrics['precision']:.4f}        "
            f"{metrics['recall']:.4f}        "
            f"{metrics['f1']:.4f}        "
            f"{metrics['iou']:.4f}"
        )
    
    # Calculate improvement
    if len(results) == 2:
        baseline_f1 = results[0][1]["f1"]
        ours_f1 = results[1][1]["f1"]
        improvement = (ours_f1 - baseline_f1) / (baseline_f1 + 1e-8) * 100
        
        print("-" * 80)
        print(f"\n📊 Improvement (Ours - Baseline):")
        print(f"   F1 gain: {ours_f1 - baseline_f1:+.4f} ({improvement:+.2f}%)")
        print(f"   IoU gain: {results[1][1]['iou'] - results[0][1]['iou']:+.4f}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
