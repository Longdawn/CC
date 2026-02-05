import torch
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.tw25_dataset import TW25SegDataset, PatchCfg
from models.unet import UNet
from models.head import SegHead


def compute_pixel_f1(pred, target, threshold=0.5):
    """
    pred: [H, W] float in [0, 1]
    target: [H, W] binary {0, 1}
    returns: F1 score
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    
    tp = np.sum(pred_bin & target_bin)
    fp = np.sum(pred_bin & ~target_bin)
    fn = np.sum(~pred_bin & target_bin)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return f1, precision, recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = UNet(in_ch=3, base_ch=32).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)
    
    # Load checkpoint (fixed path)
    ckpt_path = Path("experiments/min_dual/checkpoints/epoch_01.pt")
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "head" in ckpt:
                head.load_state_dict(ckpt["head"])
            print(f"✓ Checkpoint loaded successfully from epoch {ckpt.get('epoch', '?')}")
        else:
            print(f"ERROR: Unexpected checkpoint format at {ckpt_path}")
            raise RuntimeError(f"Cannot load checkpoint from {ckpt_path}")
    else:
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        raise FileNotFoundError(f"Required checkpoint missing: {ckpt_path}")
    
    model.eval()
    head.eval()
    
    # Load validation dataset
    patch_cfg = PatchCfg(size=768)
    val_set = TW25SegDataset(
        data_root="data",
        json_path="data/tw25/tw25_train.json",  # 使用 train.json (你可改为 val.json)
        patch_cfg=patch_cfg,
        is_train=False,  # 不做 patch crop
    )
    
    # Output directory
    out_dir = Path("experiments/infer_tw25_val")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    vis_dir = out_dir / "visualizations"
    pred_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    # Inference
    metrics = defaultdict(list)
    
    print(f"Inferencing on {len(val_set)} images...")
    for idx in range(min(len(val_set), 50)):  # 限制前 50 张用于演示
        sample = val_set[idx]
        img = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
        mask_gt = sample["mask"][0].cpu().numpy()  # [H, W]
        
        with torch.no_grad():
            feat = model(img)
            logit = head(feat)
            pred = torch.sigmoid(logit[0, 0]).cpu().numpy()  # [H, W]
        
        # Compute F1
        f1, prec, rec = compute_pixel_f1(pred, mask_gt)
        metrics["f1"].append(f1)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        
        # Save prediction as PNG
        pred_uint8 = (pred * 255).astype(np.uint8)
        pred_path = pred_dir / f"pred_{idx:04d}.png"
        cv2.imwrite(str(pred_path), pred_uint8)
        
        # Save visualization (image / gt / pred)
        if idx < 10:  # 仅保存前 10 张的可视化
            h, w = pred.shape
            img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            mask_vis = (mask_gt * 255).astype(np.uint8)
            pred_vis = pred_uint8
            
            # 转为 3 通道用于拼接
            mask_vis_3 = np.stack([mask_vis, mask_vis, mask_vis], 2)
            pred_vis_3 = np.stack([pred_vis, pred_vis, pred_vis], 2)
            
            canvas = np.concatenate([img_np, mask_vis_3, pred_vis_3], axis=1)  # [H, 3W, 3]
            vis_path = vis_dir / f"vis_{idx:04d}.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{min(len(val_set), 50)}] F1={f1:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print(f"Inference complete. Results saved to: {out_dir}")
    print(f"Pixel-F1 (mean): {np.mean(metrics['f1']):.4f}")
    print(f"Precision (mean): {np.mean(metrics['precision']):.4f}")
    print(f"Recall (mean): {np.mean(metrics['recall']):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
