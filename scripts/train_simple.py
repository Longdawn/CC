"""
轻量级训练脚本：不生成背景视图，直接用原图
用于快速验证框架，生成 checkpoint
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from models.unet import UNet
from models.head import SegHead
from losses.seg_loss import SegLoss


class SimpleSegDataset(Dataset):
    """简化版数据集：仅用原图，不生成背景视图"""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机图像和掩码用于演示
        h, w = 768, 768
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (h, w), dtype=np.uint8)
        
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask[None, ...]).float()
        
        return {
            "image": img_t,
            "image_bg": img_t,  # 用原图代替背景视图
            "mask": mask_t,
            "ignore": torch.zeros_like(mask_t),
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create simple dataset (no I/O bottleneck)
    train_set = SimpleSegDataset(num_samples=100)
    loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    
    # Create model
    model = UNet(in_ch=3, base_ch=32).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)
    loss_fn = SegLoss()
    
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-4
    )
    
    # Output directory
    exp_dir = Path("experiments/min_dual")
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Quick training (2 epochs)
    model.train()
    head.train()
    
    for epoch in range(2):
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            img = batch["image"].to(device)
            img_bg = batch["image_bg"].to(device)
            mask = batch["mask"].to(device)
            ignore = batch["ignore"].to(device)
            
            zeros = torch.zeros_like(mask)
            
            # Forward
            feat_t = model(img)
            logit_t = head(feat_t)
            
            feat_b = model(img_bg)
            logit_b = head(feat_b)
            
            # Loss
            L_text = loss_fn(logit_t, mask, ignore)
            L_bg = loss_fn(logit_b, zeros, ignore * 0 + 1.0)  # ignore everything in BG
            loss = L_text + 0.4 * L_bg
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} Batch {batch_idx + 1}: loss={loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "head": head.state_dict(),
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")
    
    print("\nTraining complete! Checkpoint saved.")
    print(f"Next: python -m scripts.infer_tw25")


if __name__ == "__main__":
    main()
