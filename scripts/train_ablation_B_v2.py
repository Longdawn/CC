"""
消融实验 Variant B: 双流无约束
- 文本分支 + 背景分支
- 有BG-view
- 无BG抑制（lambda_bg=0）
- 无正交约束（lambda_ortho=0）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from datasets.tw25_dataset import TW25SegDataset, PatchCfg
from models.dual_detector import DualHeadDetector
from losses.seg_loss import SegLoss


def train_dual_stream(model, loss_fn, train_set, device, exp_dir, epochs=30, batch_size=4, lr=1e-4):
    """
    双流训练：文本分支处理原图，背景分支处理BG-view
    但不使用正交约束（lambda_ortho=0）
    """
    model.train()
    
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    exp_dir = Path(exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        count = 0
        
        for batch in loader:
            img = batch["image"].to(device)
            img_bg = batch["image_bg"].to(device)  # BG-view
            mask = batch["mask"].to(device)
            ignore = batch["ignore"].to(device)
            
            # 前向：双流
            logit_text, logit_bg = model(img, img_bg, return_features=False)
            
            # 损失：
            # - 文本分支：在原图上学文本
            L_text = loss_fn(logit_text, mask, ignore)
            
            # - 背景分支：无约束（不参与损失）
            # 这里不加 L_bg 和 L_ortho
            loss = L_text
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            epoch_loss += loss.item()
            count += 1
            
            if step % 200 == 0:
                print(f"  Step {step:4d} | Loss={loss.item():.4f} (Text={L_text.item():.4f})")
            
            step += 1
        
        avg_loss = epoch_loss / count
        print(f"[Epoch {epoch:2d}/{epochs}] Avg Loss={avg_loss:.4f}")
        
        # 保存checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "config": {
                "variant": "B_dual_no_constraint",
                "lambda_bg": 0.0,
                "lambda_ortho": 0.0,
            }
        }, ckpt_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # 准备数据
    patch_cfg = PatchCfg(size=768)
    train_set = TW25SegDataset(
        data_root="data",
        json_path="data/TW25/tw25_train.json",
        patch_cfg=patch_cfg,
        is_train=True,
    )
    
    # 模型和损失
    model = DualHeadDetector(in_ch=3, base_ch=32).to(device)
    loss_fn = SegLoss()
    
    print("=" * 70)
    print("ABLATION VARIANT B: Dual-Stream (No Orthogonality)")
    print("  - 文本分支 + 背景分支")
    print("  - 有BG-view")
    print("  - λ_bg = 0 (无BG抑制)")
    print("  - λ_ortho = 0 (无正交约束)")
    print("=" * 70 + "\n")
    
    train_dual_stream(
        model=model,
        loss_fn=loss_fn,
        train_set=train_set,
        device=device,
        exp_dir="experiments/ablation_B_dual_no_constraint",
        epochs=30,
        batch_size=4,
        lr=1e-4,
    )
    
    print("\n✓ Training completed!")


if __name__ == "__main__":
    main()
