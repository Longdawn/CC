"""
消融实验 Variant A: 单流基线
- 只有文本分支（无背景分支）
- 无BG-view
- 无正交约束
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from datasets.tw25_dataset import TW25SegDataset, PatchCfg
from models.dual_detector import DualHeadDetector
from losses.seg_loss import SegLoss


def train_single_stream(model, loss_fn, train_set, device, exp_dir, epochs=30, batch_size=4, lr=1e-4):
    """
    单流训练：只用文本分支，处理原图
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
            mask = batch["mask"].to(device)
            ignore = batch["ignore"].to(device)
            
            # 前向：只用原图，推理模式（不需要bg_view）
            logit_text = model(img)
            
            # 损失：只有文本分割损失
            loss = loss_fn(logit_text, mask, ignore)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            epoch_loss += loss.item()
            count += 1
            
            if step % 200 == 0:
                print(f"  Step {step:4d} | Loss={loss.item():.4f}")
            
            step += 1
        
        avg_loss = epoch_loss / count
        print(f"[Epoch {epoch:2d}/{epochs}] Avg Loss={avg_loss:.4f}")
        
        # 保存checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "config": {
                "variant": "A_single_stream",
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
    print("ABLATION VARIANT A: Single-Stream Baseline")
    print("  - 只有文本分支")
    print("  - 无背景分支")
    print("  - 无BG-view")
    print("  - 无正交约束")
    print("=" * 70 + "\n")
    
    train_single_stream(
        model=model,
        loss_fn=loss_fn,
        train_set=train_set,
        device=device,
        exp_dir="experiments/ablation_A_single_stream",
        epochs=30,
        batch_size=4,
        lr=1e-4,
    )
    
    print("\n✓ Training completed!")


if __name__ == "__main__":
    main()
