import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from utils.viz import save_vis


def decorrelation_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """
    feat_a, feat_b: [B, C, H, W]
    目标：让两个分支的通道表示"尽量不相关"
    做法：对每个样本做通道相关矩阵，然后取 off-diagonal 的 L1
    """
    B, C, H, W = feat_a.shape
    xa = feat_a.flatten(2)  # [B, C, HW]
    xb = feat_b.flatten(2)  # [B, C, HW]

    # 去均值
    xa = xa - xa.mean(dim=2, keepdim=True)
    xb = xb - xb.mean(dim=2, keepdim=True)

    # 通道归一化（避免尺度影响）
    xa = F.normalize(xa, dim=2)
    xb = F.normalize(xb, dim=2)

    # cross-correlation: [B, C, C]
    corr = torch.bmm(xa, xb.transpose(1, 2)) / (H * W)

    # 我们希望 corr 接近 0（全零矩阵）
    return corr.abs().mean()


def train_dual(
    model, head, loss_fn,
    train_set,
    device,
    exp_dir,
    epochs=5,
    batch_size=4,
    lr=1e-4,
    lambda_bg=0.4,
):
    model.train()
    head.train()

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=lr
    )

    exp_dir = Path(exp_dir)
    vis_dir = exp_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(epochs):
        for batch in loader:
            img = batch["image"].to(device)
            img_bg = batch["image_bg"].to(device)
            mask = batch["mask"].to(device)
            ignore = batch["ignore"].to(device)

            zeros = torch.zeros_like(mask)

            # ---- forward ----
            feat_t = model(img)
            logit_t = head(feat_t)

            feat_b = model(img_bg)
            logit_b = head(feat_b)

            # ---- loss ----
            L_text = loss_fn(logit_t, mask, ignore)
            bg_ignore = 1.0 - mask  # 只在文本区域强制 BG=0
            L_bg = loss_fn(logit_b, zeros, bg_ignore)
            L_ortho = decorrelation_loss(feat_t, feat_b)
            
            lambda_ortho = 0.01
            loss = L_text + lambda_bg * L_bg + lambda_ortho * L_ortho

            optim.zero_grad()
            loss.backward()
            optim.step()

            # ---- viz ----
            if step % 200 == 0:
                print(f"  Step {step} | L_text={float(L_text.item()):.4f} | L_bg={float(L_bg.item()):.4f} | L_ortho={L_ortho.item():.12e}")
                print(f"    feat_t abs mean: {feat_t.abs().mean().item():.6f} | feat_b abs mean: {feat_b.abs().mean().item():.6f}")
                
                # 计算相关矩阵用于诊断
                with torch.no_grad():
                    B, C, H, W = feat_t.shape
                    xa = feat_t.flatten(2)
                    xb = feat_b.flatten(2)
                    xa = xa - xa.mean(dim=2, keepdim=True)
                    xb = xb - xb.mean(dim=2, keepdim=True)
                    xa = F.normalize(xa, dim=2)
                    xb = F.normalize(xb, dim=2)
                    corr = torch.bmm(xa, xb.transpose(1, 2)) / (H * W)
                    print(f"    corr abs mean: {corr.abs().mean().item():.12e} | corr abs max: {corr.abs().max().item():.12e}")
                
                with torch.no_grad():
                    pred_t = torch.sigmoid(logit_t[0])
                    pred_b = torch.sigmoid(logit_b[0])
                    save_vis(vis_dir, step, img[0], mask[0], pred_t)
                    save_vis(vis_dir, step + 1, img_bg[0], zeros[0], pred_b)

            step += 1

        print(f"[Epoch {epoch}] loss={loss.item():.4f}")
        
        # Save checkpoint every epoch
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "head": head.state_dict(),
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")


def train_dual_with_lambda_ortho(
    model, head, loss_fn,
    train_set,
    device,
    exp_dir,
    epochs=5,
    batch_size=4,
    lr=1e-4,
    lambda_bg=0.4,
    lambda_ortho=0.01,  # ← 可配置的 orthogonality 系数
):
    """
    Enhanced dual-stream training with configurable lambda_ortho.
    Allows ablation study: set lambda_ortho=0 for baseline.
    """
    model.train()
    head.train()

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=lr
    )

    exp_dir = Path(exp_dir)
    vis_dir = exp_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(epochs):
        for batch in loader:
            img = batch["image"].to(device)
            img_bg = batch["image_bg"].to(device)
            mask = batch["mask"].to(device)
            ignore = batch["ignore"].to(device)

            zeros = torch.zeros_like(mask)

            # ---- forward ----
            feat_t = model(img)
            logit_t = head(feat_t)

            feat_b = model(img_bg)
            logit_b = head(feat_b)

            # ---- loss ----
            L_text = loss_fn(logit_t, mask, ignore)
            bg_ignore = 1.0 - mask
            L_bg = loss_fn(logit_b, zeros, bg_ignore)
            L_ortho = decorrelation_loss(feat_t, feat_b)
            
            # Loss = L_text + lambda_bg * L_bg + lambda_ortho * L_ortho
            loss = L_text + lambda_bg * L_bg + lambda_ortho * L_ortho

            optim.zero_grad()
            loss.backward()
            optim.step()

            # ---- viz ----
            if step % 200 == 0:
                print(f"  Step {step} | L_text={float(L_text.item()):.4f} | L_bg={float(L_bg.item()):.4f} | L_ortho={L_ortho.item():.12e} | lambda_ortho={lambda_ortho}")
                print(f"    feat_t abs mean: {feat_t.abs().mean().item():.6f} | feat_b abs mean: {feat_b.abs().mean().item():.6f}")
                
                with torch.no_grad():
                    pred_t = torch.sigmoid(logit_t[0])
                    pred_b = torch.sigmoid(logit_b[0])
                    save_vis(vis_dir, step, img[0], mask[0], pred_t)
                    save_vis(vis_dir, step + 1, img_bg[0], zeros[0], pred_b)

            step += 1

        print(f"[Epoch {epoch}] loss={loss.item():.4f} (lambda_ortho={lambda_ortho})")
        
        # Save checkpoint every epoch
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "head": head.state_dict(),
            "lambda_ortho": lambda_ortho,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

