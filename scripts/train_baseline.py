"""
Baseline 训练脚本（λ_ortho=0，不使用 decorrelation）
目的：作为对比基准，验证 dual-stream orthogonality 的效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets.tw25_dataset import TW25SegDataset, PatchCfg
from models.unet import UNet
from models.head import SegHead
from losses.seg_loss import SegLoss
from engine.trainer_dual import train_dual_with_lambda_ortho


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_cfg = PatchCfg(size=768)

    train_set = TW25SegDataset(
        data_root="data",
        json_path="data/TW25/tw25_train.json",
        patch_cfg=patch_cfg,
        is_train=True,
    )

    model = UNet(in_ch=3, base_ch=32).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)
    loss_fn = SegLoss()

    print("=" * 60)
    print("BASELINE TRAINING (λ_ortho = 0)")
    print("=" * 60)

    train_dual_with_lambda_ortho(
        model=model,
        head=head,
        loss_fn=loss_fn,
        train_set=train_set,
        device=device,
        exp_dir="experiments/baseline",
        epochs=2,  # 和 ours 一致
        batch_size=4,
        lr=1e-4,
        lambda_bg=0.4,
        lambda_ortho=0.0,  # ← 关键：不使用 orthogonality
    )


if __name__ == "__main__":
    main()
