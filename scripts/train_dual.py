import torch
from pathlib import Path

from datasets.tw25_dataset import TW25SegDataset, PatchCfg
from models.unet import UNet
from models.head import SegHead
from losses.seg_loss import SegLoss
from engine.trainer_dual import train_dual


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_cfg = PatchCfg(size=768)

    train_set = TW25SegDataset(
        data_root="data",
        json_path="data/tw25/tw25_train.json",
        patch_cfg=patch_cfg,
        is_train=True,
    )

    model = UNet(in_ch=3, base_ch=32).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)
    loss_fn = SegLoss()

    train_dual(
        model=model,
        head=head,
        loss_fn=loss_fn,
        train_set=train_set,
        device=device,
        exp_dir="experiments/min_dual",
        epochs=5,
        batch_size=4,
        lr=1e-4,
        lambda_bg=0.4,
    )


if __name__ == "__main__":
    main()
