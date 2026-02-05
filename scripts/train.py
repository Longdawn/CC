from __future__ import annotations
import argparse
from pathlib import Path

import torch

from utils.io import load_config, ensure_dir
from utils.seed import set_seed
from datasets.det_dataset import DetSegDataset, PatchCfg
from models.backbones import UNet
from models.heads import SegHead
from losses.seg_loss import SegLoss
from engine.trainer import train_loop
from datasets.tw25_dataset import TW25SegDataset, PatchCfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/det_baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    exp_name = cfg.get("exp_name", "exp")
    exp_dir = str(Path(cfg.get("log_dir", "experiments")) / exp_name)
    ensure_dir(exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_cfg = PatchCfg(
        size=int(cfg["train_patch"]["size"]),
        text_center_prob=float(cfg["train_patch"]["text_center_prob"]),
        max_tries=int(cfg["train_patch"]["max_tries"]),
        min_text_pixels=int(cfg["train_patch"]["min_text_pixels"]),
    )

    train_set = TW25SegDataset(
        data_root=cfg["data"]["data_root"],
        json_path=cfg["data"]["train_json"],
        patch_cfg=patch_cfg,
        is_train=True,
    )
    val_set = TW25SegDataset(
        data_root=cfg["data"]["data_root"],
        json_path=cfg["data"]["val_json"],
        patch_cfg=patch_cfg,
        is_train=False,
    )

    model = UNet(in_ch=int(cfg["model"]["in_ch"]), base_ch=int(cfg["model"]["base_ch"])).to(device)
    head = SegHead(in_ch=model.out_ch).to(device)
    loss_fn = SegLoss()

    train_loop(model, head, loss_fn, train_set, val_set, cfg, device, exp_dir)


if __name__ == "__main__":
    main()
