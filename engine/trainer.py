from __future__ import annotations
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.io import ensure_dir
from utils.viz import save_vis


def train_loop(
    model, head, loss_fn,
    train_set, val_set,
    cfg: dict, device: torch.device,
    exp_dir: str
):
    ensure_dir(exp_dir)
    ckpt_dir = str(Path(exp_dir) / "checkpoints")
    vis_dir = str(Path(exp_dir) / "vis")
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    writer = SummaryWriter(log_dir=str(Path(exp_dir) / "logs"))

    bs = int(cfg["train"]["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", True))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"]["weight_decay"])
    optim = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)))

    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every"])
    vis_every = int(cfg["train"]["vis_every"])
    save_every = int(cfg["train"]["save_every"])

    global_step = 0
    model.train()
    head.train()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ignore = batch["ignore"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True))):
                feat = model(img)
                logits = head(feat)
                loss = loss_fn(logits, mask, ignore)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if global_step % log_every == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)

            if global_step % vis_every == 0:
                with torch.no_grad():
                    pred = torch.sigmoid(logits[0]).clamp(0, 1)
                    save_vis(vis_dir, global_step, img[0], mask[0], pred)

            global_step += 1

        # save checkpoint
        if epoch % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, str(Path(ckpt_dir) / f"epoch_{epoch:03d}.pt"))

        dt = time.time() - t0
        writer.add_scalar("train/epoch_time_sec", dt, epoch)

    writer.close()
