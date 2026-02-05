"""
使用真实缓存的 TW25 数据进行完整训练
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from datasets.tw25_dataset import TW25DatasetCache
from models.dual_head import DualHeadDetector
from utils.loss import compute_loss
from utils.metrics import compute_metrics
import time

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    
    for idx, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        gt_seg = batch["gt_seg"].to(device)
        gt_bg = batch["gt_bg"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(image)
        loss_dict = compute_loss(outputs, gt_seg, gt_bg)
        loss = loss_dict["total"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Batch {idx + 1}/{len(dataloader)}: Loss={loss.item():.4f}")
    
    return total_loss / count if count > 0 else 0.0

def validate(model, dataloader, device):
    model.eval()
    all_metrics = {"pixel_f1": [], "boundary_f1": []}
    
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            gt_seg = batch["gt_seg"].to(device)
            
            outputs = model(image)
            pred_seg = outputs["seg"].softmax(dim=1).argmax(dim=1)
            
            metrics = compute_metrics(pred_seg.cpu(), gt_seg.cpu())
            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(metrics[key])
    
    # 计算平均指标
    avg_metrics = {}
    for key, vals in all_metrics.items():
        if vals:
            avg_metrics[key] = sum(vals) / len(vals)
    
    return avg_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 参数设置
    num_epochs = 5
    batch_size = 8
    learning_rate = 1e-3
    
    # 数据加载
    print("\n=== Loading Training Data ===")
    train_dataset = TW25DatasetCache(
        data_root=Path("data/TW25"),
        split="train",
        use_cache=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Train loader batches: {len(train_loader)}")
    
    # 模型、优化器、损失函数
    model = DualHeadDetector(
        backbone="resnet50",
        num_classes=2,
        dropout_rate=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 检查点保存目录
    ckpt_dir = Path("experiments/real_dual/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Starting Training ===")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        epoch_time = time.time() - epoch_start
        
        scheduler.step()
        
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        
        # 保存检查点
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")
    
    total_time = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    # 保存最终模型
    final_model_path = ckpt_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

if __name__ == "__main__":
    main()
