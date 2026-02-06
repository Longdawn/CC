"""
完整消融实验评估 - 新版本
评估重设计后的4组实验：
  A: 单流基线
  B: 双流无约束
  C: 双流+正交约束
  D: 参考（单流）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from torch.utils.data import DataLoader
from datasets.tw25_dataset import TW25SegDataset, PatchCfg
from models.dual_detector import DualHeadDetector


def evaluate_checkpoint(ckpt_path, device, val_loader, model_class=None):
    """评估单个checkpoint"""
    
    # 加载checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 创建模型
    if model_class is None:
        model = DualHeadDetector(in_ch=3, base_ch=32).to(device)
    else:
        model = model_class().to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # 全局统计
    global_tp, global_fp, global_fn = 0, 0, 0
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            gts = batch["mask"].to(device)
            ignore = batch["ignore"].to(device)
            
            # 推理：只用原图（推理模式）
            logits = model(imgs)
            preds = (logits.sigmoid() > 0.5).long()
            
            # 计算有效区域
            valid = (ignore == 0)
            
            # 累积全局统计
            tp = ((preds == 1) & (gts == 1) & valid).sum().item()
            fp = ((preds == 1) & (gts == 0) & valid).sum().item()
            fn = ((preds == 0) & (gts == 1) & valid).sum().item()
            
            global_tp += tp
            global_fp += fp
            global_fn += fn
    
    # 计算全局指标
    precision = global_tp / (global_tp + global_fp + 1e-8)
    recall = global_tp / (global_tp + global_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = global_tp / (global_tp + global_fp + global_fn + 1e-8)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "epoch": ckpt.get("epoch", -1),
        "config": ckpt.get("config", {}),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 准备验证集
    patch_cfg = PatchCfg(size=768)
    val_set = TW25SegDataset(
        data_root="data",
        json_path="data/TW25/tw25_val.json",
        patch_cfg=patch_cfg,
        is_train=False,
    )
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4)
    
    print(f"Validation set: {len(val_set)} patches\n")
    
    # 定义要评估的checkpoints
    checkpoints = [
        {
            "path": "experiments/ablation_A_single_stream/checkpoints/epoch_29.pt",
            "desc": "A: Single-Stream Baseline",
        },
        {
            "path": "experiments/ablation_B_dual_no_constraint/checkpoints/epoch_29.pt",
            "desc": "B: Dual-Stream (No Constraints)",
        },
        {
            "path": "experiments/ablation_C_dual_bg/checkpoints/epoch_29.pt",
            "desc": "C: Dual + BG Suppression",
        },
        {
            "path": "experiments/ablation_D_full_method/checkpoints/epoch_29.pt",
            "desc": "D: Full Method (Ours)",
        },
    ]
    
    # 评估所有checkpoints
    results = []
    print("Evaluating checkpoints...\n")
    
    for ckpt_info in checkpoints:
        ckpt_path = Path(ckpt_info["path"])
        
        if not ckpt_path.exists():
            print(f"⚠ Checkpoint not found: {ckpt_path}")
            print(f"   Skipping: {ckpt_info['desc']}\n")
            continue
        
        print(f"Evaluating: {ckpt_info['desc']}")
        print(f"  Path: {ckpt_path}")
        
        result = evaluate_checkpoint(ckpt_path, device, val_loader)
        result["desc"] = ckpt_info["desc"]
        results.append(result)
        
        epoch = result["epoch"]
        config = result["config"]
        print(f"  Epoch: {epoch}")
        print(f"  Config: {config}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1: {result['f1']:.4f}")
        print(f"  IoU: {result['iou']:.4f}\n")
    
    if len(results) == 0:
        print("No checkpoints found. Please train the models first.")
        return
    
    # 打印对比表格
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS (New Design)")
    print("=" * 100)
    print(f"{'Variant':<45} {'Precision':>12} {'Recall':>12} {'F1':>12} {'IoU':>12}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['desc']:<45} "
              f"{result['precision']:>12.4f} "
              f"{result['recall']:>12.4f} "
              f"{result['f1']:>12.4f} "
              f"{result['iou']:>12.4f}")
    
    # 计算提升
    print("-" * 100)
    if len(results) >= 3:
        baseline = results[0]  # A
        final = results[2]     # C
        
        f1_diff = final['f1'] - baseline['f1']
        f1_pct = (f1_diff / (baseline['f1'] + 1e-8)) * 100
        
        print(f"Improvement (C vs A): F1 {f1_diff:+.4f} ({f1_pct:+.2f}%)")
        print(f"  - Precision: {final['precision']-baseline['precision']:+.4f}")
        print(f"  - Recall: {final['recall']-baseline['recall']:+.4f}")
        print(f"  - IoU: {final['iou']-baseline['iou']:+.4f}")
    
    print("=" * 100 + "\n")
    
    # 保存结果为JSON
    output_path = Path("experiments/ablation_results_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
