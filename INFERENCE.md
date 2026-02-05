# CC-TextDet 推理框架

## 推理脚本：`scripts/infer_tw25.py`

### 功能
- ✅ 加载 TW25 验证集（`is_train=False` 无 patch crop）
- ✅ 逐张推理，输出二值化预测（阈值 0.5）
- ✅ 计算 Pixel-F1、Precision、Recall
- ✅ 保存预测 PNG 到 `experiments/infer_tw25_val/predictions/`
- ✅ 保存前 10 张的三联图（image/gt/pred）到 `experiments/infer_tw25_val/visualizations/`

### 运行
```bash
conda activate cc
cd D:\LG\CC-TextDet
python -m scripts.infer_tw25
```

### 输出示例
```
Inferencing on 1478 images...
  [10/50] F1=0.0699
  [20/50] F1=0.0795
  ...
  [50/50] F1=0.0481

============================================================
Inference complete. Results saved to: experiments\infer_tw25_val
Pixel-F1 (mean): 0.0645
Precision (mean): 0.0336
Recall (mean): 1.0000
============================================================
```

## 下一步

### 1. 使用训练好的权重
当 `train_dual.py` 训练完成后，checkpoint 会自动保存到：
```
experiments/min_dual/checkpoints/epoch_00.pt
experiments/min_dual/checkpoints/epoch_01.pt
...
```

推理脚本会自动加载最新的 checkpoint。

### 2. 对比 Baseline（Single-Stream）
创建 `scripts/infer_baseline.py`：
```python
# 不使用 image_bg，只用 image 推理
model = UNet(...)
logit = head(model(img))
pred = sigmoid(logit) > 0.5
```

然后对比：
- Dual-stream F1 vs Single-stream F1
- 性能提升幅度

### 3. 结果保存位置
```
experiments/
├── infer_tw25_val/
│   ├── predictions/      # 50 张预测 PNG
│   └── visualizations/   # 前 10 张三联图
└── min_dual/
    └── checkpoints/      # 训练 checkpoint
```

## 指标说明
- **Pixel-F1**：所有像素的 F1 分数，是文本分割最重要的指标
- **Precision**：真正例 / (真正例 + 假正例)
- **Recall**：真正例 / (真正例 + 假负例)

---

现在推理框架已就位，等待 checkpoint 生成即可验证完整的 Dual-Stream 性能！
