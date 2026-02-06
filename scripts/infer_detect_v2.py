"""
完整检测推理脚本
输入：图片
输出：检测框 (x, y, w, h) + 可视化

使用方式：
  python scripts/infer_detect_v2.py --ckpt <path> --image <path> --out_json <path>
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dual_detector import DualHeadDetector
from utils.detect_postprocess import prob_to_boxes


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sliding_window_predict(img: np.ndarray, model, device, patch_size=768, stride=512):
    """
    滑窗推理，输出整图概率图
    """
    H, W = img.shape[:2]
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    Hp, Wp = img_pad.shape[:2]

    prob_sum = np.zeros((Hp, Wp), dtype=np.float32)
    count = np.zeros((Hp, Wp), dtype=np.float32)

    for y in range(0, Hp - patch_size + 1, stride):
        for x in range(0, Wp - patch_size + 1, stride):
            patch = img_pad[y:y + patch_size, x:x + patch_size, :]
            patch_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_t = patch_t.unsqueeze(0).to(device)

            with torch.no_grad():
                logit = model(patch_t)
                prob = torch.sigmoid(logit).squeeze(0).squeeze(0).cpu().numpy()

            prob_sum[y:y + patch_size, x:x + patch_size] += prob
            count[y:y + patch_size, x:x + patch_size] += 1.0

    prob_map = prob_sum / np.maximum(count, 1e-6)
    prob_map = prob_map[:H, :W]
    return prob_map


def draw_boxes(img: np.ndarray, boxes, color=(0, 255, 0), thickness=2):
    vis = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--out_json", type=str, default="outputs/detect.json", help="Output JSON file")
    parser.add_argument("--out_vis", type=str, default="outputs/detect_vis.jpg", help="Output visualization image")
    parser.add_argument("--thresh", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--min_area", type=int, default=50, help="Min area for boxes")
    parser.add_argument("--patch", type=int, default=768, help="Patch size")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DualHeadDetector(in_ch=3, base_ch=32).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load image
    img = load_rgb(Path(args.image))

    # Predict
    prob_map = sliding_window_predict(img, model, device, patch_size=args.patch, stride=args.stride)

    # Post-process to boxes
    boxes = prob_to_boxes(prob_map, thresh=args.thresh, min_area=args.min_area)

    # Save JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"image": str(args.image), "boxes": boxes}, f, ensure_ascii=False, indent=2)

    # Save visualization
    vis = draw_boxes(img, boxes)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    out_vis = Path(args.out_vis)
    out_vis.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_vis), vis_bgr)

    print(f"Detected {len(boxes)} boxes")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_vis}")


if __name__ == "__main__":
    main()
