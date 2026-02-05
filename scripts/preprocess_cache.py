"""
预处理脚本：将 TW25 JPEG 图像缓存为 NumPy 格式以加快数据加载
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import cv2
import time

def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    data_root = Path("data")
    json_path = data_root / "TW25" / "tw25_train.json"
    cache_dir = data_root / "TW25" / "image_cache"
    
    if not json_path.exists():
        print(f"ERROR: {json_path} not found!")
        return
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    print(f"Found {len(items)} images in {json_path}")
    print(f"Caching to: {cache_dir}")
    
    start_time = time.time()
    
    cached_count = 0
    for idx, item in enumerate(items):
        img_name = item["image"]  # 格式: "tw25/train/JM20K_..."
        # 移除 "tw25/" 前缀，因为实际文件在 data/TW25/train/...，不是 data/TW25/tw25/train/...
        relative_path = img_name.replace("tw25/", "", 1) if img_name.startswith("tw25/") else img_name
        img_path = data_root / "TW25" / relative_path
        cache_path = cache_dir / f"{img_name.replace('/', '_')}.npy"
        
        if cache_path.exists():
            cached_count += 1
            continue
        
        try:
            img_rgb = _load_rgb(img_path)
            np.save(cache_path, img_rgb)
            if (idx + 1) % 100 == 0:
                print(f"  [{idx + 1}/{len(items)}] Cached...")
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nCached {len(items) - cached_count} new images ({cached_count} already cached)")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average per image: {elapsed / len(items) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
