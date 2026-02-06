"""
分割结果后处理为检测框
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2


def prob_to_boxes(prob_map: np.ndarray, thresh: float = 0.5, min_area: int = 50) -> List[Tuple[int, int, int, int]]:
    """
    将概率图转为检测框 (x, y, w, h)

    Args:
        prob_map: (H, W) 浮点概率图
        thresh: 二值化阈值
        min_area: 过滤小区域

    Returns:
        boxes: List[(x, y, w, h)]
    """
    bin_map = (prob_map >= thresh).astype(np.uint8) * 255

    # 轻微去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_map = cv2.morphologyEx(bin_map, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    return boxes
