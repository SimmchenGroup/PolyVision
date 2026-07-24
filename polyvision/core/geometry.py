from __future__ import annotations

from typing import Iterable


def square_bbox(bbox, img_shape, margin: int = 0):
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r
    w = max_c - min_c
    side = max(h, w) + 2 * margin

    c_r = (min_r + max_r) / 2
    c_c = (min_c + max_c) / 2

    new_min_r = int(round(c_r - side / 2))
    new_max_r = int(round(c_r + side / 2))
    new_min_c = int(round(c_c - side / 2))
    new_max_c = int(round(c_c + side / 2))

    H, W = img_shape[:2]
    new_min_r = max(new_min_r, 0)
    new_min_c = max(new_min_c, 0)
    new_max_r = min(new_max_r, H)
    new_max_c = min(new_max_c, W)

    return new_min_r, new_min_c, new_max_r, new_max_c


def bbox_iou_rc(a, b) -> float:
    a_min_r, a_min_c, a_max_r, a_max_c = a
    b_min_r, b_min_c, b_max_r, b_max_c = b

    inter_min_r = max(a_min_r, b_min_r)
    inter_min_c = max(a_min_c, b_min_c)
    inter_max_r = min(a_max_r, b_max_r)
    inter_max_c = min(a_max_c, b_max_c)

    inter_h = max(0, inter_max_r - inter_min_r)
    inter_w = max(0, inter_max_c - inter_min_c)
    inter = inter_h * inter_w

    a_area = max(0, a_max_r - a_min_r) * max(0, a_max_c - a_min_c)
    b_area = max(0, b_max_r - b_min_r) * max(0, b_max_c - b_min_c)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms_dets_class_agnostic(dets, iou_thresh: float = 0.6):
    if not dets:
        return []

    dets_sorted = sorted(dets, key=lambda d: float(d["conf"]), reverse=True)
    kept = []
    for d in dets_sorted:
        b = d["bbox_rc"]
        if all(bbox_iou_rc(b, k["bbox_rc"]) < iou_thresh for k in kept):
            kept.append(d)
    return kept