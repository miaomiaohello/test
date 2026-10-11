# -*- coding: utf-8 -*-
"""前景提取变体对比: close5/close7/close7+fillhole 对 Pass A 分布的影响"""
import glob
import os
import cv2
import numpy as np
from scipy import ndimage
from run_real_pipeline import load_case, D
from crossing_partition import crossing_partition


def make_fg(gray, close_k, fill):
    m = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    fg = m > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    if fill:
        fg = ndimage.binary_fill_holes(fg)
    lab, n = ndimage.label(fg)
    if n:
        keep = np.bincount(lab.ravel())[1:] >= 20
        fg = keep[lab - 1] & (lab > 0)
    return fg


variants = [("close5", 5, False), ("close7", 7, False),
            ("close7+fill", 7, True)]
paths = sorted(glob.glob(os.path.join(D, "image_*.png")),
               key=lambda p: int(os.path.basename(p)[6:-4]))

print(f"{'variant':>12s} | " + " | ".join(
    f"{k}条" for k in range(1, 7)) + " | >=7条")
for tag, ck, fl in variants:
    stat = []
    v_ok = []
    for p in paths:
        _, img, gray, v, arms = load_case(p)
        fg = make_fg(gray, ck, fl)
        masks, _ = crossing_partition(fg, v, d=8, return_debug=True)
        stat.append(len(masks))
        u = np.logical_or.reduce(masks)
        v_ok.append((u & v).sum() / max(v.sum(), 1))
    stat = np.array(stat)
    row = [f"{(stat == k).sum():3d}" for k in range(1, 7)]
    ge7 = f"{(stat >= 7).sum():3d}"
    print(f"{tag:>12s} | " + " | ".join(row) + f" | {ge7}   "
          f"V覆盖={np.mean(v_ok):.3f}")
