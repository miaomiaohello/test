# -*- coding: utf-8 -*-
"""探查 image6/16: 两个 mask 的面积/骨架, 为何 mask1 判断为 -inf"""
import os
import cv2
import numpy as np
from scipy import ndimage
from run_real_pipeline import load_case, extract_foreground, D
from crossing_partition import crossing_partition
from layer_order import _skeleton_path

for n in ("6", "16", "5"):
    p = os.path.join(D, f"image_{n}.png")
    _, img, gray, v, arms = load_case(p)
    fg = extract_foreground(gray)
    masks, dbg = crossing_partition(fg, v, d=8, return_debug=True)
    print(f"\nimage {n}: fg={fg.sum()}px, V={v.sum()}px, "
          f"输出 {len(masks)} 条")
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        path = _skeleton_path(m)
        plen = len(path) if path is not None else 0
        print(f"  mask{i}: area={m.sum():5d} "
              f"bbox=x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] "
              f"skeleton_path={plen}")
    print(f"  S块数={len(dbg['S'])}, V块数={len(dbg['V'])}, "
          f"D={dbg['D']}")
