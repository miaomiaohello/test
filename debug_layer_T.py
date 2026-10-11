# -*- coding: utf-8 -*-
"""调试 T 用例上下层判断: 打印 mask 归属、交叉样本位置与边缘值"""
import numpy as np
from demo_crossing_partition import build_cases, render, evaluate
from crossing_partition import crossing_partition
from layer_order import (_skeleton_path, _resample, _in_mask,
                         _sample_bilinear)
from scipy import ndimage

rng = np.random.default_rng(42)
size, cases = build_cases(rng)
cfg = cases["T"]
img, gts, fg, gt_inter = render(size, cfg["strips"], np.random.default_rng(42))
masks, dbg = crossing_partition(fg, gt_inter, d=10, return_debug=True)
miou, pairs = evaluate(masks, gts)
print("pairs:", pairs)

for i, m in enumerate(masks):
    ys, xs = np.nonzero(m)
    print(f"mask{i}: area={m.sum()}, bbox x[{xs.min()},{xs.max()}] "
          f"y[{ys.min()},{ys.max()}]")

# 复刻 judge 的采样, 打印每个交叉样本的坐标和边缘值
imgf = img.astype(np.float64)
for i in range(2):
    X, O = masks[i], masks[1 - i]
    path = _skeleton_path(X)
    pts, nrm = _resample(path)
    dt = ndimage.distance_transform_edt(X)
    inside_o = _in_mask(O, pts[:, 0], pts[:, 1])
    arm_sel = (~inside_o)
    h = float(np.median(dt[pts[arm_sel, 1].astype(int),
                           pts[arm_sel, 0].astype(int)]))
    print(f"\nmask{i}: path len={len(pts)}, h={h:.1f}, "
          f"cross pts={inside_o.sum()}")
    eps = 1.5
    nshow = 0
    for k in range(len(pts)):
        if not inside_o[k]:
            continue
        p, n = pts[k], nrm[k]
        for sign in (+1.0, -1.0):
            cx = p[0] + sign * n[0] * h
            cy = p[1] + sign * n[1] * h
            if not _in_mask(O, cx, cy):
                continue
            va = _sample_bilinear(imgf,
                                  p[0] + sign * n[0] * (h + eps),
                                  p[1] + sign * n[1] * (h + eps))
            vb = _sample_bilinear(imgf,
                                  p[0] + sign * n[0] * (h - eps),
                                  p[1] + sign * n[1] * (h - eps))
            if nshow < 12:
                print(f"  p=({p[0]:.0f},{p[1]:.0f}) n=({n[0]:.2f},{n[1]:.2f})"
                      f" sign={sign:+.0f} contour=({cx:.0f},{cy:.0f})"
                      f" edge={abs(va-vb):.1f} ({va:.0f} vs {vb:.0f})")
                nshow += 1
