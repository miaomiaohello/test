# -*- coding: utf-8 -*-
"""L seed=0: mask0 side-1 近零样本的横向剖面实况"""
import numpy as np
from demo_crossing_partition import build_cases, render, evaluate
from crossing_partition import crossing_partition
from layer_order import (_skeleton_path, _resample, _in_mask,
                         _sample_bilinear)
from scipy import ndimage

rng = np.random.default_rng(0)
size, cases = build_cases(rng)
cfg = cases["L"]
img, gts, fg, gt_inter = render(size, cfg["strips"], np.random.default_rng(0))
masks, _ = crossing_partition(fg, gt_inter, d=10, return_debug=True)
_, pairs = evaluate(masks, gts)
pred_of = {j: i for i, j, _ in pairs}
imgf = img.astype(np.float64)
H, W = img.shape

i = pred_of[1]                      # 真上层
X, O = masks[i], masks[1 - i]
path = _skeleton_path(X)
pts, nrm = _resample(path)
dt = ndimage.distance_transform_edt(X)
inside_o = _in_mask(O, pts[:, 0], pts[:, 1])
on_x = ((pts[:, 0] >= 0) & (pts[:, 0] < W) &
        (pts[:, 1] >= 0) & (pts[:, 1] < H))
arm_sel = (~inside_o) & on_x
h = float(np.median(dt[pts[arm_sel, 1].astype(int),
                       pts[arm_sel, 0].astype(int)]))
print(f"mask{i}, h={h:.1f}")

# 找 side=-1 的有效样本, 打印剖面 t ∈ [-4, 4]*2 相对 h
shown = 0
zeros = 0
for k in range(len(pts)):
    if not inside_o[k]:
        continue
    p, n = pts[k], nrm[k]
    sign = -1.0
    cx, cy = p[0] + sign*n[0]*h, p[1] + sign*n[1]*h
    if not _in_mask(O, cx, cy):
        continue
    prof = []
    for t in range(-5, 6):
        tt = h + sign * t * 1.0     # 沿 sign*n 方向偏离 h
        # 以 sign 方向的偏移: 距中心线距离 d = h + t (t 负=内侧)
        d = h - t * 1.0
        v = _sample_bilinear(imgf, p[0] + sign*n[0]*d,
                             p[1] + sign*n[1]*d)
        prof.append(v)
    e = abs(prof[5] - prof[7]) if False else None
    # 打印 10 个样本 (间隔)
    if shown < 10:
        print(f"p=({p[0]:.0f},{p[1]:.0f}) contour=({cx:.0f},{cy:.0f}) "
              f"prof(d=h-5..h+5)={np.round(prof, 0).astype(int).tolist()}")
        shown += 1
