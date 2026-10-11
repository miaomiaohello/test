# -*- coding: utf-8 -*-
"""调查 L seed=0 判错: 两侧样本数、中位数、走道灰度均值"""
import numpy as np
from demo_crossing_partition import build_cases, render, evaluate
from crossing_partition import crossing_partition
from layer_order import (_skeleton_path, _resample, _in_mask,
                         _sample_bilinear)
from scipy import ndimage

rng = np.random.default_rng(0)
size, cases = build_cases(rng)
cfg = cases["L"]
print("L strips:", cfg["strips"])

img, gts, fg, gt_inter = render(size, cfg["strips"], np.random.default_rng(0))
masks, _ = crossing_partition(fg, gt_inter, d=10, return_debug=True)
_, pairs = evaluate(masks, gts)
pred_of = {j: i for i, j, _ in pairs}
print("pairs:", pairs, "-> gt_top pred idx =", pred_of[1])

imgf = img.astype(np.float64)
eps, H, W = 1.5, img.shape[0], img.shape[1]

for i in range(2):
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
    tan = np.c_[-nrm[:, 1], nrm[:, 0]]
    ang = np.arctan2(tan[:, 1], tan[:, 0]) % np.pi
    ref = 0.5 * np.arctan2(np.mean(np.sin(2 * ang[arm_sel])),
                           np.mean(np.cos(2 * ang[arm_sel])))
    dev = np.abs(ang - ref)
    dev = np.minimum(dev, np.pi - dev)
    aligned = dev <= np.deg2rad(30.0)

    # 走道灰度: 交叉区中心线 vs 臂区中心线 (±3px 法向平均)
    arm_gray, cross_gray = [], []
    side_vals = {1: [], -1: []}
    side_pos = {1: [], -1: []}
    for k in range(len(pts)):
        p, n = pts[k], nrm[k]
        # 走道均值
        g = np.mean([_sample_bilinear(imgf, p[0] + n[0] * d, p[1] + n[1] * d)
                     for d in (-3, -1, 0, 1, 3)])
        (cross_gray if inside_o[k] else arm_gray).append(g)
        for sign in (+1.0, -1.0):
            cx = p[0] + sign * n[0] * h
            cy = p[1] + sign * n[1] * h
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if not inside_o[k]:
                continue
            if aligned[k] and _in_mask(O, cx, cy):
                va = _sample_bilinear(imgf, p[0] + sign*n[0]*(h+eps),
                                      p[1] + sign*n[1]*(h+eps))
                vb = _sample_bilinear(imgf, p[0] + sign*n[0]*(h-eps),
                                      p[1] + sign*n[1]*(h-eps))
                side_vals[int(sign)].append(abs(va - vb))
                side_pos[int(sign)].append((cx, cy))

    print(f"\nmask{i} {'TOP(gt)' if i == pred_of[1] else 'bottom'}: "
          f"h={h:.1f}, cross pts={inside_o.sum()}, aligned∩cross="
          f"{(inside_o & aligned).sum()}")
    for s in (1, -1):
        v = side_vals[s]
        if v:
            print(f"  side {s:+d}: n={len(v)}, med={np.median(v):.1f}, "
                  f"vals={np.round(sorted(v), 1).tolist()[:10]}")
        else:
            print(f"  side {s:+d}: n=0")
    print(f"  corridor mean: arm={np.mean(arm_gray):.1f}, "
          f"cross={np.mean(cross_gray):.1f}")
