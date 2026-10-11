# -*- coding: utf-8 -*-
"""X90 seed=12 判错调查: 两侧中位数/跳变/剖面"""
import numpy as np
from demo_crossing_partition import build_cases, render, evaluate
from crossing_partition import crossing_partition
from layer_order import judge_layer_order

rng = np.random.default_rng(0)
size, cases = build_cases(rng)
cfg = cases["X90"]
img, gts, fg, gt_inter = render(size, cfg["strips"],
                                np.random.default_rng(12))
masks, _ = crossing_partition(fg, gt_inter, d=10, return_debug=True)
_, pairs = evaluate(masks, gts)
pred_of = {j: i for i, j, _ in pairs}
gt_top = pred_of[1]
print("gt_top =", gt_top)

top_idx, info = judge_layer_order(img, masks, verbose=True,
                                  conf_thr=-1)   # 关掉裁决, 看两线索
print("jumps:", info["jumps"], "tie:", info["tiebreak"])
for i, d in enumerate(info["details"]):
    tag = "TOP(gt)" if i == gt_top else "bottom"
    print(f"mask{i} {tag}: sides={d.get('side_med')} "
          f"cross={d['cross_med']:.1f} arm={d['arm_med']:.1f} "
          f"n={d['n_cross']} jump={d.get('jump'):.1f}")
