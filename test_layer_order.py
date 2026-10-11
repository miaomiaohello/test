# -*- coding: utf-8 -*-
"""上下层判断鲁棒性测试: 多随机种子 x 4 用例, 统计判断正确率"""
import numpy as np
from demo_crossing_partition import build_cases, render, evaluate
from crossing_partition import crossing_partition
from layer_order import judge_layer_order

rng = np.random.default_rng(0)
size, cases = build_cases(rng)

SEEDS = list(range(16))
total, ok = 0, 0
edge_ok, jump_ok, jump_n = 0, 0, 0
auto_ok, auto_n, unc_n, unc_ok = 0, 0, 0, 0
worst = []

for name, cfg in cases.items():
    c_ok, c_n = 0, 0
    for seed in SEEDS:
        img, gts, fg, gt_inter = render(size, cfg["strips"],
                                        np.random.default_rng(seed))
        masks, _ = crossing_partition(fg, gt_inter, d=10,
                                      return_debug=True)
        if len(masks) != 2:
            continue
        _, pairs = evaluate(masks, gts)
        pred_of = {j: i for i, j, _ in pairs}
        if 1 not in pred_of:
            continue
        gt_top = pred_of[1]                  # gts[1] 后画 = 上层
        top_idx, info = judge_layer_order(img, masks)
        good = top_idx == gt_top
        # 单线索各自的准确率 (诊断用)
        s0, s1 = info["scores"]
        edge_ok += int((0 if s0 >= s1 else 1) == gt_top)
        j0, j1 = info["jumps"]
        if not (np.isnan(j0) or np.isnan(j1)):
            jump_ok += int((0 if j0 <= j1 else 1) == gt_top)
            jump_n += 1
        if info["uncertain"]:
            unc_n += 1
            unc_ok += int(good)
        else:
            auto_n += 1
            auto_ok += int(good)
        c_ok += int(good)
        c_n += 1
        total += 1
        ok += int(good)
        # 记录置信度最低的样本, 便于观察边界情况
        worst.append((info["confidence"], name, seed,
                      good, tuple(round(s, 3) for s in info["scores"]),
                      info["tiebreak"]))
    print(f"  {name:5s}: {c_ok}/{c_n}")

print(f"\nTOTAL: {ok}/{total} = {ok/total:.1%}")
print(f"单线索诊断: edge-only {edge_ok}/{total}, "
      f"jump-only {jump_ok}/{jump_n}")
print(f"弃权策略 (manual_thr=0.06): 自动判定 {auto_ok}/{auto_n}, "
      f"低置信转人工 {unc_n} 例 (其中建议正确 {unc_ok}/{unc_n})")
worst.sort()
print("\n置信度最低的5例 (分数差越小越危险):")
for conf, name, seed, good, scores, tb in worst[:5]:
    print(f"  {name} seed={seed} conf={conf:.3f} "
          f"scores={scores} tb={tb} [{'OK' if good else 'WRONG'}]")

wrongs = [w for w in worst if not w[3]]
print(f"\n全部判错例 ({len(wrongs)} 个):")
for conf, name, seed, good, scores, tb in wrongs:
    print(f"  {name} seed={seed} conf={conf:.3f} scores={scores} tb={tb}")
