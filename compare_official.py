# -*- coding: utf-8 -*-
"""
官方 ChromSeg crossing-partition vs 本项目实现 正面对照
=======================================================
同一批合成用例 (X90/X60/T/L)、同一 GT 交叉区输入,
分别跑第三方官方代码 third_party/ChromSeg/crossing-partition/utils.py
的 crossing_reconstruct 与本项目的 crossing_partition,
用 demo 的 evaluate() (匈牙利匹配 mean IoU) 评估。
额外记录输出掩码数量 (官方可能把两条粘成一条 -> 数量不对即失败)。
"""
import os
import sys
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "third_party", "ChromSeg",
                                "crossing-partition"))
from utils import crossing_reconstruct           # 官方实现 (第三方)

from demo_crossing_partition import build_cases, render, evaluate
from crossing_partition import crossing_partition

SEEDS = [0, 1, 2]


def official_run(img, fg, gt_inter):
    """按官方 main.py 的输入约定喂数据, 返回掩码 list。"""
    bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlapped = gt_inter.astype(np.uint8)                 # 0/1
    non_ov = (fg & ~gt_inter).astype(np.uint8)             # 0/1
    masks = crossing_reconstruct(bgr, overlapped, non_ov)
    return [np.asarray(m).astype(bool) for m in masks]


def main():
    rng = np.random.default_rng(0)
    size, cases = build_cases(rng)

    rows = []
    for name, cfg in cases.items():
        for seed in SEEDS:
            img, gts, fg, gt_inter = render(size, cfg["strips"],
                                            np.random.default_rng(seed))
            # ---- 官方 ----
            try:
                off = official_run(img, fg, gt_inter)
                o_miou, _ = evaluate(off, gts)
                o_n = len(off)
            except Exception as e:
                o_miou, o_n = float("nan"), -1
                print(f"  [official failed] {name} seed={seed}: {e!r}")
            # ---- 我们 ----
            try:
                ours, _ = crossing_partition(fg, gt_inter, d=10,
                                             return_debug=True)
                m_miou, _ = evaluate(ours, gts)
                m_n = len(ours)
            except Exception as e:
                m_miou, m_n = float("nan"), -1
                print(f"  [ours failed] {name} seed={seed}: {e!r}")

            o_flag = "PASS" if (o_n == len(gts) and o_miou >= 0.85) else \
                     ("WEAK" if o_miou >= 0.6 else "FAIL")
            m_flag = "PASS" if (m_n == len(gts) and m_miou >= 0.85) else \
                     ("WEAK" if m_miou >= 0.6 else "FAIL")
            rows.append((name, seed, o_n, o_miou, o_flag,
                         m_n, m_miou, m_flag))

    print(f"\n{'case':6s} {'seed':>4s} | {'官#':>3s} {'官IoU':>7s} {'':4s} "
          f"| {'我#':>3s} {'我IoU':>7s} {'':4s}")
    print("-" * 58)
    for name, seed, o_n, o_miou, o_f, m_n, m_miou, m_f in rows:
        print(f"{name:6s} {seed:4d} | {o_n:3d} {o_miou:7.3f} {o_f:4s} "
              f"| {m_n:3d} {m_miou:7.3f} {m_f:4s}")

    # ---- 汇总 ----
    o_ok = sum(1 for r in rows if r[4] == "PASS")
    m_ok = sum(1 for r in rows if r[7] == "PASS")
    o_i = [r[3] for r in rows if not np.isnan(r[3])]
    m_i = [r[6] for r in rows if not np.isnan(r[6])]
    print("-" * 58)
    print(f"官方: PASS {o_ok}/{len(rows)}, 平均IoU {np.mean(o_i):.3f}")
    print(f"我方: PASS {m_ok}/{len(rows)}, 平均IoU {np.mean(m_i):.3f}")


if __name__ == "__main__":
    main()
