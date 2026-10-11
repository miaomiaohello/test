# -*- coding: utf-8 -*-
"""
真实 G 显带数据集全流水线 (ChromSeg345张)
==========================================
输入约定 (inspect 已确认):
  image_*        白底黑染色体灰度(彩)图, 256x256, 单交叉簇 patch
  binary_label_* 交叉核心区 (= overlap_region, 最暗)
  binary_label2* 四条臂 (与核心区隔1-6px标注过渡带)

流程:
  Pass A (全部345张): Otsu前景 + GT交叉区 -> crossing_partition 统计
  Pass B (恰好2条的前6张): 上下层判断 -> 删下层 inpaint -> 展示面板
  Pass C (同6张): 经典检测交叉区 (detect_crossing_region) 端到端对照
"""
import glob
import os
import time
import cv2
import numpy as np

from crossing_partition import crossing_partition, detect_crossing_region
from layer_order import judge_layer_order
from demo_crossing_partition import evaluate   # 复用匈牙利匹配评估

D = os.path.join("data", "ChromSeg", "original_dataset")
OUT = "output"


# ----------------------------------------------------------------------
def load_case(path):
    n = os.path.basename(path)[6:-4]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = cv2.imread(os.path.join(D, f"binary_label_{n}.png"), 0) > 0
    r = cv2.imread(os.path.join(D, f"binary_label2_{n}.png"),
                   cv2.IMREAD_UNCHANGED)
    arms = (r[:, :, :3].astype(int).sum(2) > 0)
    return n, img, gray, v, arms


def extract_foreground(gray):
    """白底黑目标: Otsu 反向 + 闭运算 + 填孔 (亮带/过渡带) + 去小噪点。"""
    m = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    fg = m > 0
    # 消融结论(ablate_fg.py): close5 的"恰好2条"最多(159), fill 反而过合并
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    from scipy import ndimage
    # 去除 <20px 噪点
    lab, n = ndimage.label(fg)
    if n:
        keep = np.bincount(lab.ravel())[1:] >= 20
        fg = keep[lab - 1] & (lab > 0)
    return fg


# ======================================================================
# Pass A: 全部345张统计
# ======================================================================
def pass_a(paths):
    print("=" * 60)
    print("Pass A:345张, Otsu前景 + GT交叉区 -> crossing_partition")
    print("=" * 60)
    stat_n = []
    stat_cov = []
    stat_vcov = []
    fails = []
    t0 = time.time()
    for p in paths:
        n, img, gray, v, arms = load_case(p)
        fg = extract_foreground(gray)
        try:
            masks, _ = crossing_partition(fg, v, d=8, return_debug=True)
        except Exception as e:
            fails.append((n, repr(e)))
            continue
        stat_n.append(len(masks))
        if masks:
            u = np.logical_or.reduce(masks)
            stat_v = np.logical_or.reduce([v, np.zeros_like(v)])
            stat_cov.append((u & fg).sum() / max(fg.sum(), 1))
            # V 被覆盖的比例 (V 应属于输出的并集)
            stat_vcov.append((u & v).sum() / max(v.sum(), 1))
    dt = time.time() - t0
    stat_n = np.array(stat_n)
    print(f"  成功 {len(stat_n)}/345, 耗时 {dt:.1f}s "
          f"({1000*dt/max(len(paths),1):.0f} ms/张)")
    for k in sorted(set(stat_n.tolist())):
        print(f"  输出{k}条: {(stat_n == k).sum():3d} 张 "
              f"({100*(stat_n == k).mean():.1f}%)")
    print(f"  前景覆盖率(输出并集/fg): mean={np.mean(stat_cov):.3f} "
          f"min={np.min(stat_cov):.3f}")
    print(f"  交叉区覆盖率(输出并集含V): mean={np.mean(stat_vcov):.3f} "
          f"min={np.min(stat_vcov):.3f}")
    if fails:
        print(f"  异常 {len(fails)}: {fails[:3]}")
    return stat_n


# ======================================================================
# Pass B: 恰好2条的前6张 -> 上下层 -> 删底层 -> 面板
# ======================================================================
def pass_b(paths, want=6):
    print("\n" + "=" * 60)
    print("Pass B: 上下层判断 + 删下层演示 (GT交叉区, 恰好2条)")
    print("=" * 60)
    done = 0
    shown_ids = []
    skipped_frag = 0
    for p in paths:
        if done >= want:
            break
        n, img, gray, v, arms = load_case(p)
        fg = extract_foreground(gray)
        try:
            masks, _ = crossing_partition(fg, v, d=8, return_debug=True)
        except Exception:
            continue
        if len(masks) != 2:
            continue
        # 小掩码 <20% fg = 碎片残留(非第二条染色体), 跳过换下一张
        areas = sorted((int(m.sum()) for m in masks), reverse=True)
        if areas[1] < 0.2 * fg.sum():
            skipped_frag += 1
            continue
        top_idx, info = judge_layer_order(gray, masks, verbose=True)
        # 无有效交叉采样(骨架径不穿交叉区) -> 本线索不可用, 换下一张
        if not all(np.isfinite(s) for s in info["scores"]):
            skipped_frag += 1
            continue
        conf = info["confidence"]
        flag = " [需人工确认]" if info["uncertain"] else ""
        print(f"  image {n}: 判上层=mask{top_idx} "
              f"scores={tuple(round(s, 3) for s in info['scores'])}"
              f"{flag}")

        top, bot = masks[top_idx], masks[1 - top_idx]
        # 删下层: Telea inpaint, 背景白色
        canvas = gray.copy().astype(np.float32)
        canvas[bot] = 255                      # 先填白再细化 inpaint
        canvas = canvas.astype(np.uint8)
        inp = cv2.inpaint(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                          (bot * 255).astype(np.uint8), 5,
                          cv2.INPAINT_TELEA)
        inp[bot] = (inp[bot] * 0.5 + 255 * 0.5)  # 提亮残影

        # 面板: 原图 | fg+V | 分割 | 上下层 | 删底层
        h, w = gray.shape
        panel = np.zeros((h, w * 5, 3), np.uint8)
        panel[:, :w] = img
        p2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        p2[fg] = (p2[fg] * 0.5 + np.array([80, 200, 80]) * 0.5).astype(np.uint8)
        p2[v] = (p2[v] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
        panel[:, w:2*w] = p2
        from demo_crossing_partition import colorize_masks
        col = colorize_masks(gray.shape, masks)
        p3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        p3 = cv2.addWeighted(p3, 0.55, col, 0.45, 0)
        p3[v] = (p3[v] * 0.4 + np.array([255, 255, 0]) * 0.6).astype(np.uint8)
        panel[:, 2*w:3*w] = p3
        p4 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        p4[top] = (p4[top] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        p4[bot] = (p4[bot] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        cv2.putText(p4, f"top=m{top_idx}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        panel[:, 3*w:4*w] = p4
        panel[:, 4*w:] = inp
        for i, t in enumerate(["image", "fg(green)+V(red)", "partition",
                               "top(green)/bottom(blue)", "remove bottom"]):
            cv2.putText(panel, t, (i * w + 5, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1,
                        cv2.LINE_AA)
        scale = 1500 / panel.shape[1]
        panel = cv2.resize(panel, (1500, int(panel.shape[0] * scale)))
        out = os.path.join(OUT, f"real_{n}_panel.png")
        cv2.imwrite(out, panel)
        print(f"    saved {out}")
        shown_ids.append(n)
        done += 1
    print(f"  Pass B 完成 {done} 张 (跳过: 碎片/无效采样 {skipped_frag} 张)")
    return shown_ids


# ======================================================================
# Pass C: 经典检测交叉区 (无GT) 与 GT 的 IoU 对照
# ======================================================================
def pass_c(paths, ids):
    print("\n" + "=" * 60)
    print("Pass C: 经典检测交叉区 vs GT (端到端, 无GT输入)")
    print("=" * 60)
    idset = set(ids)
    for p in paths:
        n = os.path.basename(p)[6:-4]
        if n not in idset:
            continue
        _, img, gray, v, arms = load_case(p)
        fg = extract_foreground(gray)
        for rf in (0.5, 0.7, 1.0):
            det = detect_crossing_region(fg, r_factor=rf, min_r=2.0)
            inter = (det & v).sum()
            union = (det | v).sum()
            iou = inter / union if union else 0.0
            print(f"  image {n} r_factor={rf}: V_IoU={iou:.3f} "
                  f"det={det.sum()}px gt={v.sum()}px")


# ======================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(D, "image_*.png")),
                   key=lambda p: int(os.path.basename(p)[6:-4]))
    print(f"dataset: {len(paths)} images")
    pass_a(paths)
    shown = pass_b(paths, want=6)
    pass_c(paths, ids=shown)


if __name__ == "__main__":
    main()
