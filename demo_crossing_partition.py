# -*- coding: utf-8 -*-
"""
crossing-partition 算法验证 demo
================================
合成 G显带风格的交叉染色体图 (X90 / X60 / T / L 四种形态),
跑 crossing_partition 流水线, 与真值 (GT) 计算 IoU 评估,
并演示最终任务: 删掉下层染色体、保留上层。

输出保存到 ./output/
"""

import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from crossing_partition import crossing_partition, detect_crossing_region
from layer_order import judge_layer_order

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)


# ---------------- 通用: 非 ASCII 路径安全的图片保存 ----------------
def save_png(name, bgr):
    ok, buf = cv2.imencode(".png", np.asarray(bgr))
    if not ok:
        raise RuntimeError("encode fail: " + name)
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(buf.tobytes())
    print("  saved:", name)


def to_bgr(gray):
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


# ---------------- 合成染色体 ----------------
def draw_strip(shape, center, angle_deg, length, width):
    """旋转矩形条带掩码 (中心, 角度, 长, 宽)"""
    rect = cv2.boxPoints(((center[0], center[1]), (length, width), angle_deg))
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(rect)], 1)
    return m.astype(bool)


def paint_gband(img, mask, angle_deg, rng):
    """在掩码内画沿长轴方向的明暗 G 显带纹理"""
    ys, xs = np.nonzero(mask)
    a = np.deg2rad(angle_deg)
    t = xs * np.cos(a) + ys * np.sin(a)
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    nctrl = int(rng.integers(6, 11))
    ctrl = np.sort(rng.uniform(0, 1, nctrl))
    vals = rng.uniform(75, 170, nctrl)                     # 暗/亮带灰度
    profile = np.interp(t, np.r_[0.0, ctrl, 1.0],
                        np.r_[vals[0], vals, vals[-1]])
    noise = rng.normal(0, 4, len(t))
    img[ys, xs] = np.clip(profile + noise, 0, 255).astype(np.uint8)


def make_canvas(rng, size=400):
    bg = np.full((size, size), 215.0)
    bg += rng.normal(0, 3, bg.shape)                       # 轻噪声
    return np.clip(bg, 0, 255).astype(np.uint8)


# ---------------- 评估 ----------------
def evaluate(preds, gts):
    """匈牙利匹配后逐条算 IoU, 返回 (mean_iou, per_pair_list)"""
    if not preds or not gts:
        return 0.0, []
    iou = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            iou[i, j] = inter / union if union else 0.0
    ri, ci = linear_sum_assignment(-iou)
    pairs = [(int(i), int(j), float(iou[i, j])) for i, j in zip(ri, ci)]
    mean = float(np.mean([p[2] for p in pairs]))
    return mean, pairs


# ---------------- 染色体着色可视化 ----------------
def colorize_masks(shape, masks):
    out = np.zeros((shape[0], shape[1], 3), np.uint8)
    rng = np.random.default_rng(7)
    palette = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in masks]
    for m, c in zip(masks, palette):
        out[m] = (out[m] * 0.35 + np.array(c) * 0.65).astype(np.uint8)
    return out


def label_view(V_labels, S_labels):
    """块可视化: V 块红色系, S 块绿色系"""
    out = np.full((*V_labels.shape, 3), 30, np.uint8)
    nV = V_labels.max()
    nS = S_labels.max()
    for i in range(1, nV + 1):
        out[V_labels == i] = (0, 0, 200)
    for i in range(1, nS + 1):
        t = i / max(nS, 1)
        out[S_labels == i] = (int(60 + 120 * t), int(180 - 90 * t), 60)
    return out


def panel(images, titles, scale=0.55):
    """横向拼接面板, 带标题"""
    tiles = []
    for img, title in zip(images, titles):
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        tile = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        if tile.ndim == 2:
            tile = to_bgr(tile)
        canvas = np.full((nh + 26, nw, 3), 245, np.uint8)
        canvas[26:] = tile
        cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(canvas)
    hmax = max(t.shape[0] for t in tiles)
    gap = np.full((hmax, 3, 3), 245, np.uint8)
    row = []
    for i, t in enumerate(tiles):
        if t.shape[0] < hmax:
            pad = np.full((hmax - t.shape[0], t.shape[1], 3), 245, np.uint8)
            t = np.vstack([t, pad])
        if i:
            row.append(gap)
        row.append(t)
    return np.hstack(row)


# ---------------- 构造四种交叉形态 ----------------
def build_cases(rng):
    size = 400
    c = size // 2
    L, W = 300, 26

    cases = {}

    # 1) X90: 横竖十字 (用户的目标场景)
    cases["X90"] = dict(
        strips=[((c, c), 0, L, W), ((c, c), 90, L, W)])

    # 2) X60: 斜交
    cases["X60"] = dict(
        strips=[((c, c), 30, L, W), ((c, c), -30, L, W)])

    # 3) T: 竖条终止在横条中心线处
    cases["T"] = dict(
        strips=[((c, c), 0, L, W), ((c, c - L // 4), 90, L // 2, W)])

    # 4) L: 45° 斜条与水平条交叉 (臂方向不再两两对称)
    cases["L"] = dict(
        strips=[((c, c), 0, L, W),
                ((c - 45, c + 40), 45, 230, W)])

    return size, cases


def render(size, strips, rng):
    """按 strips 顺序画染色体: 后画的盖在上层 (形成天然的上下层关系)"""
    img = make_canvas(rng, size)
    gts = []
    for (center, angle, ln, wd) in strips:
        m = draw_strip((size, size), center, angle, ln, wd)
        paint_gband(img, m, angle, rng)
        gts.append(m)
    # 真值交叉区 = 两两掩码交集 (任意一对)
    inter = np.zeros((size, size), bool)
    for i in range(len(gts)):
        for j in range(i + 1, len(gts)):
            inter |= gts[i] & gts[j]
    fg = np.logical_or.reduce(gts) if gts else np.zeros((size, size), bool)
    return img, gts, fg, inter


# ---------------- 主流程 ----------------
def main():
    rng = np.random.default_rng(42)
    size, cases = build_cases(rng)

    summary = []
    for name, cfg in cases.items():
        print(f"[{name}]")
        img, gts, fg, gt_inter = render(size, cfg["strips"], rng)

        # ---- 两个输入来源: (a) GT交叉区 (b) 经典检测 ----
        for src, ov in [("GT区域", gt_inter),
                        ("经典检测", None)]:
            if ov is not None:
                masks, dbg = crossing_partition(
                    fg, ov, d=10, return_debug=True)
            else:
                masks, dbg = crossing_partition(
                    fg, None, d=10, detect_kw=dict(r_factor=1.8),
                    return_debug=True)
            miou, pairs = evaluate(masks, gts)
            ok = "PASS" if miou >= 0.85 else ("WEAK" if miou >= 0.6 else "FAIL")
            print(f"  {src:6s} -> {len(masks)} 条, mean IoU = {miou:.3f}  [{ok}]")
            summary.append((name, src, miou, ok))

            # 可视化
            pred_view = colorize_masks(img.shape, masks)
            gt_view = colorize_masks(img.shape, gts)
            blend = to_bgr(img.copy())
            blend = cv2.addWeighted(blend, 0.55, pred_view, 0.45, 0)

            ov_img = to_bgr(img.copy())
            if ov is None:
                ov = dbg["overlap"]
            ov_img[ov] = (ov_img[ov] * 0.4 + np.array([0, 0, 255]) * 0.6)

            p = panel(
                [img, ov_img, label_view(dbg["V_labels"], dbg["S_labels"]),
                 gt_view, blend],
                ["input", f"overlap ({src})", "blocks V(red)/S(green)",
                 "GT", f"pred meanIoU={miou:.2f}"], scale=0.5)
            save_png(f"{name}_{src}.png", p)

        # ---- 真实完整流水线额外存一张大图 ----
    print("\n==== SUMMARY ====")
    for name, src, miou, ok in summary:
        print(f"  {name:5s} {src:6s} IoU={miou:.3f} {ok}")

    # ---------------- 上下层判断测试 (第3步, 论文未覆盖) ----------------
    print("\n==== LAYER ORDER (轮廓连续性) ====")
    judge_ok = 0
    judge_n = 0
    for name, cfg in cases.items():
        img, gts, fg, gt_inter = render(size, cfg["strips"],
                                        np.random.default_rng(42))
        masks, _ = crossing_partition(fg, gt_inter, d=10,
                                      return_debug=True)
        if len(masks) != 2:
            print(f"  {name}: skip (分割得到 {len(masks)} 条, 需 2 条)")
            continue
        _, pairs = evaluate(masks, gts)
        pred_of = {j: i for i, j, _ in pairs}   # gt 下标 -> pred 下标
        gt_top = pred_of[1]                     # gts[1] 后画 = 上层
        top_idx, info = judge_layer_order(img, masks, verbose=True)
        ok = top_idx == gt_top
        judge_ok += int(ok)
        judge_n += 1
        flag = " [需人工确认]" if info["uncertain"] else ""
        print(f"  {name}: 判上层=mask{top_idx}, 真值=mask{gt_top} "
              f"[{'OK' if ok else 'WRONG'}] "
              f"scores={tuple(round(s, 3) for s in info['scores'])}"
              f"{flag}")
    print(f"  -> 上层判断正确率: {judge_ok}/{judge_n}")

    # ---------------- 最终任务演示: 删下层, 留上层 ----------------
    print("\n[remove-bottom demo] (X90, GT交叉区)")
    img, gts, fg, gt_inter = render(size, cases["X90"]["strips"],
                                    np.random.default_rng(42))
    masks, dbg = crossing_partition(fg, gt_inter, d=10, return_debug=True)
    assert len(masks) == 2, f"expect 2 chromosomes, got {len(masks)}"
    # render 顺序: 后画的在上层 -> 先做上下层判断(不再用真值)
    miou, pairs = evaluate(masks, gts)
    pred_of = {j: i for i, j, _ in pairs}
    top_idx, linfo = judge_layer_order(img, masks, verbose=True)
    gt_top_idx = pred_of[1]                       # 真值: gts[1] 后画=上层
    flag = " [低置信,建议人工确认]" if linfo["uncertain"] else ""
    print(f"  上下层判断: 预测上层=mask{top_idx}, "
          f"真值上层=mask{gt_top_idx} "
          f"[{'OK' if top_idx == gt_top_idx else 'WRONG'}]{flag}")
    top_pred = masks[top_idx]
    bottom_pred = masks[1 - top_idx]

    # 底层唯一可见像素 = 底层掩码 - 顶层掩码 (交叉区显示的是上层纹理)
    remove = (bottom_pred & ~top_pred).astype(np.uint8)
    # 形态学微扩, 避免残留边缘
    remove = cv2.dilate(remove, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)))
    repaired = cv2.inpaint(img, remove * 255, inpaintRadius=6,
                           flags=cv2.INPAINT_TELEA)

    mask_vis = to_bgr(img.copy())
    mask_vis[remove.astype(bool)] = (215, 215, 215)

    row = panel(
        [img, colorize_masks(img.shape, masks), mask_vis, repaired],
        ["input (bottom=horizontal, top=vertical)",
         "two instances", "bottom removed (mask)", "repaired (inpaint)"],
        scale=0.55)
    save_png("X90_remove_bottom.png", row)


if __name__ == "__main__":
    main()
