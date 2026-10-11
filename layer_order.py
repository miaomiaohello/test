# -*- coding: utf-8 -*-
"""
上下层判断 (遮挡顺序) —— ChromSeg 路线的第 3 步
==================================================
两篇论文都只做"分割出两条染色体", 不判断谁压着谁。
本模块用"轮廓连续性"线索解决:

    上层染色体的侧轮廓 *完整穿过* 交叉区 (它压在别人身上, 边缘一直可见);
    下层染色体的侧轮廓在交叉区内 *被上层的纹理切断* (预期轮廓处图像连续)。

实现:
    1) 骨架取直径路径 -> 重采样, 算切线/法线
    2) 臂区(不在对方掩码内的路径点): 校准自身轮廓强度 arm_strength
       (= 自身纹理 vs 背景 的台阶差, 除以该染色体自身的对比度基准)
    3) 交叉区(在对方掩码内的路径点): 在预期轮廓位置
       (中心线 ± 半宽 h) 测法向台阶差; *仅当预期轮廓点也落在对方掩码内*
       (即真会被遮挡) 才计入 —— 否则该样本无判别意义 (如 T 形中
       底层染色体远端轮廓并未被盖住)
    4) score = cross_median / arm_median; 分高者为上层
    5) 两线索打平 (conf < conf_thr) 时改由剖面跳变裁决:
       出入交叉边界时灰度水平跳变小者为上层 (纹理连续穿过边界)。

    中位数聚合: 防止偶发的条带边缘 (天然纹理) 拉高数值。

    三处稳健性修正 (去掉后 T/L 用例判错):
      A. 交叉区采样点须与臂区主方向同向 (切线偏差 <= 30°) ——
         骨架在 V 翼附近拐弯, 拐弯点法线不指向自身侧轮廓;
      B. 两侧分别取中位数后取 *最小值*, 缺失侧记 0 ——
         真上层的轮廓两侧都可见, 单侧强边缘多半是对方条带
         边界恰好撞上预期轮廓线的假象;
      C. 单点测量在 [h-4, h+4] 内取最强窄差 —— 斜向条带栅格化
         后轮廓位置沿路径参差, 固定偏移会错过台阶 (L 用例教训)。
"""

import numpy as np
from collections import deque
from scipy import ndimage

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def _sample_bilinear(img, x, y):
    """双线性采样, x/y 可为标量或数组; 越界裁到边界。"""
    H, W = img.shape
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = x - x0
    fy = y - y0
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
         img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return v


def _in_mask(mask, x, y):
    """点 (x, y) 是否在掩码内 (最近邻), 支持标量/数组; 越界返回 False。"""
    H, W = mask.shape
    xi = np.asarray(np.round(x), dtype=int)
    yi = np.asarray(np.round(y), dtype=int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if xi.ndim == 0:
        return bool(ok) and bool(mask[int(yi), int(xi)])
    out = np.zeros(xi.shape, bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def _skeleton_path(mask):
    """
    骨架图上求直径路径 (BFS 两遍):
      任一端点 -> 最远点 a -> a 最远点 b -> a..b 即最长路径。
    返回 (N, 2) float 数组, 列为 (x, y); 失败返回 None。
    """
    if _HAS_SKIMAGE:
        skel = skeletonize(mask)
    else:
        return None
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if len(pts) < 3:
        return None

    def nbrs(p):
        x, y = p
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    q = (x + dx, y + dy)
                    if q in pts:
                        yield q

    deg = {p: sum(1 for _ in nbrs(p)) for p in pts}
    ends = [p for p, d in deg.items() if d == 1]
    if not ends:
        ends = [next(iter(pts))]

    def bfs(src):
        prev = {src: None}
        order = [src]
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for v in nbrs(u):
                if v not in prev:
                    prev[v] = u
                    dq.append(v)
                    order.append(v)
        return order[-1], prev

    a, _ = bfs(ends[0])
    b, prev = bfs(a)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return np.asarray(path, dtype=float)          # (N, 2): x, y


def _resample(path, step=1.0, smooth=5):
    """按弧长重采样, 平滑后算单位法线 (切线旋转90°)。"""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    t = np.arange(0.0, cum[-1] + 1e-9, step)
    px = np.interp(t, cum, path[:, 0])
    py = np.interp(t, cum, path[:, 1])

    def _smooth(v):
        if smooth <= 1 or len(v) < smooth:
            return v
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        vp = np.r_[np.full(pad, v[0]), v, np.full(pad, v[-1])]
        return np.convolve(vp, k, mode="valid")

    tx = _smooth(np.gradient(px))
    ty = _smooth(np.gradient(py))
    nrm = np.hypot(tx, ty)
    nrm[nrm < 1e-9] = 1.0
    # 法线 = 切线顺时针转90°:  n = (ty, -tx)
    nx = ty / nrm
    ny = -tx / nrm
    return np.c_[px, py], np.c_[nx, ny]


# ----------------------------------------------------------------------
# 主判断
# ----------------------------------------------------------------------
def judge_layer_order(img, masks, eps=1.5, conf_thr=0.01,
                      manual_thr=0.06, verbose=False):
    """
    判断交叉的两条染色体谁在上层。

    参数
    ----------
    img        : (H, W) uint8/float 灰度图
    masks      : [maskA, maskB] 两个 bool 掩码 (crossing_partition 的输出)
    eps        : 轮廓台阶差的窄差采样间隔 (px)
    conf_thr   : 轮廓线索置信度阈值; 低于它改由剖面跳变裁决
    manual_thr : 人工确认阈值; conf 低于它时 info["uncertain"]=True,
                 决策仅供参考, 建议人工指定上层 (兜底方案)

    返回
    ----------
    (top_idx, info)
      top_idx : 上层染色体在 masks 中的下标 (0 或 1)
      info    : {"scores", "jumps", "tiebreak", "uncertain",
                 "details", "confidence"}
    """
    img = np.asarray(img, dtype=np.float64)
    assert len(masks) == 2, "本模块当前只处理一对交叉染色体"

    details = []
    for i in range(2):
        X, O = masks[i], masks[1 - i]
        d = {"arm_med": np.nan, "cross_med": np.nan,
             "n_arm": 0, "n_cross": 0, "h": np.nan}

        path = _skeleton_path(X)
        if path is None:
            details.append(d)
            continue
        pts, nrm = _resample(path)
        dt = ndimage.distance_transform_edt(X)
        H, W = img.shape
        inside_o = _in_mask(O, pts[:, 0], pts[:, 1])
        on_x = (pts[:, 0] >= 0) & (pts[:, 0] < W) & \
               (pts[:, 1] >= 0) & (pts[:, 1] < H)

        # 中心线剖面 (±2px 法向平均, 抑噪) —— 供剖面跳变裁决用
        prof = np.array([
            np.mean([_sample_bilinear(img, pts[k, 0] + nrm[k, 0] * s,
                                      pts[k, 1] + nrm[k, 1] * s)
                     for s in (-2, -1, 0, 1, 2)])
            for k in range(len(pts))])
        d["_prof"], d["_ins"] = prof, inside_o

        # ---- 半宽 h: 臂区路径点的 dt 中位数 (避开共享 V 区的膨胀) ----
        arm_sel = (~inside_o) & on_x
        if arm_sel.sum() < 3:
            details.append(d)
            continue
        h = float(np.median(dt[pts[arm_sel, 1].astype(int),
                               pts[arm_sel, 0].astype(int)]))
        d["h"] = h

        # ---- 修正A: 切线对齐门控 ----
        # 骨架在 V 翼附近会拐弯, 拐弯点的法线不再指向自身侧轮廓,
        # 测量无意义 -> 交叉区采样点须与臂区主方向一致 (偏差 <= 30°)。
        tan = np.c_[-nrm[:, 1], nrm[:, 0]]          # 切线 (与法线垂直)
        ang = np.arctan2(tan[:, 1], tan[:, 0]) % np.pi
        ref = 0.5 * np.arctan2(np.mean(np.sin(2 * ang[arm_sel])),
                               np.mean(np.cos(2 * ang[arm_sel])))
        dev = np.abs(ang - ref)
        dev = np.minimum(dev, np.pi - dev)
        aligned = dev <= np.deg2rad(30.0)

        # ---- 逐点、逐侧采样 ----
        # 修正C: 45°等斜边栅格化后轮廓位置沿路径参差 (实际边缘在
        # h±2 抖动), 固定在 h±eps 采样会错过台阶只测到小值。
        # 在 [h-4, h+4] 内取最强窄差, 自适应抖动; 交叉区的整线
        # "条带边界撞车"已由两侧取 min (修正B) 挡住, 此处安全。
        ts = np.arange(max(h - 4.0, eps + 0.5), h + 4.0 + 1e-9, 1.0)

        def edge_at(p, n, sign):
            """sign*n 方向上, 预期轮廓附近的最强法向台阶差 + 轮廓点坐标"""
            cx = p[0] + sign * n[0] * h
            cy = p[1] + sign * n[1] * h
            best = 0.0
            for t in ts:
                va = _sample_bilinear(img,
                                      p[0] + sign * n[0] * (t + eps),
                                      p[1] + sign * n[1] * (t + eps))
                vb = _sample_bilinear(img,
                                      p[0] + sign * n[0] * (t - eps),
                                      p[1] + sign * n[1] * (t - eps))
                d = abs(va - vb)
                if d > best:
                    best = d
            return best, cx, cy

        arm_vals = []
        cross_by_side = {1: [], -1: []}
        for k in range(len(pts)):
            p, n = pts[k], nrm[k]
            for sign in (+1.0, -1.0):
                e, cx, cy = edge_at(p, n, sign)
                if not (0 <= cx < W and 0 <= cy < H):
                    continue
                if not inside_o[k]:
                    arm_vals.append(e)          # 臂区: 校准基准
                else:
                    # 交叉区: 仅当预期轮廓点也被对方覆盖 (真会遮挡)
                    # 且路径点切线与臂区主方向一致 (修正A) 才有效
                    if aligned[k] and _in_mask(O, cx, cy):
                        cross_by_side[int(sign)].append(e)

        cross_vals = cross_by_side[1] + cross_by_side[-1]
        d["n_arm"] = len(arm_vals)
        d["n_cross"] = len(cross_vals)
        d["arm_med"] = float(np.median(arm_vals)) if arm_vals else np.nan
        # ---- 修正B: 两侧分别取中位数, 再取两侧最小值 ----
        # 真上层的轮廓在交叉区两侧都可见; 下层若只有单侧测到强边缘,
        # 往往是对方条带边界恰好撞上预期轮廓线的假象 (T 用例教训),
        # 因此缺失的一侧记 0, 不允许单侧证据定"上层"。
        side_med = [float(np.median(cross_by_side[s]))
                    if cross_by_side[s] else 0.0 for s in (1, -1)]
        d["side_med"] = tuple(side_med)
        d["cross_med"] = min(side_med) if cross_vals else np.nan
        if arm_vals and cross_vals and d["arm_med"] > 1e-6:
            d["score"] = d["cross_med"] / d["arm_med"]
        else:
            d["score"] = -np.inf                # 无有效样本 -> 不可能是上层
        details.append(d)

    # ---- 剖面跳变线索: 进出交叉边界时的灰度水平跳变 ----
    # 上层纹理穿过对方边界是连续的 -> 跳变小; 下层进入对方纹理 -> 跳变大。
    # 独立于轮廓线索, 用于两者打平 (conf < conf_thr) 时的裁决。
    for d in details:
        prof, ins = d.get("_prof"), d.get("_ins")
        jumps = []
        if prof is not None and ins is not None and len(ins) > 20:
            diff = np.diff(ins.astype(np.int8))
            for k in np.nonzero(diff)[0]:
                a = k + 1
                if diff[k] == 1:               # 进入对方: 内=之后, 外=之前
                    win_in = prof[a:a + 10]
                    win_out = prof[max(0, a - 10):a]
                else:                          # 离开对方
                    win_in = prof[max(0, a - 10):a]
                    win_out = prof[a:a + 10]
                if len(win_in) >= 6 and len(win_out) >= 6:
                    jumps.append(abs(float(np.mean(win_in)) -
                                     float(np.mean(win_out))))
        d["jump"] = float(np.mean(jumps)) if jumps else np.nan

    s0 = details[0].get("score", -np.inf)
    s1 = details[1].get("score", -np.inf)
    conf = float(abs(s0 - s1))
    top_idx = 0 if s0 >= s1 else 1
    tiebreak = "edge"
    j0, j1 = details[0].get("jump", np.nan), details[1].get("jump", np.nan)
    if conf < conf_thr and not (np.isnan(j0) or np.isnan(j1)):
        top_idx = 0 if j0 <= j1 else 1           # 跳变小者 = 上层
        tiebreak = "profile-jump"
    info = {"scores": (s0, s1), "details": details,
            "jumps": (j0, j1), "tiebreak": tiebreak,
            "confidence": conf,
            "uncertain": bool(conf < manual_thr)}
    if verbose:
        for i, dd in enumerate(details):
            tag = "top " if i == top_idx else "bottom"
            jp = dd.get("jump", float("nan"))
            print(f"    mask{i} [{tag}] score={dd.get('score', -np.inf):.3f} "
                  f"(cross={dd['cross_med']:.1f} sides="
                  f"{tuple(round(s, 1) for s in dd.get('side_med', ()))} "
                  f"/ arm={dd['arm_med']:.1f}, "
                  f"n={dd['n_cross']}/{dd['n_arm']}, h={dd['h']:.1f}, "
                  f"jump={jp:.1f})")
        if tiebreak == "profile-jump":
            print(f"    ** 低置信度 (conf={conf:.3f} < {conf_thr}), "
                  f"由剖面跳变裁决 (jump={j0:.1f} vs {j1:.1f}) **")
        if info["uncertain"]:
            print(f"    !! conf={conf:.3f} < {manual_thr}: "
                  f"建议人工确认上层 (决策仅供参考) !!")
    return top_idx, info
