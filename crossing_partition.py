# -*- coding: utf-8 -*-
"""
ChromSeg · crossing-partition 算法 Python 实现
==============================================
论文: Cao et al., "ChromSeg: Two-Stage Framework for Overlapping
      Chromosome Segmentation and Reconstruction" (Algorithm 1)

在 ChromSeg 流水线中的位置:
    阶段一 (论文: region-guided UNet++)  ──► overlap_mask (交叉重叠区域)
    阶段二 crossing-partition ◄── 本文件
        1) split_blocks        前景切成 重叠块V / 非重叠块S
        2) build_dictionary    圆盘范围 d 内 V-S 邻接字典
        3) classify            射线配对 + 并查集 (核心启发式)
        4) assemble            森林树 -> 单条染色体

阶段一缺失时提供经典 CV 替代 detect_crossing_region():
    骨架化找度数>=3的分叉点(十字中心), 以距离变换值为半径画圆盘
    近似交叉区域 —— 适用于无训练数据的场景。

坐标约定: 图像数组索引为 [y, x]; 中心点存 (x, y) 即 (列, 行)。
"""

import numpy as np
import cv2
from collections import defaultdict
from scipy import ndimage

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:          # 缺 skimage 时退化为形态学骨架
    _HAS_SKIMAGE = False


# ----------------------------------------------------------------------
# 并查集 (Union-Find): 论文 Algorithm 1 中的 Forest F
# ----------------------------------------------------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, x):
        # 路径压缩
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # 按秩合并
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self):
        g = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return g


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _disk(shape, center_xy, r):
    """在 shape 尺寸的 bool 数组上画实心圆盘 (cv2 椭圆核=真圆)。"""
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (int(round(center_xy[0])), int(round(center_xy[1]))),
               int(round(r)), 1, -1)
    return m.astype(bool)


def _dilate_disk(mask, d):
    """以半径 d 的圆盘做二值膨胀 (论文的 disk area = dilate(V, d))。"""
    if d <= 0:
        return mask.copy()
    k = 2 * d + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _centroid(mask, clip=None):
    """质心, 返回 (x, y)。clip: 仅在该 bool 区域内取点
    (论文: 计算非重叠块中心时只考虑圆盘范围内的点)。"""
    m = mask if clip is None else (mask & clip)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _block_info(label_img, n, extra_masks=None):
    """把 label 图整理成块列表:
    [{id, mask, center_xy, bbox}]  id 从 1 开始。"""
    blocks = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(label_img == i)
        if len(xs) == 0:
            continue
        blocks.append({
            "id": i,
            "mask": label_img == i,
            "center_xy": (float(xs.mean()), float(ys.mean())),
            "bbox": (xs.min(), ys.min(), xs.max(), ys.max()),
        })
    return blocks


# ----------------------------------------------------------------------
# 步骤 0: 交叉区域检测 (阶段一的经典 CV 替代)
# ----------------------------------------------------------------------
def detect_crossing_region(foreground, r_factor=1.0, min_r=2.0):
    """
    经典方法近似论文阶段一 (region-guided UNet++) 的输出:
      骨架化 -> 找度数>=3的分叉节点(十字/T/L中心)
      -> 以 dt[节点]*r_factor 为半径画圆盘, 求并 -> 交叉区域。

    参数
    ----------
    foreground : bool (H, W)  染色体前景掩码
    r_factor   : 圆盘半径系数。90度交叉时 dt(中心) = 0.707*W = 交叉
                 正方形外接圆半径, 1.0 即可完整覆盖; 过大会吞掉臂。
    min_r      : 最小半径, 过滤骨架毛刺

    返回 bool (H, W) 交叉区域掩码 (已与前景求交)
    """
    fg = foreground.astype(bool)
    dt = ndimage.distance_transform_edt(fg)

    if _HAS_SKIMAGE:
        skel = skeletonize(fg)
    else:  # 形态学骨架退化实现
        skel = fg.copy()
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            er = cv2.erode(skel.astype(np.uint8), k).astype(bool)
            op = cv2.dilate(er, k).astype(bool)
            hit = skel & ~op
            new = skel & ~hit
            if new.sum() == skel.sum() or not new.any():
                break
            skel = new

    # 8 邻域度数 (pad 避免环绕)
    p = np.pad(skel, 1)
    deg = np.zeros(skel.shape, np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            deg += p[dy:dy + skel.shape[0], dx:dx + skel.shape[1]]
    junction = skel & (deg >= 3)

    ys, xs = np.nonzero(junction)
    out = np.zeros(fg.shape, bool)
    for y, x in zip(ys, xs):
        r = float(dt[y, x]) * r_factor
        if r < min_r:
            continue
        cv2.circle(out.view(np.uint8), (int(x), int(y)), int(round(r)), 1, -1)
    return out & fg


# ----------------------------------------------------------------------
# 步骤 1: split_blocks —— 前景切成 重叠块V / 非重叠块S
# ----------------------------------------------------------------------
def split_blocks(foreground, overlap_region):
    r"""
    论文中由分水岭+区域掩码得到 V/S 两组块。这里用连通域:
      V = connected_components(foreground ∩ overlap_region)   交叉块
      S = connected_components(foreground \ overlap_region)   臂/非交叉块

    返回 (V_blocks, S_blocks, V_labels, S_labels)
    """
    fg = foreground.astype(bool)
    ov = overlap_region.astype(bool) & fg

    V_lab, nV = ndimage.label(ov)
    S_lab, nS = ndimage.label(fg & ~ov)

    V = _block_info(V_lab, nV)
    S = _block_info(S_lab, nS)
    return V, S, V_lab, S_lab


# ----------------------------------------------------------------------
# 步骤 2: build_dictionary —— 圆盘范围 d 内 V-S 邻接字典 D[V]
# ----------------------------------------------------------------------
def build_dictionary(V, S, S_labels, d):
    """
    论文 Algorithm 1 · Function BuildDictionary:
      for Vi: for Sj: if Sj adjacent to Vi within scope d -> D[Vi].add(Sj)

    实现: 对每个 V 做半径 d 的膨胀, 取区域内出现的 S 标签集合。
    """
    D = {}
    for v in V:
        region = _dilate_disk(v["mask"], d)          # disk area of Vi
        ids = np.unique(S_labels[region])
        D[v["id"]] = [int(i) for i in ids if i != 0]
    return D


# ----------------------------------------------------------------------
# 步骤 3: classify —— 射线配对 (核心启发式)
# ----------------------------------------------------------------------
def _ray_emit(v_center, s_center, S_labels, scope, s_id, w_max=np.inf,
              step=0.5):
    """
    论文 Algorithm 1 · Ray_Emit:
      从 Sj 中心 -> Vi 中心 -> 越过 Vi 中心继续向前扫描;
      若在 disk area(scope) 内碰到另一个非重叠块 Sh, 则 Sj 与 Sh 同条染色体。

    扩展: 宽度校验 —— 命中块在命中点处的横向(射线法向)宽度若 >= w_max,
      说明命中的是"粗大的块"(如 T 形中连通两侧臂的大块)而非细条状的
      对侧臂, 拒绝配对。这对应论文提到的失败模式之一。

    返回: 命中的 S 块 id, 或 None
    """
    dx = v_center[0] - s_center[0]
    dy = v_center[1] - s_center[1]
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        return None
    ux, uy = dx / norm, dy / norm

    H, W = S_labels.shape
    t = 0.0
    max_t = float(np.hypot(W, H))          # 图像对角线, 循环会提前 break
    while t < max_t:
        px = int(round(v_center[0] + ux * t))
        py = int(round(v_center[1] + uy * t))
        if not (0 <= px < W and 0 <= py < H):
            break
        if t > 0 and not scope[py, px]:    # 离开 disk area -> 停止
            break
        lab = int(S_labels[py, px])
        if lab != 0:
            if lab == s_id:
                return None                # 打回自身(臂弯折), 阻断
            # ---- 宽度校验: 法向扫描命中块 ----
            pvx, pvy = -uy, ux              # 射线法向
            for sgn in (1.0, -1.0):
                tt = step
                while tt <= w_max:
                    qx = int(round(px + sgn * pvx * tt))
                    qy = int(round(py + sgn * pvy * tt))
                    if not (0 <= qx < W and 0 <= qy < H):
                        break
                    if int(S_labels[qy, qx]) != lab:
                        break
                    tt += step
                if tt > w_max:              # 横向过宽 -> 非细条臂
                    return None
            return lab
        t += step
    return None


def _contact_fraction(S_mask, V_mask, vc, bins=36):
    """
    S 块与 V 块的"接触弧"占 V 周界的角域覆盖比例。
    细条状的臂只接触 V 的一条边 (~90°/360° ≈ 0.25);
    T 形中连通两侧的粗块 C 会绕 V 一圈(左+底+右 ≈ 0.5+)。
    比例过高说明该块不是"一条臂", 拒绝与其相关的配对。
    """
    contact = ndimage.binary_dilation(S_mask, iterations=1) & V_mask
    ys, xs = np.nonzero(contact)
    if len(xs) == 0:
        return 1.0                              # 无接触视为异常
    ang = np.arctan2(ys - vc[1], xs - vc[0])    # (-pi, pi]
    idx = ((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    return len(np.unique(idx)) / bins


def classify(V, S, D, S_labels, d, dt):
    """
    论文 Algorithm 1 · Function Classify:
      对每个 Vi 及其邻接 Sj:
        1) Sj 中心只取 *圆盘范围内* 的点 (论文原文要求, 防止弯曲臂质心偏移)
        2) 发射射线 Sj中心 -> Vi中心 -> 越过中心
        3) 碰到 Sh 则 union(Sj, Sh)

    扩展: 每个 V 的允许命中宽度 w_max = 3.6 * max(dt over Vi)。
      细条臂宽度约 2*dt, 系数 3.6 留出斜交/膨胀余量;
      T 形中连通两侧的粗块 (宽度可达整幅图) 会被拒绝。
    扩展: 接触弧校验 —— 发送方 Sj 与接收方 Sh 各自与 V 的接触弧
      占比 > 0.45 时拒绝配对 (粗块绕 V 多边接触, 臂只接触一条边)。

    返回 UnionFind (森林 F)
    """
    S_map = {s["id"]: s for s in S}
    uf = UnionFind([s["id"] for s in S])

    for v in V:
        scope = _dilate_disk(v["mask"], d)            # disk area of Vi
        w_max = 3.6 * float(dt[v["mask"]].max())      # 命中宽度上限
        vc = v["center_xy"]
        for sj_id in D.get(v["id"], []):
            sj = S_map[sj_id]
            # 发送方校验: 接触弧过宽 = 粗块 -> 跳过
            if _contact_fraction(sj["mask"], v["mask"], vc) >= 0.45:
                continue
            sc = _centroid(sj["mask"], clip=scope)    # 仅圆盘内的点求中心
            if sc is None:
                sc = sj["center_xy"]
            sh = _ray_emit(vc, sc, S_labels, scope, sj_id, w_max=w_max)
            if sh is None:
                continue
            # 接收方校验: 命中的是粗块 -> 拒绝
            if _contact_fraction(S_map[sh]["mask"], v["mask"], vc) >= 0.45:
                continue
            uf.union(sj_id, sh)
    return uf


# ----------------------------------------------------------------------
# 步骤 4: assemble —— 森林树 -> 单条染色体
# ----------------------------------------------------------------------
def assemble(V, S, uf, V_labels, S_labels, adj_d=2):
    """
    论文 Algorithm 1 主循环:
      对每棵树: 树内所有 S 块 + 与其相邻的 V 块 = 一条染色体
      (交叉块 V 与多棵树都相邻时, 会被多条染色体共享 —— 符合"两条
       染色体都穿过交叉区"的物理事实)

    扩展(论文未写): 未被任何树认领的 V 块按最近邻归属, 避免丢失像素。

    返回: list[bool mask]
    """
    S_map = {s["id"]: s for s in S}
    results = []
    used_v = set()

    k = 2 * adj_d + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for _, members in sorted(uf.groups().items()):
        smask = np.zeros(V_labels.shape, bool)
        for sid in members:
            smask |= S_map[sid]["mask"]
        # 相邻 V 块 (论文: Vi adjacent to node)
        grown = cv2.dilate(smask.astype(np.uint8), kernel).astype(bool)
        vids = np.unique(V_labels[grown])
        vmask = np.zeros(V_labels.shape, bool)
        for vi in vids:
            if vi != 0:
                vmask |= (V_labels == vi)
                used_v.add(int(vi))
        results.append(smask | vmask)

    # ---- 扩展: 收留无人认领的 V 块 (归属最近的染色体) ----
    leftover = [v for v in V if v["id"] not in used_v]
    if leftover and results:
        for v in leftover:
            best, best_d = 0, np.inf
            for i, m in enumerate(results):
                dist = ndimage.distance_transform_edt(~m)
                dv = float(dist[v["mask"]].min()) if v["mask"].any() else np.inf
                # 用 bbox 粗筛加速
                if dv < best_d:
                    best_d, best = dv, i
            results[best] |= v["mask"]
    return results


# ----------------------------------------------------------------------
# 主入口: crossing_partition
# ----------------------------------------------------------------------
def crossing_partition(foreground, overlap_region=None, d=10,
                       ov_dilate=3, detect_kw=None, return_debug=False):
    """
    crossing-partition 完整流水线 (论文阶段二)。

    参数
    ----------
    foreground      : bool/0-255 (H, W) 染色体前景
    overlap_region  : bool (H, W) 交叉区域。None 时用经典方法自动检测
                      (detect_crossing_region, 即阶段一的替代)
    d               : disk area 半径, 论文取 8~12 像素
    ov_dilate       : 交叉区域膨胀半径。斜交时两条臂在交叉区顶点处
                      4-连通相碰会把不同染色体的臂粘成一个块,
                      膨胀几像素将其隔开 (顶点区域划入 V)。0 = 关闭。
    detect_kw       : 传给 detect_crossing_region 的参数 dict
    return_debug    : True 时额外返回 dict(overlap, V_labels, S_labels, ...)

    返回
    ----------
    list[bool mask]  每条(重建后的)染色体掩码; 或 (list, debug)
    """
    fg = foreground.astype(bool)
    if overlap_region is None:
        overlap_region = detect_crossing_region(fg, **(detect_kw or {}))
    ov = overlap_region.astype(bool) & fg
    if ov_dilate > 0:                        # 隔开斜交顶点处相碰的臂
        k = 2 * int(ov_dilate) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ov = cv2.dilate(ov.astype(np.uint8), kernel).astype(bool) & fg

    dt = ndimage.distance_transform_edt(fg)

    V, S, V_labels, S_labels = split_blocks(fg, ov)          # 步骤1
    D = build_dictionary(V, S, S_labels, d)                  # 步骤2
    uf = classify(V, S, D, S_labels, d, dt)                  # 步骤3
    masks = assemble(V, S, uf, V_labels, S_labels)           # 步骤4

    if return_debug:
        dbg = dict(overlap=ov, V_labels=V_labels, S_labels=S_labels,
                   V=V, S=S, D=D, uf=uf)
        return masks, dbg
    return masks
