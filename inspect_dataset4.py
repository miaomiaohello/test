# -*- coding: utf-8 -*-
"""横穿 lab->空隙->lab2 的灰度剖面 + lab 形态学角色确认"""
import glob
import os
import cv2
import numpy as np
from scipy import ndimage

D = os.path.join("data", "ChromSeg", "original_dataset")
imgs = sorted(glob.glob(os.path.join(D, "image_*.png")),
              key=lambda p: int(os.path.basename(p)[6:-4]))


def load(k):
    n = os.path.basename(imgs[k])[6:-4]
    g = cv2.imread(imgs[k], 0)
    lab = cv2.imread(os.path.join(D, f"binary_label_{n}.png"), 0) > 0
    r = cv2.imread(os.path.join(D, f"binary_label2_{n}.png"),
                   cv2.IMREAD_UNCHANGED)
    lab2 = (r[:, :, :3].astype(int).sum(2) > 0)
    return n, g, lab, lab2


# ---- 1) 过 lab 质心的水平剖面 ----
n, g, lab, lab2 = load(0)
ys, xs = np.nonzero(lab)
cy = int(ys.mean())
cx = int(xs.mean())
print(f"#0 lab质心=({cx},{cy}), 水平剖面 y={cy}:")
print(" x:    " + "".join(f"{x%100//10}{x%100%10}" for x in range(cx-25, cx+45, 2)))
print(" gray: " + "".join(f"{g[cy, x]//10:2d}" for x in range(cx-25, cx+45, 2)))
print(" lab : " + "".join((" L" if lab[cy, x] else " .")
                          for x in range(cx-25, cx+45, 2)))
print(" lab2: " + "".join((" A" if lab2[cy, x] else " .")
                          for x in range(cx-25, cx+45, 2)))

# ---- 2) lab 区域的图像灰度分布 (是暗染色体还是别的) ----
print(f"\n#0 lab内 gray: mean={g[lab].mean():.0f} "
      f"p10={np.percentile(g[lab],10):.0f} p90={np.percentile(g[lab],90):.0f}")
gap = ndimage.binary_dilation(lab, iterations=3) & ~lab & ~lab2 & (g < 230)
print(f"lab外围3px非lab2区: n={gap.sum()} meanGray={g[gap].mean() if gap.any() else -1:.0f}")

# ---- 3) lab 形状: 是否为两染色体的几何交集 (凸? 长条?) ----
print(f"lab bbox-fill率: {lab.sum()/((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1)):.2f}")

# ---- 4) union 的连通性: lab 是否嵌在 union 内部 ----
union = lab | lab2
u_lab, u_n = ndimage.label(union)
lab_u = u_lab[cy, cx]
print(f"union连通数={u_n}, lab质心所在连通域大小="
      f"{(u_lab==lab_u).sum()} / union={union.sum()}")

# ---- 5) 图像整体: 有几个暗簇 (是单簇patch还是整张散片) ----
dark = g < 200
d_lab, d_n = ndimage.label(dark)
sizes = np.bincount(d_lab.ravel())[1:]
big = sorted([int(s) for s in sizes if s > 50], reverse=True)
print(f"图像暗区连通域(>50px): {len(big)} 个, sizes={big[:15]}")
