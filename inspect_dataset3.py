# -*- coding: utf-8 -*-
"""几何分析: lab / lab2 / 图像暗区 三者结构关系
   假设检验: lab=交叉区, lab2=非重叠臂 (二者并集=该交叉簇前景)"""
import glob
import os
import cv2
import numpy as np
from scipy import ndimage

D = os.path.join("data", "ChromSeg", "original_dataset")
imgs = sorted(glob.glob(os.path.join(D, "image_*.png")),
              key=lambda p: int(os.path.basename(p)[6:-4]))

for k in (0, 4, 7):
    n = os.path.basename(imgs[k])[6:-4]
    g = cv2.imread(imgs[k], 0)
    lab = cv2.imread(os.path.join(D, f"binary_label_{n}.png"), 0) > 0
    r = cv2.imread(os.path.join(D, f"binary_label2_{n}.png"),
                   cv2.IMREAD_UNCHANGED)
    lab2 = (r[:, :, :3].astype(int).sum(2) > 0)
    union = lab | lab2
    dark = g < 200

    lab2_lab, lab2_n = ndimage.label(lab2)
    sizes = np.bincount(lab2_lab.ravel())[1:]
    ys, xs = np.nonzero(lab)
    ly, lx = np.nonzero(lab2)

    print(f"\n#{n}:")
    print(f"  lab:  size={lab.sum()} bbox=x[{xs.min()},{xs.max()}] "
          f"y[{ys.min()},{ys.max()}]")
    print(f"  lab2: size={lab2.sum()} components={lab2_n} "
          f"sizes={[int(s) for s in sizes]}")
    print(f"  union vs dark: |u|={union.sum()} |dark|={dark.sum()} "
          f"IoU={np.logical_and(union, dark).sum()/np.logical_or(union, dark).sum():.3f}")
    # lab 到 lab2 的最近距离 (0=相接)
    dist = ndimage.distance_transform_edt(~lab2)
    print(f"  lab->lab2 dist: min={dist[lab].min():.1f} "
          f"median={np.median(dist[lab]):.1f} max={dist[lab].max():.1f}")
    # lab 周围接触的 lab2 连通域数量 (交叉处应 >=3 条臂)
    touch = set(np.unique(lab2_lab[ndimage.binary_dilation(lab, iterations=2)
                                   & lab2_lab])) - {0}
    print(f"  lab 接触的 lab2 臂数: {len(touch)}")
    # lab 是否被 lab2 的凸包/臂包围: lab 中心到 union 边界距离
    print(f"  dark 覆盖 union: {np.logical_and(dark, union).sum()/union.sum():.3f}")
