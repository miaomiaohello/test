# -*- coding: utf-8 -*-
"""用数值确定 binary_label / binary_label2 语义:
   图像暗区(前景) vs 两个标签的交并关系、直方图"""
import glob
import os
import cv2
import numpy as np

D = os.path.join("data", "ChromSeg", "original_dataset")
imgs = sorted(glob.glob(os.path.join(D, "image_*.png")),
              key=lambda p: int(os.path.basename(p)[6:-4]))

print(f"{'n':>5s} {'img<128%':>9s} {'img<200%':>9s} "
      f"{'lab%':>7s} {'lab2%':>7s} {'lab&lab2%':>9s} "
      f"{'labInDark':>9s} {'lab2InDrk':>9s}")
for p in imgs[:12]:
    n = os.path.basename(p)[6:-4]
    g = cv2.imread(p, 0)
    lab = cv2.imread(os.path.join(D, f"binary_label_{n}.png"), 0) > 0
    r = cv2.imread(os.path.join(D, f"binary_label2_{n}.png"),
                   cv2.IMREAD_UNCHANGED)
    lab2 = (r[:, :, :3].astype(int).sum(2) > 0)   # BGR 白色区域
    dark128 = g < 128
    dark200 = g < 200
    inter = (lab & lab2).sum()
    print(f"{n:>5s} {100*dark128.mean():9.1f} {100*dark200.mean():9.1f} "
          f"{100*lab.mean():7.2f} {100*lab2.mean():7.2f} "
          f"{100*inter/max(lab.sum(),1):9.1f} "
          f"{str(np.all(dark200 | ~lab)):>8s} "
          f"{str(np.all(dark200 | ~lab2)):>9s}")

# 统计全体: lab 是否总在图像暗区内; lab2 与暗区的关系
print("\n全体345张:")
cnt = dict(lab_in_dark=0, lab2_in_dark=0, lab2_covers_dark=0)
ratios_l2 = []
for p in imgs:
    n = os.path.basename(p)[6:-4]
    g = cv2.imread(p, 0)
    lab = cv2.imread(os.path.join(D, f"binary_label_{n}.png"), 0) > 0
    r = cv2.imread(os.path.join(D, f"binary_label2_{n}.png"),
                   cv2.IMREAD_UNCHANGED)
    lab2 = (r[:, :, :3].astype(int).sum(2) > 0)
    dark = g < 200
    cnt["lab_in_dark"] += int(np.all(dark | ~lab))
    cnt["lab2_in_dark"] += int(np.all(dark | ~lab2))
    # lab2 覆盖了多少暗区
    ratios_l2.append((lab2 & dark).sum() / max(dark.sum(), 1))
print("  lab subset of dark(g<200):", cnt["lab_in_dark"], "/345")
print("  lab2 subset of dark(g<200):", cnt["lab2_in_dark"], "/345")
print(f"  lab2 覆盖暗区比例: mean={np.mean(ratios_l2):.2f} "
      f"min={np.min(ratios_l2):.2f} max={np.max(ratios_l2):.2f}")
