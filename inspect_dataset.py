# -*- coding: utf-8 -*-
"""确认 binary_label / binary_label2 的语义 (通道分析) + 预览"""
import glob
import os
import cv2
import numpy as np

D = os.path.join("data", "ChromSeg", "original_dataset")
imgs = sorted(glob.glob(os.path.join(D, "image_*.png")),
              key=lambda p: int(os.path.basename(p)[6:-4]))

for k in (0, 1, 5):
    n = os.path.basename(imgs[k])[6:-4]
    img = cv2.imread(imgs[k])
    lab = cv2.imread(os.path.join(D, f"binary_label_{n}.png"), 0)
    lab2 = cv2.imread(os.path.join(D, f"binary_label2_{n}.png"),
                      cv2.IMREAD_UNCHANGED)
    print(f"\n#{n}: lab2 shape={lab2.shape}")
    if lab2.ndim == 3:
        for c in range(lab2.shape[2]):
            ch = lab2[:, :, c]
            u = np.unique(ch)
            print(f"  ch{c}: vals={u[:5]}{'...' if len(u)>5 else ''} "
                  f"fg%={100*(ch>0).mean():.1f}")
        bgr, alpha = lab2[:, :, :3], lab2[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        m_bgr = gray > 0
        m_a = alpha > 0
        print(f"  bgr>0%={100*m_bgr.mean():.1f}  alpha>0%={100*m_a.mean():.1f}  "
              f"bgr==alpha? {np.array_equal(m_bgr, m_a)}")
        m2 = m_bgr & m_a
    else:
        m2 = lab2 > 0
    m1 = lab > 0
    print(f"  lab(fg%={100*m1.mean():.1f}) subset of lab2? "
          f"{np.all(m2 | ~m1)};  lab2 subset of lab? {np.all(m1 | ~m2)}")
    # 与前景的关系: image 灰度图暗区 = 染色体?
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  mean gray: lab区域={g[m1].mean() if m1.any() else -1:.0f}  "
          f"lab2区域={g[m2].mean() if m2.any() else -1:.0f}  "
          f"背景={g[~(m1|m2)].mean():.0f}")

# 预览 #0
n = os.path.basename(imgs[0])[6:-4]
img = cv2.imread(imgs[0])
lab = cv2.imread(os.path.join(D, f"binary_label_{n}.png"), 0)
lab2r = cv2.imread(os.path.join(D, f"binary_label2_{n}.png"),
                   cv2.IMREAD_UNCHANGED)
m2 = (lab2r[:, :, :3].astype(int).sum(2) > 0) & (lab2r[:, :, 3] > 0)
h, w = img.shape[:2]
canvas = np.zeros((h, w * 4, 3), np.uint8)
canvas[:, :w] = img
canvas[:, w:2*w] = cv2.cvtColor((lab > 0).astype(np.uint8) * 255,
                                cv2.COLOR_GRAY2BGR)
canvas[:, 2*w:3*w] = cv2.cvtColor((m2.astype(np.uint8)) * 255,
                                  cv2.COLOR_GRAY2BGR)
ov = img.copy()
ov[m2] = (ov[m2] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
ov[lab > 0] = (ov[lab > 0] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
canvas[:, 3*w:] = ov
scale = 1400 / canvas.shape[1]
canvas = cv2.resize(canvas, (1400, int(canvas.shape[0] * scale)))
out = os.path.join("output", "dataset_preview.png")
cv2.imwrite(out, canvas)
print("\nsaved", out)
