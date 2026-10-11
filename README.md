# G显带染色体十字交叉修复

去掉交叉重叠图像中的**下层**染色体、识别并保留**上层**染色体（重建被遮挡处、补背景）。

走 ChromSeg 论文两阶段路线（区域检测 + crossing-partition 几何重组），并补上论文未解决的"上下层判断"问题。

## 方法

1. **阶段一：交叉区检测** — 经典 CV 实现（骨架分叉点 + 距离变换画圆盘），交叉区也可由外部/数据集直接传入
2. **阶段二：crossing-partition 分割** — 在论文伪代码基础上加三处修正：
   - `ov_dilate = 3`（交叉区并入邻接块）
   - 宽度校验 `w_max = 3.6 × dt(V)`（拒绝穿心射线）
   - 接触弧校验 ≥ 0.45（拒绝 T 形粘连）
3. **上下层判断（论文未解决，本项目提出）** — 轮廓连续性为主线索（骨架直径路径上的法向台阶差），剖面跳变作低置信裁决，低置信输出 `uncertain=True` 转人工确认

## 运行

```powershell
py demo_crossing_partition.py   # 合成4形态 demo (分割+层判断+删底层)
py test_layer_order.py          # 层判断64例鲁棒性测试
py compare_official.py          # 与 ChromSeg 官方代码正面对照
py run_real_pipeline.py         # ChromSeg 345张真实数据集全流水线
```

依赖：numpy / scipy / opencv-python / scikit-image / matplotlib / pymupdf

## 主要结果

| 实验 | 官方代码 | 本项目 |
|---|---|---|
| GT交叉区×4形态×3种子 | 9/12, mean IoU 0.890 | **12/12, mean IoU 0.976** |
| T 形粘连案例 | 0.667（射线穿心导致3臂并条） | 通过（宽度/接触弧校验） |
| 层判断16种子×4形态 | — | 63/64 = 98.4%；弃权后自动判定 54/54 = 100% |
| 真实数据集345张 | — | 全部跑通 17ms/张；前景覆盖 1.000 |

## 目录结构

- `crossing_partition.py` — 阶段二核心分割算法
- `layer_order.py` — 上下层判断（3处修正 + 跳变裁决 + 弃权标志）
- `demo_crossing_partition.py` — 主 demo
- `compare_official.py` — 官方对照实验
- `run_real_pipeline.py` / `ablate_fg.py` — 真实数据集流水线与前景提取消融
- `data/ChromSeg/` — 官方345张数据集（`image_*` 原图 / `binary_label_*` 交叉核 / `binary_label2_*` 臂）
- `output/` — 结果可视化

## 说明

- ChromSeg 官方代码（`third_party/`，本地参考）**无 LICENSE，未随仓库分发**，仅用于对照实验
- 数据集来源：ChromSeg 论文主页 http://www.bio8.cs.hku.hk/bibm/
