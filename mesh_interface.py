# -*- coding: utf-8 -*-
"""N 波 N6：interface 处理（纯 numpy，双环境可用）。

多区域体网格/分块网格中，区域间共享边界面被重复生成（coincident interface），
或周期边界按某平移矢量配对（periodic）。本模块做几何配对：

  - `interface_faces`：给定共享顶点 V + 分区面索引 regions（list of 面索引数组，
    index 指向同一 V），检测不同区域间**几何重合**的面片对（centroid 键去重 →
    同名三角形视作同一面）→ 返回每对区域的 interface 数量与匹配对；
  - `periodic_pairs`：对单个封闭面上的两个边界面集，按平移矢量 translation
    匹配「面_i 顶点 + t == 面_j 顶点」的面片对，用于周期边界（偶极、旋转周期
    可由多个平移矢量拆解，此处一次一个矢量）。

用途：interface 数量守恒、周期边界配对面统计、区域贴合度核验；供 P 波
(overset/sliding interface) 与求解器周期 BC 消费。

验收（tests/test_mesh_quality_n6.py）：两块共享面 x=const 的立方体，跨区域
interface 面数 = 共享面三角数、无对内重合；周期盒沿某轴平移匹配的面数 = 该轴
对向面三角数。
"""
import numpy as np


def _face_key(V, F, tol):
    """把三角形 (a,b,c) → (质心, 排序后顶点坐标元组) 的规范化键，tol 控制量化精度。"""
    pts = V[F]
    c = pts.mean(axis=0)
    cq = tuple(round(float(x), tol) for x in c)
    # 顶点坐标量化成元组并排序（绕向无关），顶点点集再整体排序
    srt = tuple(sorted(tuple(round(float(x), tol) for x in p) for p in pts))
    return cq, srt


def interface_faces(V, regions, tol=6):
    """检测不同 region 间几何重合面片对。

    参数：
      V：Nx3 共享顶点数组；
      regions：list，每个元素为 M_i x 3 的三角形顶点索引数组（index 指向 V）；
      tol：坐标量化小数位（默认 6 位，即 1e-6 精度）。

    返回 {n_regions, pairs, counts, matched}：
      pairs：list of (ri, rj, fi, fj)；counts：{(ri,rj): n}；
      matched：bool —— 每个 region 所有面都被恰好配对（不跨 region 自配）。
    """
    V = np.asarray(V, float)
    regions = [np.asarray(F, np.int64) for F in regions]
    keys = []
    for F in regions:
        keys.append([_face_key(V, row, tol) for row in F])

    # 每个面 → 唯一键计数，找出所有区域中共现的面
    from collections import defaultdict
    slot = defaultdict(list)                     # key -> [(ri, fi)]
    all_idx_global = []
    for ri, (F, ks) in enumerate(zip(regions, keys)):
        for fi, k in enumerate(ks):
            slot[k].append((ri, fi))
    pairs = []
    counts = {}
    for k, occ in slot.items():
        if len(occ) < 2:
            continue
        # 只统计跨区域对（同一区域内共享边界的重复面不视作 interface）
        for i in range(len(occ)):
            ri, fi = occ[i]
            for j in range(i + 1, len(occ)):
                rj, fj = occ[j]
                if ri == rj:
                    continue
                if ri > rj:
                    ri, fi, rj, fj = rj, fj, ri, fi
                pairs.append((ri, rj, int(fi), int(fj)))
                counts[(ri, rj)] = counts.get((ri, rj), 0) + 1

    # matched：每个 region 的每个面，在跨区域配对中恰好出现一次
    matched_count = defaultdict(int)
    for ri, rj, fi, fj in pairs:
        matched_count[(ri, fi)] += 1
        matched_count[(rj, fj)] += 1
    n_total = sum(len(F) for F in regions)
    n_matched = sum(1 for (r, f), cnt in matched_count.items() if cnt == 1)
    return {"n_regions": len(regions), "pairs": pairs,
            "counts": dict(counts),
            "matched": bool(n_matched == n_total)}


def periodic_pairs(V, F, translation, tol=6):
    """按平移矢量匹配周期面片对。

    面_i 的顶点 + translation 与面_j 的顶点（作为多重集）相符 → (i, j) 配对。
    返回 {n_faces, pairs, n_matched, matched(bool)}；无配对返回空。
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    t = np.asarray(translation, float).reshape(3)
    if len(F) == 0:
        return {"n_faces": 0, "pairs": [], "n_matched": 0, "matched": False}

    def key_of(idx, shift):
        pts = V[idx] + shift
        c = pts.mean(axis=0)
        srt = tuple(sorted(tuple(round(float(x), tol) for x in p) for p in pts))
        return (tuple(round(float(x), tol) for x in c), srt)

    # 建立 `面 F_j 顶点集 + t` 的反查表 —— 以「F_j + t 规范化键」检索 F_i
    plus = {}
    for j, row in enumerate(F):
        k = key_of(row, t)
        if k not in plus:
            plus[k] = j
    pairs = []
    for i, row in enumerate(F):
        k = key_of(row, np.zeros(3))             # F_i 原键
        j = plus.get(k)
        if j is not None and j != i:
            pairs.append((int(i), int(j)))
    seen = set()
    uniq = []
    for i, j in pairs:
        if j in seen:
            continue
        seen.add(i)
        seen.add(j)
        uniq.append((i, j))
    return {"n_faces": len(F), "pairs": uniq,
            "n_matched": len(uniq), "matched": bool(len(uniq) > 0)}
