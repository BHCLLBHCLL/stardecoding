# -*- coding: utf-8 -*-
"""N 波 N3b（二）：trimmer 八叉树切割网格器。

STAR-CCM+ trimmer mesher 的核心思想：背景八叉树在表面附近逐级加密，
远离表面处保持粗单元；与表面相交的最深叶单元被表面**切割**成贴体多面体
（cut cell），内部为结构六面体、外部丢弃。

实现（纯 numpy，两环境皆可用）：
  - 八叉树：根 = 表面包围盒；叶与任一三角相交（Akenine-Möller，复用
    `occ_wrap._tri_box_overlap`）且未达 `max_depth` → 一分八；
    候选三角沿树下传（子盒 ⊂ 父盒，只需复验父候选）；
  - 叶分类：无三角接触的叶整块或在内部或在外部（表面为其间边界），
    叶心 even-odd 射线判定精确（`mesh_tet.point_in_mesh`）；
  - 切割单元：最深叶从六面盒出发，被邻近三角面的**内侧半空间**逐面裁剪
    （`mesh_poly.clip_convex`）。凸表面（立方体/球等）下为精确贴体；
    非凸表面为局部平面近似。法向外向由实体参考心定向；
  - `trimmer_cell_volume` / `trimmer_quality`：体积（六面体=三向积、
    切割=多面体体积）与统计。

验收要点（tests/test_mesh_trimmer.py）：
  立方体体积守恒 ≈ 1.0（轴对齐面精确切割）；
  球体积 ≈ 4π/3（凸贴体，分辨率容差内）；
  近表面叶 = 目标尺寸、远场叶更粗（八叉树分级）；
  全部单元正体积；开表面 ValueError。
"""
import numpy as np

from mesh_poly import box_polyhedron, clip_convex, poly_volume
from mesh_tet import point_in_mesh, _watertight_ok
from occ_wrap import _tri_box_overlap

_MAX_DEPTH_CAP = 10


def _leaf_target_depth(lo, hi, cell_size):
    """达到目标叶尺寸所需的细分深度。"""
    root = float(np.max(np.asarray(hi, float) - np.asarray(lo, float)))
    cs = max(float(cell_size), 1e-12)
    d = int(np.ceil(np.log2(max(root / cs, 1.0))))
    return max(1, min(d, _MAX_DEPTH_CAP))


def _orient_outward(V, F):
    """三角面外向单位法向 + 面上一点（质心）。参考心取顶点均值（凸体精确）。"""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    p = V[F[:, 0]]
    q = V[F[:, 1]]
    r = V[F[:, 2]]
    n = np.cross(q - p, r - q)
    ln = np.linalg.norm(n, axis=1)
    safe = ln > 1e-300
    n[safe] /= ln[safe][:, None]
    cen = (p + q + r) / 3.0
    ref = V.mean(axis=0)
    flip = np.einsum("ij,ij->i", n, cen - ref) < 0.0
    n[flip] *= -1.0
    return n, cen


def trimmer_mesh(vertices, faces, cell_size, max_depth=None):
    """水密表面 → trimmer 八叉树网格。

    返回 {ok, cells, n_cells, n_hex, n_cut, n_outside, volume_total,
    quality, max_depth}；非水密输入抛 ValueError。
    cells 元素：{"kind":"hex","lo","hi"} 或 {"kind":"cut","vertices","faces"}。
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    if not _watertight_ok(V, F):
        raise ValueError("trimmer 需要水密闭合表面（当前有开放边界）")
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    if max_depth is None:
        max_depth = _leaf_target_depth(lo, hi, cell_size)
    max_depth = max(1, min(int(max_depth), _MAX_DEPTH_CAP))

    nrm, cen = _orient_outward(V, F)
    # 三角 AABB（候选快速剔除用）
    tv = V[F]                                    # (T,3,3)
    tlo = tv.min(axis=1)
    thi = tv.max(axis=1)

    cells = []
    n_outside = 0
    stack = [(lo, hi, 0, np.arange(len(F)))]
    while stack:
        clo, chi, depth, cand = stack.pop()
        clo = np.asarray(clo, float)
        chi = np.asarray(chi, float)
        # 候选：三角 AABB 与叶盒重叠（矢量化粗筛）
        if len(cand):
            m = np.all((tlo[cand] <= chi) & (thi[cand] >= clo), axis=1)
            cand = cand[m]
        cut_ids = [int(t) for t in cand
                   if _tri_box_overlap(clo, chi, V[F[t]])]
        if cut_ids and depth < max_depth:
            mid = 0.5 * (clo + chi)
            for sx in (0, 1):
                for sy in (0, 1):
                    for sz in (0, 1):
                        nlo = np.where([sx, sy, sz], mid, clo)
                        nhi = np.where([sx, sy, sz], chi, mid)
                        stack.append((nlo, nhi, depth + 1, cand))
            continue
        if cut_ids:
            cell = _cut_cell(clo, chi, V, F, nrm, cen, tlo, thi,
                             cut_ids, cand)
            if cell is not None:
                cells.append(cell)
            else:
                n_outside += 1
        else:
            c = 0.5 * (clo + chi)
            if point_in_mesh(c, V, F):
                cells.append({"kind": "hex", "lo": clo, "hi": chi})
            else:
                n_outside += 1
    total = float(sum(_cell_volume(c) for c in cells))
    out = {"ok": len(cells) > 0, "cells": cells, "n_cells": len(cells),
           "n_hex": sum(1 for c in cells if c["kind"] == "hex"),
           "n_cut": sum(1 for c in cells if c["kind"] == "cut"),
           "n_outside": int(n_outside), "volume_total": total,
           "max_depth": max_depth,
           "quality": trimmer_quality(cells)}
    return out


def _cut_cell(clo, chi, V, F, nrm, cen, tlo, thi, cut_ids, cand):
    """最深切割叶：六面盒被邻近三角内侧半空间裁剪；体积过小则丢弃。"""
    size = float(np.max(chi - clo))
    elo = clo - size
    ehi = chi + size
    # 邻近 = 扩展盒内三角（含 cut_ids；凸体下内侧半空间交 = 精确贴体）
    m = np.all((tlo[cand] <= ehi) & (thi[cand] >= elo), axis=1)
    near = [int(t) for t in cand[m]]
    for t in cut_ids:
        if t not in near:
            near.append(t)
    pv, pf = box_polyhedron(clo, chi)
    for t in near:
        if float(np.linalg.norm(nrm[t])) < 1e-12:
            continue                      # 退化三角不参与裁剪
        pv, pf = clip_convex(pv, pf, cen[t], nrm[t])
        if not pf:
            return None
    vol = poly_volume(pv, pf)
    if vol <= 1e-12:
        return None
    return {"kind": "cut", "vertices": pv, "faces": pf, "volume": float(vol)}


def _cell_volume(cell):
    if cell["kind"] == "hex":
        d = np.asarray(cell["hi"], float) - np.asarray(cell["lo"], float)
        return float(np.prod(np.maximum(d, 0.0)))
    return float(cell.get("volume", poly_volume(cell["vertices"],
                                                cell["faces"])))


def trimmer_cell_volume(cell):
    """单个 trimmer 单元体积（六面体=三向积；切割=多面体体积）。"""
    return _cell_volume(cell)


def trimmer_quality(cells):
    """trimmer 单元质量统计：单元数/六面体/切割数、体积和/最小体积。"""
    if not cells:
        return {"n_cells": 0, "n_hex": 0, "n_cut": 0, "volume_total": 0.0,
                "volume_min": 0.0, "ok": False}
    vols = [_cell_volume(c) for c in cells]
    return {"n_cells": len(cells),
            "n_hex": sum(1 for c in cells if c["kind"] == "hex"),
            "n_cut": sum(1 for c in cells if c["kind"] == "cut"),
            "volume_total": float(sum(vols)),
            "volume_min": float(min(vols)),
            "ok": min(vols) > 0.0}
