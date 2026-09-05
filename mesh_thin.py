# -*- coding: utf-8 -*-
"""N 波 N5（二）：thin mesher 薄壁网格器（纯 numpy，双环境可用）。

STAR-CCM+ thin mesher 的核心思想：薄壁区域（壁厚 ≤ 阈值）不生成四面体，
而是每个薄面**一次贯通整个壁厚**生成 prism 列（可多单元分层），两侧壁
由同一列连接。

实现（`thin_mesh`）：
  - 薄面判定（面级）：面质心沿面向内法向射线到最近表面 h_f ≤ thickness
    ——面法向不受角点对角污染（[0,1]²×0.05 盒的侧壁面法向 ±y 量得 1.0
    不薄，顶点法向在角点是对角线会误判）；
  - 法向净化：顶点推进方向只用**薄面**累加的面积加权法向（厚壁面不
    参与——立方体角点法向由对角转为纯薄面法向，平壁列体积精确），
    逐顶点沿净化法向射线取 h_v/t_v 做提升与配对；
  - 对面配对去重：薄面 f 的质心命中面 t_f，f < t_f 时 f 为源面（配对
    中取较小索引一侧贯通），t_f 标记 consumed 防链式重复贯通；
    提升点按键 (v, j) 顶点共享 → 相邻列侧壁共享、拼合无缝；
  - 每列 n_through 个单元均分全厚：level j 深度 = j·h_v/n_through（逐
    顶点深度各自贯通到对面命中点）；
  - side 同 prism_layers（inward/outward 闭合面探针自动定向，开放面
    +n/-n）。

边界（已知限制，超出薄壁语义）：厚度阈值 ≥ 两向跨度时（多向同时"薄"），
不同向的列会互相重叠——该类输入应走 tet/poly/trimmer。
质量走 mesh_prism.prism_quality（有向体积翻转检测）。

验收要点（tests/test_mesh_thin.py）：
  薄盒 [0,1]²×0.05：仅底/顶面薄、底面 2 列贯通、体积=盒体积精确
  （n_through=1/2 一致）；厚立方体：无薄面 ok=False；单薄向盒
  1×1×0.3@阈值0.5：侧壁不薄、体积=0.3 精确；球壳 r=0.9..1.0：
  全部外球面贯通、体积≈壳体积（10% 内）、零翻转。
"""
import numpy as np

from mesh_prism import (ray_hit_nearest, _side_sign, prism_quality)


def _normals_from_faces(V, F, face_ids):
    """面积加权顶点法向，只累计 face_ids 中的面（无邻接面 → 零向量）。"""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    acc = np.zeros_like(V)
    for fi in face_ids:
        a, b, c = F[fi]
        fn = np.cross(V[b] - V[a], V[c] - V[a])
        acc[a] += fn
        acc[b] += fn
        acc[c] += fn
    ln = np.linalg.norm(acc, axis=1)
    out = np.zeros_like(acc)
    ok = ln > 1e-300
    out[ok] = acc[ok] / ln[ok][:, None]
    return out


def thin_mesh(vertices, faces, thickness, n_through=1, side="inward"):
    """薄壁网格器：薄面一次贯通全厚生成 prism 列（含对面配对去重）。

    参数：
      thickness：薄壁阈值（面质心沿法向壁厚 h_f ≤ 之判薄，>0）；
      n_through：每列贯通单元数（均分全厚）；side：同 prism_layers
      （inward/outward 需闭合表面自动定向，开放面用 "+n"/"-n"）。

    返回 {ok, vertices, cells, face_sources, n_thin, n_sources, n_consumed,
    face_h, thickness, n_through, h_stats, quality, volume_total}；空输入/
    参数非法抛 ValueError。
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or len(V) == 0 or len(F) == 0:
        raise ValueError("thin_mesh 需要非空表面 (vertices Nx3, faces Mx3)")
    N = len(V)
    th = float(thickness)
    if not (th > 0.0):
        raise ValueError("thickness 须为正数")
    k = max(1, int(n_through))
    sign = _side_sign(V, F, side)

    # 薄面判定（面级：质心沿面向内法向的壁厚）
    thin = set()
    t_of = {}                                    # 面 → 质心命中面索引
    h_of = {}                                    # 面 → 质心壁厚
    for fi, (a, b, c) in enumerate(F):
        A, B, C = V[a], V[b], V[c]
        n = np.cross(B - A, C - A)
        ln = float(np.linalg.norm(n))
        if ln < 1e-300:
            continue                             # 退化面
        d = sign * (n / ln)
        cen = (A + B + C) / 3.0
        hf, tf = ray_hit_nearest(cen, d, V, F)
        h_of[fi], t_of[fi] = hf, tf
        if np.isfinite(hf) and hf <= th:
            thin.add(fi)

    # 源面：薄面中 f < 质心命中面 t_f（配对取小索引侧贯通），命中面
    # consumed 防链式重复；按面索引序处理（确定性）。
    sources = []
    consumed = set()
    for fi in sorted(thin):
        if fi in consumed:
            continue
        tf = t_of.get(fi, -1)
        if tf < 0 or fi >= tf:
            continue                             # 对面（更小索引）侧贯通
        sources.append(fi)
        consumed.add(int(tf))

    # 顶点推进方向：只用薄面累加的面积加权法向（厚壁面不污染角点）
    dirs = sign * _normals_from_faces(V, F, thin)
    h = np.full(N, np.inf)
    for v in range(N):
        if float(np.linalg.norm(dirs[v])) < 1e-12:
            continue
        h[v] = ray_hit_nearest(V[v], dirs[v], V, F)[0]

    # 提升点（按顶点共享）：level j = v + dirs[v]·(j·h_v/n_through)
    newV = [list(map(float, p)) for p in V]
    lift = {}
    for v in range(N):
        if not np.isfinite(h[v]) or float(np.linalg.norm(dirs[v])) < 1e-12:
            continue
        for j in range(1, k + 1):
            lift[(v, j)] = len(newV)
            newV.append((V[v] + dirs[v] * (h[v] * j / k)).tolist())

    cells = []
    for fi in sources:
        a, b, c = F[fi]
        if (a, 1) not in lift or (b, 1) not in lift or (c, 1) not in lift:
            continue                             # 顶点无方向（退化），跳过
        for j in range(1, k + 1):
            a0 = a if j == 1 else lift[(a, j - 1)]
            b0 = b if j == 1 else lift[(b, j - 1)]
            c0 = c if j == 1 else lift[(c, j - 1)]
            cells.append([a0, b0, c0, lift[(a, j)], lift[(b, j)],
                          lift[(c, j)]])
    cells_arr = (np.asarray(cells, np.int64) if cells
                 else np.zeros((0, 6), np.int64))
    src_set = set(sources)
    face_sources = [fi in src_set for fi in range(len(F))]
    hv = h[np.isfinite(h)]
    quality = prism_quality(newV, cells_arr)
    return {
        "ok": len(cells) > 0,
        "vertices": np.asarray(newV, float),
        "cells": cells_arr,
        "face_sources": face_sources,
        "n_thin": len(thin),
        "n_sources": len(sources),
        "n_consumed": len(consumed),
        "face_h": h_of,
        "thickness": th,
        "n_through": k,
        "h_stats": (float(hv.min()), float(hv.max())) if len(hv)
                   else (float("inf"), float("inf")),
        "quality": quality,
        "volume_total": quality["volume_total"],
    }
