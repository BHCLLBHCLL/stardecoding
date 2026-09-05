# -*- coding: utf-8 -*-
"""N 波 N3b（一）：poly 多面体网格器 —— tet 的 Voronoi 对偶。

STAR-CCM+ 的 polyhedral mesher 在概念上即「Delaunay 四面体 → Voronoi 对偶」：
每个输入点得到一个凸多面体单元，单元面 ⊥ 连接边且过边中点（Voronoi 性质），
内部单元的顶点恰为共享该点的四面体外接球心（对偶性质）。

实现（纯 numpy，两环境皆可用）：
  - 凸多面体核心：`box_polyhedron`（六面盒）/ `clip_convex`（Sutherland-Hodgman
    3D 平面裁剪：逐面拆内外顶点串、跨界插值、截面盖帽按平面极角排序）；
  - `poly_volume`：散度定理扇形三角化有向体积（外向面序 → 正体积）；
  - `voronoi_dual(vertices, tets)`：对每个 tet 顶点，从包围盒出发，逐个
    Delaunay 邻接点（tet 边邻接）的中垂面裁剪 → 得到该点的（盒裁）Voronoi
    单元。内部点 = 精确 Voronoi 胞；凸包边界点被盒裁剪（ clipped Voronoi，
    全体单元恰好无重叠铺满包围盒）；
  - `poly_mesh(vertices, faces, spacing)`：水密表面 → `mesh_tet.tet_mesh`
    → Voronoi 对偶（poly 网格生成完整工作流）；
  - `poly_quality`：单元数/体积统计。

验收要点（tests/test_mesh_poly.py）：
  均匀栅格 Delaunay 对偶 → 内部单元体积 ≈ h³（盒状）、6 张面；
  全体（含盒裁边界）单元体积和 == 包围盒体积（铺满不变量）；
  立方体表面 tet → poly 体积和 ≈ 1.0。
"""
import numpy as np


# --------------------------------------------------------------- 凸多面体核心
def box_polyhedron(lo, hi):
    """轴对齐盒 [lo,hi] 的外向多面体 (V, F)：8 顶点 + 6 张四边形面。"""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    # 每面按外向法向（右手序）排列
    F = [[0, 3, 2, 1],    # z0 底，法向 -z
         [4, 5, 6, 7],    # z1 顶，法向 +z
         [0, 1, 5, 4],    # y0，法向 -y
         [2, 3, 7, 6],    # y1，法向 +y
         [0, 4, 7, 3],    # x0，法向 -x
         [1, 2, 6, 5]]    # x1，法向 +x
    return V, F


def clip_convex(V, F, p, n, eps=1e-12):
    """凸多面体 (V,F) 被平面 {x: (x-p)·n=0} 裁剪，保留 (x-p)·n ≤ 0 一侧。

    Sutherland-Hodgman 三维版：逐面保留内侧顶点串 + 跨界边插值；
    截面盖帽由全部跨界交点按平面内极角排序构成，法向沿 +n（指向被裁掉一侧，
    保持外向面序）。返回新 (V, F)；若整块被裁掉则返回 ([], [])。
    """
    V = [list(map(float, q)) for q in V]
    n = np.asarray(n, float)
    p = np.asarray(p, float)
    nn = float(np.linalg.norm(n))
    if nn < 1e-300:
        return V, list(F)
    n = n / nn
    eye = {}
    newV = []

    def vid(q):
        key = (round(float(q[0]), 11), round(float(q[1]), 11),
               round(float(q[2]), 11))
        if key in eye:
            return eye[key]
        eye[key] = len(newV)
        newV.append([float(q[0]), float(q[1]), float(q[2])])
        return eye[key]

    for q in V:                      # 原顶点先登记（去重）
        vid(q)

    def dist(i):
        return float((np.asarray(V[i]) - p) @ n)

    d = [dist(i) for i in range(len(V))]
    newF = []
    cap_pts = []
    for face in F:
        inside = []                  # 本面内侧子多边形（顶点 id）
        m = len(face)
        for k in range(m):
            i, j = face[k], face[(k + 1) % m]
            di, dj = d[i], d[j]
            if di <= eps:            # i 在内侧（含面上）
                inside.append(i)
            if (di > eps and dj < -eps) or (di < -eps and dj > eps):
                # 边跨界 → 插值交点
                t = di / (di - dj)
                q = np.asarray(V[i]) * (1.0 - t) + np.asarray(V[j]) * t
                ci = vid(q)
                inside.append(ci)
                cap_pts.append(ci)
        if len(inside) >= 3:
            newF.append(inside)
            # 平面恰好过既有顶点（退化接触）时无跨界边，但保留面边界上的
            # 贴面顶点同样是截面多边形顶点 —— 不收集则盖帽缺失、体不闭合
            for i in inside:
                if i < len(V) and abs(d[i]) <= eps:
                    cap_pts.append(i)
    if not newF:
        return [], []
    if len(cap_pts) >= 3:
        # 盖帽：交点共面，投影到平面基 (e1,e2)，绕质心极角升序 = 绕 +n 逆时针
        pts = np.asarray([newV[i] for i in cap_pts], float)
        c = pts.mean(axis=0)
        a = np.zeros(3)
        a[int(np.argmin(np.abs(n)))] = 1.0
        e1 = np.cross(n, a)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-300)
        e2 = np.cross(n, e1)
        rel = pts - c
        ang = np.arctan2(rel @ e2, rel @ e1)
        order = [cap_pts[int(k)] for k in np.argsort(ang)]
        # 去重相邻重复（交点可能被两条面边各计一次）
        dedup = []
        for i in order:
            if not dedup or i != dedup[-1]:
                dedup.append(i)
        if len(dedup) >= 3 and dedup[0] == dedup[-1]:
            dedup = dedup[:-1]
        if len(dedup) >= 3:
            # 平面与某既有面重合（面贴合裁剪）：该面本身已在 newF，
            # 再加盖帽会重复计该面贡献（体积翻倍计）→ 跳过
            cap_set = frozenset(dedup)
            if not any(frozenset(f) == cap_set for f in newF):
                newF.append(dedup)
    # 压缩：仅保留被引用顶点
    used = sorted({i for f in newF for i in f})
    remap = {old: new for new, old in enumerate(used)}
    V2 = [newV[i] for i in used]
    F2 = [[remap[i] for i in f] for f in newF]
    return V2, F2


def poly_volume(V, F):
    """凸多面体有向体积（散度定理 + 面扇形三角化）。外向面序 → 正值。"""
    if not F:
        return 0.0
    V = np.asarray(V, float)
    tot = 0.0
    for face in F:
        m = len(face)
        if m < 3:
            continue
        p0 = V[face[0]]
        for k in range(1, m - 1):
            a = V[face[k]]
            b = V[face[k + 1]]
            tot += float(p0 @ np.cross(a, b))       # det(p0,a,b)
    return tot / 6.0


def poly_face_count(F):
    return len([f for f in F if len(f) >= 3])


# --------------------------------------------------------------- Voronoi 对偶
def _tet_neighbors(tets):
    """tet 边邻接表：顶点 → 邻接顶点集合（Delaunay 邻居）。"""
    nbr = {}
    for t in tets:
        ids = [int(x) for x in t]
        for a in ids:
            s = nbr.setdefault(a, set())
            for b in ids:
                if b != a:
                    s.add(b)
    return nbr


def voronoi_dual(vertices, tets, clip_box=None):
    """tet 网格的 Voronoi 对偶 poly 网格。

    - vertices (N,3)：tet 顶点（= poly 单元种子点）；
    - tets (M,4)：四面体单元；
    - clip_box：包围盒 (lo,hi)，缺省取顶点包围盒。

    对每个种子 v：从 clip 盒出发，被全部 Delaunay 邻居 u 的中垂面
    {x: |x-v|=|x-u|}（保 v 侧）裁剪 → v 的（盒裁）Voronoi 单元。
    Delaunay tet 的边邻接恰为 Voronoi 邻接 → 内部单元精确、全体铺满盒。

    返回 {ok, cells:[{seed, vertices, faces, volume, n_faces}], n_cells,
    volume_total, quality}。
    """
    V = np.asarray(vertices, float)
    C = np.asarray(tets, np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or len(C) == 0:
        raise ValueError("voronoi_dual 需要非空 (vertices Nx3, tets Mx4)")
    if clip_box is None:
        clip_box = (V.min(axis=0), V.max(axis=0))
    lo, hi = (np.asarray(clip_box[0], float), np.asarray(clip_box[1], float))
    nbr = _tet_neighbors(C)
    cells = []
    for v in range(len(V)):
        boxV, boxF = box_polyhedron(lo, hi)
        for u in sorted(nbr.get(v, ())):
            m = 0.5 * (V[v] + V[u])               # 中垂面过点
            nvec = V[u] - V[v]                    # 法向指向 u（被裁侧）
            if float(nvec @ nvec) < 1e-300:
                continue
            boxV, boxF = clip_convex(boxV, boxF, m, nvec)
            if not boxF:
                break
        vol = poly_volume(boxV, boxF)
        if boxF and vol > 0.0:
            cells.append({"seed": int(v), "vertices": boxV,
                          "faces": boxF, "volume": float(vol),
                          "n_faces": poly_face_count(boxF)})
    total = float(sum(c["volume"] for c in cells))
    out = {"ok": len(cells) > 0, "cells": cells, "n_cells": len(cells),
           "volume_total": total, "quality": poly_quality(cells)}
    return out


def poly_quality(cells):
    """poly 单元质量统计：单元数/体积和/最小体积/平均面数。"""
    if not cells:
        return {"n_cells": 0, "volume_total": 0.0, "volume_min": 0.0,
                "faces_mean": 0.0, "ok": False}
    vols = [c["volume"] for c in cells]
    return {"n_cells": len(cells),
            "volume_total": float(sum(vols)),
            "volume_min": float(min(vols)),
            "faces_mean": float(np.mean([c["n_faces"] for c in cells])),
            "ok": min(vols) > 0.0}


def poly_mesh(vertices, faces, spacing=None, method=None):
    """水密表面 → tet（Delaunay）→ Voronoi 对偶 poly 网格（完整工作流）。

    返回 {ok, tet, poly, n_cells, volume_total, quality, method}；
    非水密输入由 tet_mesh 抛 ValueError。
    """
    from mesh_tet import tet_mesh
    out_tet = tet_mesh(vertices, faces, spacing=spacing, method=method)
    dual = voronoi_dual(out_tet["vertices"], out_tet["cells"])
    return {"ok": dual["ok"] and out_tet["ok"],
            "tet": {"n_cells": out_tet["n_cells"],
                    "method": out_tet["method"]},
            "poly": dual, "n_cells": dual["n_cells"],
            "volume_total": dual["volume_total"],
            "quality": dual["quality"],
            "method": "voronoi_dual(%s)" % out_tet["method"]}
