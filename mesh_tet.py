# -*- coding: utf-8 -*-
"""N 波 N3：四面体体积网格生成器（Delaunay / Gmsh 双路由）。

输入一张**水密**闭合三角表面 (vertices, faces)，输出四面体单元填满内部。

路由（客户端自动选可用库，两环境皆出实际四面体单元）：
  A) scipy.spatial.Delaunay（默认 Python 3.14 有 scipy）：表面顶点 + 内部填充点
     Delaunay → 全部保留（水密凸性至少覆盖内部）；
  B) Gmsh（conda occ 环境有 gmsh）：三角表面→面环→体，前沿/约束 Delaunay；
  C) 二者皆无 → 抛 ValueError（由上层门控，不在测试断言路径）。

纯 numpy 质量诊断 `tet_quality` 两环境可用：逐四面体体积/负体积/边长纵横比。
"""
import numpy as np


def _has(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _watertight_ok(vertices, faces):
    from occ_repair import boundary_edges
    return len(boundary_edges(vertices, faces)[0]) == 0


# ------------------------------------------------------------ 点在水密体内
def point_in_mesh(p, vertices, faces, eps=1e-9):
    """even-odd 射线法（沿 +x）：点 p 是否在闭合三角网格内部（含表面）。

    表面判定对**所有**三角前置（任意朝向一致），且要求真实贴近三角形
    （平面距离 + 重心坐标双重检查）——只查重心会把面外侧近处点误判在面上。
    命中按射线参数 t 去重：恰好打在共享棱/顶点上时相邻三角会同点重复
    计数，直接数会把奇偶翻错（轴对齐盒的对角棱是系统性命中）。
    """
    p = np.asarray(p, float)
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    for a, b, c in F:
        if _pt_on_tri(p, V[a], V[b], V[c], eps):
            return True
    D = np.array([1.0, 0.0, 0.0])          # +x 射线
    ts = []
    for a, b, c in F:
        A, B, C = V[a], V[b], V[c]
        E1 = B - A
        E2 = C - A
        P = np.cross(D, E2)
        det = float(E1 @ P)
        if abs(det) < 1e-300:
            continue                        # 平行/共面且不在面上 → 跳过
        inv = 1.0 / det
        T = p - A
        u = float(T @ P) * inv
        if not (-eps <= u <= 1 + eps):
            continue
        Q = np.cross(T, E1)
        v = float(D @ Q) * inv
        if not (-eps <= v <= 1 + eps) or not (-eps <= u + v <= 1 + eps):
            continue
        t = float(E2 @ Q) * inv
        if t > 1e-9:
            ts.append(t)
    if not ts:
        return False
    ts.sort()
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = eps * max(1.0, diag)              # 同点/近棱双计的合并容差
    hits = 0
    prev = None
    for t in ts:
        if prev is None or t - prev > tol:
            hits += 1                       # 新的一簇命中（去重后计 1）
        prev = t
    return (hits % 2) == 1


def _pt_on_tri(p, A, B, C, eps):
    v0 = C - A
    v1 = B - A
    v2 = p - A
    n = np.cross(v0, v1)
    ln = float(np.linalg.norm(n))
    if ln < 1e-300:
        return False
    if abs(float(n @ v2)) > eps * ln:       # 离三角形平面太远（相对面积尺度）
        return False
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-300:
        return False
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    return (-eps <= u and -eps <= v and -eps <= w
            and u + v + w <= 1 + eps)


def _interior_fill_points(V, F, spacing):
    """水密体内均匀栅格填充点（even-odd 判定）。"""
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    pts = []
    nx = max(int((hi[0] - lo[0]) / spacing), 1)
    ny = max(int((hi[1] - lo[1]) / spacing), 1)
    nz = max(int((hi[2] - lo[2]) / spacing), 1)
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                p = lo + spacing * np.array([i, j, k])
                if point_in_mesh(p, V, F):
                    pts.append(p)
    if not pts:
        pts.append(V.mean(axis=0))
    return pts


# ------------------------------------------------------------ 四面体生成
def tet_mesh(vertices, faces, spacing=None, method=None):
    """从水密表面生成四面体体网格。

    返回 {ok, vertices[N,3], cells[M,4], n_cells, method, quality}，
    或对非闭合输入抛 ValueError。
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    if not _watertight_ok(V, F):
        raise ValueError("体网格需要水密闭合表面（当前有开放边界）")
    if spacing is None:
        spacing = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) * 0.2
    spacing = max(spacing, 1e-6)

    if method is None:
        method = "scipy" if _has("scipy") else ("gmsh" if _has("gmsh") else None)
    if method == "scipy":
        return _tet_scipy(V, F, spacing)
    if method == "gmsh":
        return _tet_gmsh(V, F, spacing)
    raise ValueError("当前 Python 无 scipy/gmsh，无法生成体网格")


def _tet_scipy(V, F, spacing):
    from scipy.spatial import Delaunay
    interior = _interior_fill_points(V, F, spacing)
    pts = np.vstack([V, np.asarray(interior, float)])
    tri = Delaunay(pts)
    cells = np.asarray(tri.simplices, np.int64)
    vol = _tet_volumes(pts, cells)
    keep = vol > 1e-12
    cells = cells[keep]
    return {"ok": len(cells) > 0, "vertices": pts, "cells": cells,
            "n_cells": int(len(cells)), "method": "scipy.delaunay",
            "n_vertices": len(pts), "quality": tet_quality(pts, cells)}


def _tet_gmsh(V, F, spacing):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", spacing * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", spacing)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay 3D
    m = gmsh.model
    m.add("tet")
    geo = gmsh.model.geo
    vmap = {}
    edge_line = {}          # (lo,hi) -> line id（共享边去重）
    tri_surfs = []
    for t in F:
        ids = []
        for p in V[t]:
            k = (round(float(p[0]), 9), round(float(p[1]), 9),
                 round(float(p[2]), 9))
            if k not in vmap:
                vmap[k] = geo.addPoint(float(p[0]), float(p[1]), float(p[2]))
            ids.append(vmap[k])
        lines = []
        for ia, ib in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            key = (ia, ib) if ia < ib else (ib, ia)
            if key not in edge_line:
                edge_line[key] = geo.addLine(key[0], key[1])
            lines.append(edge_line[key])
        loop = geo.addCurveLoop(lines, reorient=True)
        s = geo.addSurfaceFilling([loop])
        tri_surfs.append(s)
    geo.synchronize()
    sloop = m.geo.addSurfaceLoop(tri_surfs)
    m.geo.addVolume([sloop])
    m.geo.synchronize()
    m.mesh.generate(3)
    node_tags, coords, _ = m.mesh.getNodes()
    nid = {int(t): i for i, t in enumerate(node_tags)}
    pts = np.array(coords, float).reshape(-1, 3)
    cells = []
    ev_tags, ev_nodes = m.mesh.getElementsByType(4)   # 4 = tetra
    if len(ev_nodes):
        for i in range(0, len(ev_nodes), 4):
            cells.append([nid[int(ev_nodes[i + k])] for k in range(4)])
    gmsh.finalize()
    if not cells:
        raise ValueError("Gmsh 未生成四面体")
    cells = np.asarray(cells, np.int64)
    # 裁剪未索引节点（getNodes 全量）
    used = np.unique(cells.ravel())
    remap = {int(j): i for i, j in enumerate(used)}
    pts = pts[used]
    cells = np.array([[remap[int(x)] for x in row] for row in cells], np.int64)
    return {"ok": True, "vertices": pts, "cells": cells,
            "n_cells": int(len(cells)), "method": "gmsh",
            "n_vertices": len(pts), "quality": tet_quality(pts, cells)}


def _tet_volumes(V, cells):
    a = V[cells[:, 0]]
    b = V[cells[:, 1]]
    c = V[cells[:, 2]]
    d = V[cells[:, 3]]
    e1 = b - a
    e2 = c - a
    e3 = d - a
    return (e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
            - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
            + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])) / 6.0


def tet_quality(vertices, cells):
    """四面体单元质量统计（纯 numpy）。"""
    V = np.asarray(vertices, float)
    C = np.asarray(cells, np.int64)
    if len(C) == 0:
        return {"n_cells": 0, "n_negative": 0, "n_null": 0,
                "ok": True, "volume": {"mean": 0.0, "min": 0.0},
                "edge_aspect": {"mean": 0.0, "max": 0.0}}
    vol = _tet_volumes(V, C)
    e0 = np.linalg.norm(V[C[:, 1]] - V[C[:, 0]], axis=1)
    e1 = np.linalg.norm(V[C[:, 2]] - V[C[:, 1]], axis=1)
    e2 = np.linalg.norm(V[C[:, 0]] - V[C[:, 2]], axis=1)
    e3 = np.linalg.norm(V[C[:, 3]] - V[C[:, 0]], axis=1)
    e4 = np.linalg.norm(V[C[:, 3]] - V[C[:, 1]], axis=1)
    e5 = np.linalg.norm(V[C[:, 3]] - V[C[:, 2]], axis=1)
    emin = np.minimum.reduce([e0, e1, e2, e3, e4, e5])
    emax = np.maximum.reduce([e0, e1, e2, e3, e4, e5])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = emax / np.maximum(emin, 1e-300)
    n_neg = int((vol <= 1e-12).sum())
    return {"n_cells": int(len(C)), "n_negative": n_neg,
            "n_null": int((vol <= 1e-12).sum()),
            "volume": {"mean": float(np.abs(vol).mean()),
                       "min": float(vol.min())},
            "edge_aspect": {"mean": float(rel.mean()), "max": float(rel.max())},
            "ok": n_neg == 0 and len(C) > 0}