# -*- coding: utf-8 -*-
"""C 波 C4：表面修复工具集（hole fill / coarse / fine / quality metrics）。

作用于三角面网格 (vertices, faces)——与 `mesh_io` 的 STL/OBJ 载荷一致，
也与 `occ_bridge.tessellate` 的产物一致。

能力：
  - 孔洞：`boundary_loops` 探测开放边界环 + `fill_holes` 以 ear-clipping
    三角化补洞（纯 numpy，任意 Python 环境可用）；
  - 粗细网格：`surface_mesh(shape, mode='fine'|'coarse'|deflection)`
    （OCC B-Rep 细分密度控制，需 occ 环境）；
  - 质量：`quality_metrics(vertices, faces)` 面积/边长/纵横比/内角/退化统计；
  - 一条龙：`repair_surface` 读 STL/OBJ → 补洞 → 质量报告。

非 OCC 环境：hole fill / quality 照常工作；surface_mesh 抛 `OccUnavailable`。
"""
import numpy as np

from occ_bridge import OccUnavailable, has_occ

_HAS_OCC = has_occ()


# ---------------------------------------------------------------------------
# 孔洞检测
# ---------------------------------------------------------------------------
def boundary_edges(vertices, faces):
    """返回 (boundary_edges, edge_counts)。

    edge 用 (min_i, max_i) 归一化；boundary 边出现次数==1。
    """
    counts = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (a, b) if a < b else (b, a)
            counts[e] = counts.get(e, 0) + 1
    boundary = [(a, b) for (a, b), n in counts.items() if n == 1]
    return boundary, counts


def boundary_loops(vertices, faces):
    """将边界边串成有序环列表，每环为顶点索引序列（首尾相接）。"""
    boundary, _ = boundary_edges(vertices, faces)
    if not boundary:
        return []
    adj = {}
    for a, b in boundary:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    used_edges = set()
    loops = []
    for a0, b0 in boundary:
        fs = frozenset((a0, b0))
        if fs in used_edges:
            continue
        loop = [a0]
        cur, nxt = a0, b0
        while True:
            used_edges.add(frozenset((cur, nxt)))
            if nxt == loop[0]:
                break
            loop.append(nxt)
            nxt2 = None
            for c in adj.get(nxt, ()):
                if c != cur and frozenset((nxt, c)) not in used_edges:
                    nxt2 = c
                    break
            if nxt2 is None:
                break
            cur, nxt = nxt, nxt2
        loops.append(loop)
    return loops


# ---------------------------------------------------------------------------
# 补洞（ear-clipping）
# ---------------------------------------------------------------------------
def _signed_area(pts):
    return 0.5 * float(sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                           - pts[i][1] * pts[(i + 1) % len(pts)][0]
                           for i in range(len(pts))))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _in_tri(p, a, b, c):
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def _triangulate_loop(p2, idx):
    """ear-clipping；p2 为 XY 坐标数组，idx 为对应原始索引列表。

    返回原始索引三元组列表 (i,j,k)。ccw 规整后逐耳裁剪，兜底扇形。
    """
    pts = [tuple(p2[i]) for i in idx]
    if _signed_area(pts) < 0:
        idx = list(reversed(idx))
        pts.reverse()
    tri = []
    guard = 0
    while len(idx) > 3 and guard < 10 * len(idx) + 20:
        clipped = False
        n = len(idx)
        for i in range(n):
            a = pts[i - 1]
            b = pts[i]
            c = pts[(i + 1) % n]
            if _cross(a, b, c) <= 0:
                continue
            inside = any(_in_tri(p, a, b, c) for j, p in enumerate(pts)
                         if j != (i - 1) % n and j != i and j != (i + 1) % n)
            if inside:
                continue
            tri.append((idx[i - 1], idx[i], idx[(i + 1) % n]))
            idx.pop(i)
            pts.pop(i)
            clipped = True
            break
        if not clipped:            # 兜底：扇形
            for i in range(1, len(idx) - 1):
                tri.append((idx[0], idx[i], idx[i + 1]))
            break
        guard += 1
    if len(idx) == 3:
        tri.append(tuple(idx))
    return tri


def fill_holes(vertices, faces, min_loop=3):
    """三角形网格补洞：检测开放边界环，对每环 ear-clipping 三角化补上。

    返回 (new_vertices, new_faces)。补环不引入新顶点（顶点表不变）。
    """
    V = np.asarray(vertices, dtype=float)
    loops = boundary_loops(V, faces)
    new_faces = [list(map(int, t)) for t in faces]
    added = 0
    for loop in loops:
        if len(loop) < min_loop:
            continue
        pts = V[loop]
        cen = pts.mean(axis=0)
        A = pts - cen
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        u, v = vt[0], vt[1]
        p2 = np.column_stack([A @ u, A @ v])
        for (i0, i1, i2) in _triangulate_loop(p2, list(range(len(loop)))):
            new_faces.append([loop[i0], loop[i1], loop[i2]])
            added += 1
    return [list(map(float, r)) for r in V], new_faces


# ---------------------------------------------------------------------------
# 粗细网格（OCC 细分密度控制）
# ---------------------------------------------------------------------------
def surface_mesh(shape, mode="fine", deflection=None, angular=None, occ=None):
    """B-Rep → (vertices, faces, meta)。mode=fine 细/coarse 粗。

    粗细同时控制弦长偏差 deflection 与角公差 angular（越粗两者越大）。
    也可直接给 deflection/angular（弧度）。仅 OCC 环境可用。
    返回 {vertices, faces, n_vertices, n_triangles, deflection, angular, mode}。
    """
    if not _HAS_OCC:
        raise OccUnavailable("surface_mesh 需 conda 环境 occ（pythonocc-core）")
    from occ_bridge import tessellate, shape_counts
    if deflection is None:
        deflection = 0.05 if mode == "fine" else (0.5 if mode == "coarse" else 0.1)
    if angular is None:
        angular = 0.15 if mode == "fine" else (0.5 if mode == "coarse" else 0.3)
    verts, faces = tessellate(shape, deflection, angular=angular, occ=occ)
    return {"vertices": verts, "faces": faces,
            "n_vertices": len(verts), "n_triangles": len(faces),
            "deflection": deflection, "angular": angular, "mode": mode,
            "counts": shape_counts(shape, occ=occ)}


# ---------------------------------------------------------------------------
# 网格质量指标
# ---------------------------------------------------------------------------
def quality_metrics(vertices, faces):
    """三角形网格质量统计（纯 numpy）。

    返回 {n_triangles, n_vertices, area:min/max/mean, edge_len:min/max/mean,
          aspect:min/mean/max, min_angle_deg, max_angle_deg,
          n_degenerate, n_skinny(<10°), closed(bool)}。
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    P = V[F]                                   # [M,3,3]
    e0 = np.linalg.norm(P[:, 1] - P[:, 0], axis=1)
    e1 = np.linalg.norm(P[:, 2] - P[:, 1], axis=1)
    e2 = np.linalg.norm(P[:, 0] - P[:, 2], axis=1)
    cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = 0.5 * np.sqrt((cr * cr).sum(axis=1))
    s = 0.5 * (e0 + e1 + e2)
    with np.errstate(divide="ignore", invalid="ignore"):
        inrad = np.where(area > 1e-15, area / np.maximum(s, 1e-300), 0.0)
        aspect = np.where(inrad > 0, np.maximum.reduce([e0, e1, e2]) / (2.0 * inrad), 0.0)

    def _ang(p1, p2, p3):
        d1 = p1 - p2
        d2 = p3 - p2
        c = (d1 * d2).sum(axis=1) / np.maximum(
            np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1), 1e-300)
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    a0 = _ang(P[:, 0], P[:, 1], P[:, 2])
    a1 = _ang(P[:, 1], P[:, 2], P[:, 0])
    a2 = _ang(P[:, 2], P[:, 0], P[:, 1])
    min_ang = np.minimum.reduce([a0, a1, a2])
    max_ang = np.maximum.reduce([a0, a1, a2])

    deg = (np.count_nonzero(area <= 1e-15),
           np.count_nonzero(min_ang < 10.0))
    _, counts = boundary_edges(vertices, faces)
    return {
        "n_triangles": len(F), "n_vertices": len(V),
        "closed": len([e for e, n in counts.items() if n == 1]) == 0,
        "area": {"min": float(area.min()) if len(area) else 0.0,
                 "max": float(area.max()) if len(area) else 0.0,
                 "mean": float(area.mean()) if len(area) else 0.0},
        "edge_len": {"min": float(np.minimum.reduce([e0, e1, e2]).min()) if len(e0) else 0.0,
                     "max": float(np.maximum.reduce([e0, e1, e2]).max()) if len(e0) else 0.0,
                     "mean": float(np.maximum.reduce([e0, e1, e2]).mean()) if len(e0) else 0.0},
        "aspect": {"min": float(aspect.min()) if len(aspect) else 0.0,
                   "mean": float(aspect.mean()) if len(aspect) else 0.0,
                   "max": float(aspect.max()) if len(aspect) else 0.0},
        "min_angle_deg": float(min_ang.min()) if len(min_ang) else 0.0,
        "max_angle_deg": float(max_ang.max()) if len(max_ang) else 0.0,
        "n_degenerate": int(deg[0]), "n_skinny": int(deg[1]),
    }


def repair_surface(source, min_loop=3, mode="fine", deflection=None):
    """一条龙：读 STL/OBJ 或 B-Rep 文件 → 补洞 → 质量报告。

    source 为 .stl/.obj（mesh_io 读网格后补洞）或 B-Rep 后缀（OCC 细分后
    补洞）。返回 {ok, counts, metrics, filled_holes}；失败带 reason 且 ok=False。
    """
    import os
    ext = os.path.splitext(str(source))[1].lower()
    if ext in (".stl", ".obj"):
        from mesh_io import read_surface
        verts, faces = read_surface(source)
    elif ext in (".step", ".stp", ".igs", ".iges", ".brep"):
        if not _HAS_OCC:
            return {"ok": False, "reason": "B-Rep 需 OCC 环境（去 .obj/.stl 用纯 numpy）"}
        from occ_bridge import import_shape
        m = surface_mesh(import_shape(source), mode=mode, deflection=deflection)
        verts, faces = m["vertices"], m["faces"]
    else:
        return {"ok": False, "reason": "不支持的输入: %s" % ext}
    before = len(boundary_loops(verts, faces))
    verts2, faces2 = fill_holes(verts, faces, min_loop=min_loop)
    after = len(boundary_loops(verts2, faces2))
    return {"ok": True, "vertices": verts2, "faces": faces2,
            "counts": {"n_vertices": len(verts2), "n_triangles": len(faces2)},
            "metrics": quality_metrics(verts2, faces2),
            "holes_before": before, "holes_after": after}