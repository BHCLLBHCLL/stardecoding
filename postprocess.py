# -*- coding: utf-8 -*-
"""V 波：后处理深化内核（纯 numpy + 标准库，occ / scdm 两环境皆可用）。

面向 P4 `FVM` 四面体网格 + 单元中心解场，提供 V1–V6 全谱系后处理原语：

  V1 标量/矢量 color-by —— 官方色表断点解析 / 采样 / 标量与矢量场取色
     （`parse_colormap`/`sample_colormap`/`map_scalars`/`color_by`/`vector_magnitude`）。
  V2 显示器几何 —— 流线/迹线/质点 RK 积分（`streamline`/`pathline`/`particle_trace`）、
     等值面 marching tetrahedra（`marching_tets`）、切片/裁剪平面（`section_plane`/
     `clip_plane`）、阈值面（`threshold_surface`）、镜像（`mirror_geometry`）、
     矢量符号（`vector_glyphs`）。
  V3 派生零件 —— 探针重心采样（`probe`）/线采样（`line_sample`）/面采样
     （`plane_sample`）/等值体（`iso_volume`）/阈值零件（`threshold_part`）/
     平面-包围盒多边形（`plane_box_polygon`）/派生缓存（`DerivedCache`）。
  V4 绘图数据 —— XY 序列（`xy_series`）/直方图（`histogram`）/累积分布
     （`cumulative_distribution`）/降采样（`decimate_series`）/实时监视缓冲
     （`MonitorBuffer`）。
  V5 注记/图例/色标尺 + 动画 —— 色标刻度（`colorbar_ticks`）/色标条
     （`colorbar_strip`）/图例项（`legend_items`）/注记（`annotation`）/
     帧序列（`frame_times`/`frame_indices`/`frame_name`）/硬拷贝与动画写出
     （`write_ppm`/`write_png`/`write_gif`/`export_animation`）。
  V6 数据写出 —— CSV（`write_csv`/`export_csv`）、EnSight Gold ASCII
     （`write_ensight`：.case/.geo/.dat）、CGNS/HDF5 子集（`write_cgns`）。

另提供 P10 风格门面 `PostProcessor`（字段存取 + 各后处理操作 + `monitor_payload`）
与工厂 `make_postprocessor`（别名不敏感），并从求解器后端抽取解场
`fields_from_solver`。

**诚实边界**：图像动画 mp4 需外部 ffmpeg，缺失时 `export_animation` 如实降级
（返回已写帧、`ok=False` 并注明）；CGNS 写为基于 h5py 的 CGNS/HDF5 命名子集，
无 h5py 时诚实拒绝（`ok=False`）；流线/面采样依赖四面体重心定位，域外点返回 NaN。

纯 numpy 约束：不新增 scipy 依赖；范数统一走 `solver_run._safe_norm`。
"""
import math
import os
import struct
import zlib

import numpy as np

from solver_run import _safe_norm

EPS = 1.0e-12
LOC_TOL = 1.0e-7
DEFAULT_BINS = 20
DEFAULT_MAX_PTS = 512
DEFAULT_CMAP_SAMPLES = 256

# 官方 PredefinedLookupTable 默认蓝→黄→红（与 star_gui_vtk.lut_from_colormap 同族）
DEFAULT_COLORMAP_VALUES = [
    0.000, 0.0, 0.0, 1.0,
    0.125, 0.0, 0.5, 1.0,
    0.250, 0.0, 1.0, 1.0,
    0.375, 0.0, 1.0, 0.5,
    0.500, 1.0, 1.0, 0.0,
    0.625, 1.0, 0.75, 0.0,
    0.750, 1.0, 0.5, 0.0,
    0.875, 1.0, 0.25, 0.0,
    1.000, 1.0, 0.0, 0.0,
]


# ===========================================================================
# V1 标量/矢量 color-by
# ===========================================================================
def parse_colormap(values, alphas=None):
    """官方色表断点（4n 组 位置,R,G,B）→ (pos[n], rgb[n,3], alpha[n])。

    位置降序自动翻转；断点不足 2 组或长度非法返回 (None, None, None)。
    AlphaValues 仅在与断点等长时逐断点生效，否则全不透明。
    """
    vals = [float(x) for x in (values or [])]
    if len(vals) < 8 or len(vals) % 4:
        return None, None, None
    n = len(vals) // 4
    pos = np.asarray(vals[0::4], float)
    rgb = np.asarray([[vals[4 * i + 1], vals[4 * i + 2], vals[4 * i + 3]]
                      for i in range(n)], float)
    if pos[0] > pos[-1]:
        pos = pos[::-1].copy()
        rgb = rgb[::-1].copy()
    al = [float(x) for x in (alphas or [])]
    alpha = np.asarray(al, float) if len(al) == n else np.ones(n, float)
    return pos, rgb, alpha


def sample_colormap(values=None, alphas=None, n=DEFAULT_CMAP_SAMPLES,
                    lo=0.0, hi=1.0):
    """色表断点按位置线性插值重采样为 n 级 RGBA 表 (n,4)。

    `values` 为 None 时用官方默认蓝→黄→红。位置断点非均匀（黄≈0.5），
    重采样后供逐点映射使用。
    """
    if values is None:
        values = DEFAULT_COLORMAP_VALUES
    pos, rgb, alpha = parse_colormap(values, alphas)
    if pos is None:
        return np.zeros((0, 4), float)
    m = max(2, int(n))
    t = np.linspace(0.0, 1.0, m)
    out = np.zeros((m, 4), float)
    for k in range(3):
        out[:, k] = np.interp(t, pos, rgb[:, k])
    out[:, 3] = np.interp(t, pos, alpha)
    out[:, 0:3] = np.clip(out[:, 0:3], 0.0, 1.0)
    out[:, 3] = np.clip(out[:, 3], 0.0, 1.0)
    return out


def scalar_range(values, robust=False):
    """标量场范围 (lo, hi)，忽略 NaN/Inf；空场返回 (0.0, 1.0)。"""
    arr = np.asarray(values, float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _normalize(values, lo, hi, clamp=True):
    """标量线性归一化到 [0,1]（hi<=lo 时退化返回 0）。"""
    arr = np.asarray(values, float)
    if hi <= lo:
        return np.zeros_like(arr)
    t = (arr - lo) / (hi - lo)
    return np.clip(t, 0.0, 1.0) if clamp else t


def map_scalars(values, cmap=None, lo=None, hi=None, clamp=True):
    """标量场 → RGBA (N,4)。cmap 为 (m,4) 色表，为 None 时用默认色表。"""
    if cmap is None:
        cmap = sample_colormap()
    cmap = np.asarray(cmap, float)
    if cmap.ndim != 2 or cmap.shape[0] < 2:
        return np.zeros((0, 4), float)
    arr = np.asarray(values, float).reshape(-1)
    if lo is None or hi is None:
        lo2, hi2 = scalar_range(arr)
        lo = lo2 if lo is None else lo
        hi = hi2 if hi is None else hi
    t = _normalize(arr, lo, hi, clamp=clamp)
    pos = t * (cmap.shape[0] - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, cmap.shape[0] - 2)
    f = (pos - i0)[:, None]
    return cmap[i0] * (1.0 - f) + cmap[i0 + 1] * f


def vector_magnitude(vecs):
    """矢量场 (N,3) → 幅值 (N,)；已是 1D 则原样返回。"""
    arr = np.asarray(vecs, float)
    if arr.ndim == 1:
        return arr
    return _safe_norm(arr, axis=1)


def color_by(field, kind="scalar", comp=None, cmap=None, lo=None, hi=None):
    """解场 color-by：返回 {"rgba":(N,4), "scalars":(N,), "range":(lo,hi)}。

    kind=scalar 直接用 field；magnitude 取矢量幅值；component 取分量
    （comp 为 0/1/2 或 "x"/"y"/"z"）。
    """
    arr = np.asarray(field, float)
    if kind in ("magnitude", "mag"):
        scalars = vector_magnitude(arr)
    elif kind in ("component", "comp"):
        if arr.ndim != 2:
            raise ValueError("component color-by 需要 (N,3) 矢量场")
        c = {"x": 0, "y": 1, "z": 2}.get(comp, comp)
        c = 0 if c is None else int(c)
        if c < 0 or c >= arr.shape[1]:
            raise ValueError("矢量分量越界: %r" % (comp,))
        scalars = arr[:, c]
    else:
        scalars = arr.reshape(-1) if arr.ndim == 1 else vector_magnitude(arr)
    lo2, hi2 = scalar_range(scalars)
    lo = lo2 if lo is None else lo
    hi = hi2 if hi is None else hi
    rgba = map_scalars(scalars, cmap=cmap, lo=lo, hi=hi)
    return {"rgba": rgba, "scalars": np.asarray(scalars, float),
            "range": (float(lo), float(hi))}


# ===========================================================================
# 几何核心：四面体重心定位 / 场插值
# ===========================================================================
def _sixvol(a, b, c, d):
    """带号体积的 6 倍 det(b-a, c-a, d-a)，支持广播。"""
    e1 = b - a
    e2 = c - a
    e3 = d - a
    return (e1[..., 0] * (e2[..., 1] * e3[..., 2] - e2[..., 2] * e3[..., 1])
            - e1[..., 1] * (e2[..., 0] * e3[..., 2] - e2[..., 2] * e3[..., 0])
            + e1[..., 2] * (e2[..., 0] * e3[..., 1] - e2[..., 1] * e3[..., 0]))


def locate_cell(fv, point, tol=LOC_TOL):
    """定位点所在四面体，返回 (cell, weights[4])；域外返回 (-1, None)。"""
    V = fv.vertices
    C = fv.cells
    A = V[C[:, 0]]
    B = V[C[:, 1]]
    Cc = V[C[:, 2]]
    Dd = V[C[:, 3]]
    ref = _sixvol(A, B, Cc, Dd)
    safe = np.where(np.abs(ref) > 1e-14, ref, 1.0)
    p = np.asarray(point, float)
    la = _sixvol(p, B, Cc, Dd) / safe
    lb = _sixvol(A, p, Cc, Dd) / safe
    lc = _sixvol(A, B, p, Dd) / safe
    ld = _sixvol(A, B, Cc, p) / safe
    ok = (la >= -tol) & (lb >= -tol) & (lc >= -tol) & (ld >= -tol)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return -1, None
    k = int(idx[0])
    return k, np.array([la[k], lb[k], lc[k], ld[k]], float)


def sample_scalar(fv, field, points):
    """标量场在任意点集的插值（逐点重心线性），域外为 NaN。

    field 若为逐顶点数组直接插值；若长度等于单元数则先 `cell_to_vertex`。
    """
    field = np.asarray(field, float).reshape(-1)
    if field.size == fv.n_cells and field.size != fv.n_vertices:
        field = cell_to_vertex(fv, field)
    pts = np.atleast_2d(np.asarray(points, float))
    out = np.full(len(pts), np.nan, float)
    for i, p in enumerate(pts):
        c, w = locate_cell(fv, p)
        if c >= 0:
            out[i] = float(np.dot(w, field[fv.cells[c]]))
    return out


def sample_vector(fv, vec, points):
    """矢量场 (M,3) 在任意点集的插值，返回 (N,3)，域外为 NaN。

    vec 若为逐顶点数组直接插值；若长度等于单元数则先逐分量 `cell_to_vertex`。
    """
    vec = np.asarray(vec, float)
    if vec.shape[0] == fv.n_cells and vec.shape[0] != fv.n_vertices:
        vec = np.column_stack([cell_to_vertex(fv, vec[:, d])
                               for d in range(vec.shape[1])])
    pts = np.atleast_2d(np.asarray(points, float))
    out = np.full((len(pts), 3), np.nan, float)
    for i, p in enumerate(pts):
        c, w = locate_cell(fv, p)
        if c >= 0:
            out[i] = w @ vec[fv.cells[c]]
    return out


# ===========================================================================
# V2 显示器几何
# ===========================================================================
def _empty_surface():
    return {"vertices": np.zeros((0, 3), float),
            "triangles": np.zeros((0, 3), np.int64),
            "scalars": {}}


def _dedup_points(points, tol=1e-9):
    """点集去重，返回 (unique[N,3], inverse[M])。

    inverse[i] 为原始点 points[i] 在 unique 中的行索引，长度等于输入点数，
    可配合 reshape(-1, 3) 还原三角形索引。
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3), float), np.zeros(0, np.int64)
    key = np.round(pts / tol).astype(np.int64)
    _, first_idx, inverse = np.unique(
        key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1).astype(np.int64)
    return pts[first_idx], inverse


def _order_polygon(pts, normal):
    """共面多边形绕质心按平面内极角排序（右手绕 +normal）。"""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    n = np.asarray(normal, float)
    nn = _safe_norm(n)
    if nn < EPS:
        return pts
    n = n / nn
    c = pts.mean(axis=0)
    a = np.zeros(3)
    a[int(np.argmin(np.abs(n)))] = 1.0
    e1 = np.cross(n, a)
    e1 = e1 / max(_safe_norm(e1), EPS)
    e2 = np.cross(n, e1)
    rel = pts - c
    ang = np.arctan2(rel @ e2, rel @ e1)
    return pts[np.argsort(ang)]


def _velocity_at(fv, velocity, p):
    """速度场（(M,3) 数组或 callable）在点 p 的取值，域外 None。"""
    if callable(velocity):
        v = np.asarray(velocity(p), float)
        return None if v.shape != (3,) or not np.all(np.isfinite(v)) else v
    c, w = locate_cell(fv, p)
    if c < 0:
        return None
    v = w @ np.asarray(velocity, float)[fv.cells[c]]
    return v


def _integrate(fv, velocity, seed, step, max_steps, direction, bounds=None):
    """单条 RK4 流线积分，返回点列 (K,3)。"""
    p = np.asarray(seed, float)
    pts = [p.copy()]
    for _ in range(int(max_steps)):
        v1 = _velocity_at(fv, velocity, p)
        if v1 is None:
            break
        v2 = _velocity_at(fv, velocity, p + 0.5 * step * direction * v1)
        if v2 is None:
            break
        v3 = _velocity_at(fv, velocity, p + 0.5 * step * direction * v2)
        if v3 is None:
            break
        v4 = _velocity_at(fv, velocity, p + step * direction * v3)
        if v4 is None:
            break
        disp = step * direction * (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
        if _safe_norm(disp) < EPS:
            break
        p = p + disp
        if bounds is not None:
            lo, hi = bounds
            if np.any(p < lo) or np.any(p > hi):
                pts.append(p.copy())
                break
        pts.append(p.copy())
    return np.asarray(pts, float)


def streamline(fv, velocity, seed, step=None, max_steps=256, direction=1,
               bounds=None):
    """单条流线（稳态速度场 RK4），返回 {"vertices":(K,3)}。"""
    if step is None:
        step = 0.25 * float(np.mean(np.cbrt(fv.volumes)))
    pts = _integrate(fv, velocity, seed, step, max_steps, direction, bounds)
    return {"vertices": pts}


def streamlines(fv, velocity, seeds, step=None, max_steps=256,
                bidirectional=False, bounds=None):
    """多条流线，返回 {"lines":[{"vertices":(K,3)}, ...]}。

    bidirectional=True 时对每个种子同时正/反向积分并拼接（反向段逆序）。
    """
    lines = []
    for s in np.atleast_2d(np.asarray(seeds, float)):
        parts = []
        if bidirectional:
            back = _integrate(fv, velocity, s, step or 0.25
                              * float(np.mean(np.cbrt(fv.volumes))),
                              max_steps, -1, bounds)
            if len(back) > 1:
                parts.append(back[::-1][:-1])
        fwd = _integrate(fv, velocity, s,
                         step or 0.25 * float(np.mean(np.cbrt(fv.volumes))),
                         max_steps, 1, bounds)
        parts.append(fwd)
        lines.append({"vertices": np.concatenate(parts, axis=0)})
    return {"lines": lines}


def pathline(fv, velocities, seed, times=None, step=None, max_steps=256,
             direction=1):
    """迹线：非稳态速度快照序列上 RK4 积分（快照间线性时间插值）。

    velocities 为 [(M,3), ...] 快照列表；times 为对应时刻（缺省 0..n-1）。
    返回 {"vertices":(K,3)}。
    """
    snaps = [np.asarray(v, float) for v in velocities]
    if not snaps:
        return {"vertices": np.zeros((0, 3), float)}
    if times is None:
        times = list(range(len(snaps)))
    times = np.asarray(times, float)
    if step is None:
        step = 0.25 * float(np.mean(np.cbrt(fv.volumes)))

    def vel_at(p, t):
        if len(snaps) == 1 or t <= times[0]:
            return _velocity_at(fv, snaps[0], p)
        if t >= times[-1]:
            return _velocity_at(fv, snaps[-1], p)
        k = int(np.searchsorted(times, t) - 1)
        k = min(max(k, 0), len(snaps) - 2)
        span = times[k + 1] - times[k]
        f = 0.0 if span <= 0 else (t - times[k]) / span
        return _velocity_at(fv, snaps[k], p) * (1.0 - f) + \
            _velocity_at(fv, snaps[k + 1], p) * f

    p = np.asarray(seed, float)
    t = float(times[0])
    pts = [p.copy()]
    for _ in range(int(max_steps)):
        v1 = vel_at(p, t)
        if v1 is None:
            break
        v2 = vel_at(p + 0.5 * step * direction * v1, t)
        v3 = vel_at(p + 0.5 * step * direction * v2, t)
        v4 = vel_at(p + step * direction * v3, t)
        if v2 is None or v3 is None or v4 is None:
            break
        disp = step * direction * (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
        if _safe_norm(disp) < EPS:
            break
        p = p + disp
        t = t + step * direction
        pts.append(p.copy())
    return {"vertices": np.asarray(pts, float)}


def particle_trace(fv, velocity, seeds, n_steps=64, step=None):
    """质点轨迹（显式位置推进），返回 {"vertices":(N,3), "ids":(N,)}。"""
    if step is None:
        step = 0.25 * float(np.mean(np.cbrt(fv.volumes)))
    pts, ids = [], []
    seeds = np.atleast_2d(np.asarray(seeds, float))
    live = [s.copy() for s in seeds]
    for _ in range(int(n_steps)):
        nxt = []
        for i, p in enumerate(live):
            v = _velocity_at(fv, velocity, p)
            pts.append(p.copy())
            ids.append(i)
            nxt.append(None if v is None else p + step * v)
        live = [q if q is not None else live[i]
                for i, q in enumerate(nxt)]
    return {"vertices": np.asarray(pts, float),
            "ids": np.asarray(ids, np.int64)}


def _tet_scalar_grad(P, vals):
    """四面体线性标量梯度（显式伴随式，规避 np.linalg）。"""
    e1 = P[1] - P[0]
    e2 = P[2] - P[0]
    e3 = P[3] - P[0]
    det = float(e1 @ np.cross(e2, e3))
    if abs(det) < 1e-14:
        return None
    b1 = vals[1] - vals[0]
    b2 = vals[2] - vals[0]
    b3 = vals[3] - vals[0]
    g = (b1 * np.cross(e2, e3) + b2 * np.cross(e3, e1)
         + b3 * np.cross(e1, e2)) / det
    return g


def cell_to_vertex(fv, field, weight="volume"):
    """单元中心场 → 顶点场（体积加权或缺省均权），供 marching tets 使用。"""
    field = np.asarray(field, float).reshape(-1)
    nodal = np.zeros(fv.n_vertices, float)
    wsum = np.zeros(fv.n_vertices, float)
    if weight == "volume":
        w = np.asarray(fv.volumes, float)
    else:
        w = np.ones(fv.n_cells, float)
    wc = w[:, None] * np.ones((1, 4), float)
    vid = fv.cells.reshape(-1)
    np.add.at(nodal, vid, (wc * field[:, None]).reshape(-1))
    np.add.at(wsum, vid, wc.reshape(-1))
    good = wsum > 0.0
    nodal[good] /= wsum[good]
    return nodal


def _cut_tets_nodal(fv, nodal, iso, orient=True):
    """按逐顶点标量 nodal 切四面体，返回 (unique[N,3], tris[M,3])。

    法向统一指向标量增大方向（由四面体线性梯度定向）。
    """
    V = fv.vertices
    C = fv.cells
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    tris = []
    for k in range(len(C)):
        P = V[C[k]]
        v = nodal[C[k]]
        d = v - float(iso)
        if np.all(d >= 0.0) or np.all(d <= 0.0):
            continue
        g = _tet_scalar_grad(P, v) if orient else None
        pts = []
        for i, j in edges:
            if (d[i] > 0.0) == (d[j] > 0.0):
                continue
            t = float(d[i] / (d[i] - d[j]))
            pts.append(P[i] + t * (P[j] - P[i]))
        if len(pts) < 3:
            continue
        for a in range(1, len(pts) - 1):
            tri = np.asarray([pts[0], pts[a], pts[a + 1]], float)
            if (_safe_norm(tri[1] - tri[0]) < 1e-11
                    or _safe_norm(tri[2] - tri[0]) < 1e-11):
                continue
            if g is not None:
                nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                if float(nrm @ g) < 0.0:
                    tri = np.asarray([tri[0], tri[2], tri[1]], float)
            tris.append(tri)
    if not tris:
        return np.zeros((0, 3), float), np.zeros((0, 3), np.int64)
    uniq, inv = _dedup_points(np.vstack(tris))
    return uniq, inv.reshape(-1, 3)


def marching_tets(fv, scalar, iso, fields=None):
    """等值面（marching tetrahedra），返回三角面片表面。

    scalar 为**逐顶点**标量（单元中心场请先 `cell_to_vertex`）。逐四面体按
    4 顶点与等值线的关系切出 1 或 2 个三角形，顶点去重；法向统一指向标量
    增大方向（由四面体线性梯度定向）。fields 为 {name: 数组}，在等值面顶点
    处额外采样。
    """
    nodal = np.asarray(scalar, float).reshape(-1)
    verts, tris = _cut_tets_nodal(fv, nodal, float(iso))
    out = {"vertices": verts, "triangles": tris, "scalars": {},
           "iso": float(iso)}
    if len(verts) == 0:
        return out
    out["scalars"]["scalar"] = sample_scalar(fv, nodal, verts)
    for name, arr in (fields or {}).items():
        out["scalars"][name] = sample_scalar(fv, arr, verts)
    return out


# ===========================================================================
# V2 平面切片 / 裁剪（纯 numpy 凸多面体裁剪，不用 np.linalg）
# ===========================================================================
_TET_FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]


def _clip_convex(V, F, p, n, eps=1e-12):
    """凸多面体 (V,F) 被平面 {x:(x-p)·n=0} 裁剪，保留 (x-p)·n<=0 侧。

    Sutherland-Hodgman 三维版（纯 numpy，规避 np.linalg）；截面盖帽按平面极角
    排序。返回新 (V, F)；整块被裁掉返回 ([], [])。
    """
    V = [np.asarray(q, float) for q in V]
    n = np.asarray(n, float)
    p = np.asarray(p, float)
    nn = _safe_norm(n)
    if nn < 1e-300:
        return V, [list(f) for f in F]
    n = n / nn
    eye = {}
    newV = []

    def vid(q):
        key = (round(float(q[0]), 11), round(float(q[1]), 11),
               round(float(q[2]), 11))
        if key not in eye:
            eye[key] = len(newV)
            newV.append(np.asarray(q, float))
        return eye[key]

    for q in V:
        vid(q)
    d = [float((V[i] - p) @ n) for i in range(len(V))]
    newF, cap = [], []
    for face in F:
        m = len(face)
        inside = []
        for k in range(m):
            i, j = face[k], face[(k + 1) % m]
            di, dj = d[i], d[j]
            if di <= eps:
                inside.append(i)
            if (di > eps and dj < -eps) or (di < -eps and dj > eps):
                t = di / (di - dj)
                ci = vid(V[i] * (1.0 - t) + V[j] * t)
                inside.append(ci)
                cap.append(ci)
        if len(inside) >= 3:
            newF.append(inside)
            for i in inside:
                if i < len(V) and abs(d[i]) <= eps:
                    cap.append(i)
    if not newF:
        return [], []
    if len(cap) >= 3:
        pts = np.asarray([newV[i] for i in cap], float)
        c = pts.mean(axis=0)
        a = np.zeros(3)
        a[int(np.argmin(np.abs(n)))] = 1.0
        e1 = np.cross(n, a)
        e1 = e1 / max(_safe_norm(e1), 1e-300)
        e2 = np.cross(n, e1)
        ang = np.arctan2((pts - c) @ e2, (pts - c) @ e1)
        order = [cap[int(k)] for k in np.argsort(ang)]
        dedup = []
        for i in order:
            if not dedup or i != dedup[-1]:
                dedup.append(i)
        if len(dedup) >= 3 and dedup[0] == dedup[-1]:
            dedup = dedup[:-1]
        if len(dedup) >= 3:
            cs = frozenset(dedup)
            if not any(frozenset(f) == cs for f in newF):
                newF.append(dedup)
    used = sorted({i for f in newF for i in f})
    remap = {o: k for k, o in enumerate(used)}
    return ([newV[i] for i in used],
            [[remap[i] for i in f] for f in newF])


def _poly_triangles(V, F):
    """凸多面体面列表 → 三角形列表（扇形三角化 + 朝外定向）。"""
    V = np.asarray(V, float)
    if len(V) == 0:
        return []
    center = V.mean(axis=0)
    tris = []
    for face in F:
        m = len(face)
        for k in range(1, m - 1):
            a, b, c = V[face[0]], V[face[k]], V[face[k + 1]]
            nrm = np.cross(b - a, c - a)
            if _safe_norm(nrm) < 1e-13:
                continue
            if float(nrm @ (a - center)) < 0.0:
                a, c = c, a
            tris.append(np.asarray([a, b, c], float))
    return tris


def section_plane(fv, point, normal, fields=None):
    """平面切片：返回切面三角面片 + 顶点处场值。

    fields 为 {name: 数组（逐顶点或单元中心）}，在切面顶点采样。返回
    {"vertices","triangles","scalars"(dict),"normal","point"}。
    """
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    nn = _safe_norm(n)
    if nn < EPS:
        raise ValueError("section_plane 法向不能为零")
    n = n / nn
    dist = (fv.vertices - p) @ n
    verts, tris = _cut_tets_nodal(fv, dist, 0.0)
    if len(verts) == 0:
        out = _empty_surface()
        out.update({"normal": n, "point": p})
        return out
    out = {"vertices": verts, "triangles": tris, "scalars": {},
           "normal": n, "point": p}
    for name, arr in (fields or {}).items():
        out["scalars"][name] = sample_scalar(fv, arr, verts)
    return out


def clip_plane(fv, point, normal, keep="negative", fields=None):
    """平面裁剪：保留半空间内的网格部分，返回裁剪后**外表面**三角形。

    keep="negative" 保留 (x-p)·n<=0 侧，"positive" 保留 >=0 侧；fields 在裁剪
    后顶点采样。返回 {"vertices","triangles","scalars","normal","point","keep"}。
    相邻裁剪单元之间的内部共享面（出现两次者）会被剔除，仅保留域边界与切面
    盖帽，语义与 `extract_surface` 一致。
    """
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    nn = _safe_norm(n)
    if nn < EPS:
        raise ValueError("clip_plane 法向不能为零")
    n = n / nn
    keep_low = str(keep).lower() not in ("positive", "pos", "above", "+")
    if not keep_low:
        n = -n

    pts, pmap = [], {}

    def gid(q):
        key = (round(float(q[0]), 9), round(float(q[1]), 9),
               round(float(q[2]), 9))
        if key not in pmap:
            pmap[key] = len(pts)
            pts.append(np.asarray(q, float))
        return pmap[key]

    faces, centers = [], []
    for k in range(fv.n_cells):
        cv, cf = _clip_convex(fv.vertices[fv.cells[k]], _TET_FACES, p, n)
        if not cv:
            continue
        ci = len(centers)
        centers.append(np.asarray(cv, float).mean(axis=0))
        for face in cf:
            if len(face) < 3:
                continue
            faces.append(([gid(cv[i]) for i in face], ci))

    if not faces:
        out = _empty_surface()
        out.update({"normal": n, "point": p, "keep": keep})
        return out

    count = {}
    for gf, _ in faces:
        key = tuple(sorted(set(gf)))
        count[key] = count.get(key, 0) + 1

    tris = []
    for gf, ci in faces:
        if count.get(tuple(sorted(set(gf))), 0) != 1:
            continue
        ctr = centers[ci]
        for j in range(1, len(gf) - 1):
            a, b, c = pts[gf[0]], pts[gf[j]], pts[gf[j + 1]]
            nrm = np.cross(b - a, c - a)
            if _safe_norm(nrm) < 1e-13:
                continue
            if float(nrm @ (a - ctr)) < 0.0:
                a, c = c, a
            tris.append(np.asarray([a, b, c], float))

    if not tris:
        out = _empty_surface()
        out.update({"normal": n, "point": p, "keep": keep})
        return out
    uniq, inv = _dedup_points(np.vstack(tris))
    out = {"vertices": uniq, "triangles": inv.reshape(-1, 3), "scalars": {},
           "normal": n, "point": p, "keep": keep}
    for name, arr in (fields or {}).items():
        out["scalars"][name] = sample_scalar(fv, arr, uniq)
    return out


def extract_surface(fv, cells, fields=None):
    """给定单元子集，返回其外表面三角面片（剔除子集内部共享面）。"""
    keep = np.zeros(fv.n_cells, bool)
    keep[np.asarray(cells, np.int64)] = True
    tris = []
    for f in range(fv.n_faces):
        o = int(fv.owner[f])
        if o < 0 or not keep[o]:
            continue
        nb = int(fv.neighbor[f])
        if nb >= 0 and keep[nb]:
            continue
        tris.append(fv.vertices[fv.face_vertices[f]])
    if not tris:
        out = _empty_surface()
        out["cells"] = np.asarray(cells, np.int64)
        return out
    uniq, inv = _dedup_points(np.vstack(tris))
    out = {"vertices": uniq, "triangles": inv.reshape(-1, 3), "scalars": {}}
    for name, arr in (fields or {}).items():
        out["scalars"][name] = sample_scalar(fv, arr, uniq)
    return out


def threshold_cells(fv, field, lo=None, hi=None):
    """取值落在 [lo,hi] 内的单元索引（None 表示该端不设限）。"""
    arr = np.asarray(field, float).reshape(-1)
    if arr.size != fv.n_cells:
        raise ValueError("threshold_cells 需要单元中心场")
    ok = np.ones(arr.size, bool)
    if lo is not None:
        ok &= arr >= lo
    if hi is not None:
        ok &= arr <= hi
    return np.where(ok)[0]


def threshold_surface(fv, field, lo=None, hi=None, fields=None):
    """阈值面：提取取值在 [lo,hi] 内的单元子区域外表面。"""
    cells = threshold_cells(fv, field, lo, hi)
    out = extract_surface(fv, cells, fields=fields)
    out["cells"] = cells
    out["count"] = int(len(cells))
    return out


def mirror_geometry(surface, point=None, normal=(1.0, 0.0, 0.0), merge=False):
    """镜像：把表面几何关于平面 {x:(x-p)·n=0} 反射。

    point 缺省取顶点包围盒中心；merge=True 返回原几何 + 镜像几何的合并
    （顶点拼接、三角形偏移、标量拼接），三角形绕序翻转以保持外向法向。
    """
    V = np.asarray(surface.get("vertices"), float)
    T = np.asarray(surface.get("triangles"), np.int64)
    if V.size == 0:
        return dict(surface)
    p = V.mean(axis=0) if point is None else np.asarray(point, float)
    n = np.asarray(normal, float)
    nn = _safe_norm(n)
    if nn < EPS:
        raise ValueError("mirror_geometry 法向不能为零")
    n = n / nn
    d = (V - p) @ n
    Vm = V - 2.0 * d[:, None] * n
    Tm = T[:, ::-1].copy() if T.size else T.copy()
    if not merge:
        out = dict(surface)
        out["vertices"] = Vm
        out["triangles"] = Tm
    else:
        nv = len(V)
        out = dict(surface)
        out["vertices"] = np.vstack([V, Vm])
        out["triangles"] = (np.vstack([T, Tm + nv]) if T.size
                            else Tm.copy())
    sc = dict(surface.get("scalars") or {})
    if merge and sc:
        nv = len(V)
        merged = {}
        for k, a in sc.items():
            a = np.asarray(a, float)
            if a.ndim == 1 and a.shape[0] == nv:
                merged[k] = np.concatenate([a, a])
            elif a.ndim == 2 and a.shape[0] == nv:
                merged[k] = np.vstack([a, a])
            else:
                merged[k] = a
        out["scalars"] = merged
    out["mirrored"] = True
    out["point"] = p
    out["normal"] = n
    return out


def vector_glyphs(fv, vector, stride=1, scale=None, points=None,
                  max_glyphs=2000, cmap=None):
    """矢量符号：在单元质心（或给定点）生成矢量线段几何。

    返回 {"points"(N,3) 起点, "tips"(N,3) 终点, "vectors"(N,3), "scalars"(N,)
    幅值, "rgba"(N,4), "scale", "count"}。scale 缺省使最大矢量长度约为
    0.25·包围盒对角线。
    """
    if points is None:
        pts = np.asarray(fv.centroids, float)
        vec = np.asarray(vector, float)
        if vec.shape[0] != fv.n_cells:
            raise ValueError("vector_glyphs 需单元中心矢量场或显式 points")
    else:
        pts = np.atleast_2d(np.asarray(points, float))
        vec = sample_vector(fv, np.asarray(vector, float), pts)
    mag = vector_magnitude(vec)
    sel = np.arange(len(pts))
    if stride and int(stride) > 1:
        sel = sel[::int(stride)]
    if max_glyphs and len(sel) > int(max_glyphs):
        sel = sel[np.linspace(0, len(sel) - 1,
                              int(max_glyphs)).astype(np.int64)]
    finite = np.isfinite(mag[sel]) & np.all(np.isfinite(vec[sel]), axis=1)
    sel = sel[finite]
    pts, vec, mag = pts[sel], vec[sel], mag[sel]
    if scale is None:
        diag = _safe_norm(fv.vertices.max(axis=0) - fv.vertices.min(axis=0))
        mmax = float(np.max(mag)) if len(mag) else 0.0
        scale = (0.25 * diag / mmax) if mmax > EPS else 1.0
    return {"points": pts, "tips": pts + float(scale) * vec, "vectors": vec,
            "scalars": mag, "rgba": map_scalars(mag, cmap=cmap),
            "scale": float(scale), "count": int(len(pts))}


# ===========================================================================
# V3 派生零件
# ===========================================================================
def probe(fv, fields, points, vectors=None):
    """探针：在任意点集上重心插值全部字段。

    fields: {名称: 数组}，1 维（单元中心或逐顶点）标量走 sample_scalar，
    2 维 (M,3) 矢量走 sample_vector；vectors 为额外的矢量字典。
    返回 {"points","cells","inside","values":{名称: 数组}}；
    域外单元索引为 -1，对应值为 NaN。
    """
    pts = np.atleast_2d(np.asarray(points, float))
    cells = np.full(len(pts), -1, np.int64)
    for i, p in enumerate(pts):
        c, _ = locate_cell(fv, p)
        cells[i] = c
    values = {}
    for name, arr in (fields or {}).items():
        a = np.asarray(arr, float)
        values[name] = (sample_vector(fv, a, pts) if a.ndim == 2
                        else sample_scalar(fv, a, pts))
    for name, arr in (vectors or {}).items():
        values[name] = sample_vector(fv, np.asarray(arr, float), pts)
    return {"points": pts, "cells": cells, "inside": cells >= 0,
            "values": values}


def line_sample(fv, fields, a, b, n=100, vectors=None):
    """直线采样：a→b 上均匀取 n 点探针。

    返回 probe(...) 的结果并补 "t"（0→1）, "length"。
    """
    a = np.asarray(a, float).reshape(3)
    b = np.asarray(b, float).reshape(3)
    n = max(2, int(n))
    t = np.linspace(0.0, 1.0, n)
    pts = a[None, :] + t[:, None] * (b - a)[None, :]
    out = probe(fv, fields, pts, vectors=vectors)
    out["t"] = t
    out["length"] = float(_safe_norm(b - a))
    return out


def plane_sample(fv, fields, point, normal, n=40, vectors=None):
    """平面采样：在给定平面上对包围盒投影生成 n×n 网格并插值。

    返回 probe(...) 结果并补 "u","v","grid_shape","normal","point"。
    """
    p = np.asarray(point, float).reshape(3)
    nrm = np.asarray(normal, float).reshape(3)
    nn = _safe_norm(nrm)
    nrm = nrm / nn if nn > EPS else np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ nrm)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - float(ref @ nrm) * nrm
    u = u / _safe_norm(u)
    v = np.cross(nrm, u)
    d = fv.vertices - p
    uu = d @ u
    vv = d @ v
    n = max(2, int(n))
    us = np.linspace(float(uu.min()), float(uu.max()), n)
    vs = np.linspace(float(vv.min()), float(vv.max()), n)
    U, Vg = np.meshgrid(us, vs, indexing="ij")
    pts = (p[None, :] + U.reshape(-1, 1) * u[None, :]
           + Vg.reshape(-1, 1) * v[None, :])
    out = probe(fv, fields, pts, vectors=vectors)
    out["u"] = U.reshape(-1)
    out["v"] = Vg.reshape(-1)
    out["grid_shape"] = (n, n)
    out["normal"] = nrm
    out["point"] = p
    return out


def iso_volume(fv, scalar, iso, above=True):
    """等值体：标量 >= iso（above=True）或 <= iso 的单元体积之和。

    scalar 为单元中心场；若为逐顶点场则按单元顶点均值折算为单元值。
    返回 {"volume","fraction","cells","count","iso","above"}。
    """
    val = np.asarray(scalar, float).reshape(-1)
    if val.size == fv.n_vertices and val.size != fv.n_cells:
        val = np.asarray([float(np.mean(val[fv.cells[c]]))
                          for c in range(fv.n_cells)], float)
    mask = (val >= float(iso)) if above else (val <= float(iso))
    cells = np.nonzero(mask)[0].astype(np.int64)
    vol_cells = np.asarray(fv.volumes, float)[cells] if len(cells) else None
    vol = float(np.sum(vol_cells)) if vol_cells is not None else 0.0
    total = float(np.sum(np.asarray(fv.volumes, float)))
    return {"volume": vol, "fraction": (vol / total) if total > EPS else 0.0,
            "cells": cells, "count": int(len(cells)), "iso": float(iso),
            "above": bool(above)}


def threshold_part(fv, field, lo=None, hi=None, fields=None):
    """阈值零件：单元子集 + 其外表面 + 体积统计。

    返回 extract_surface(...) 的结果并补 "cells","count","volume","lo","hi"。
    """
    cells = threshold_cells(fv, field, lo=lo, hi=hi)
    surf = extract_surface(fv, cells, fields=fields)
    vol = (float(np.sum(np.asarray(fv.volumes, float)[cells]))
           if len(cells) else 0.0)
    surf["cells"] = cells
    surf["count"] = int(len(cells))
    surf["volume"] = vol
    surf["lo"] = lo
    surf["hi"] = hi
    return surf


def plane_box_polygon(lo, hi, point, normal):
    """平面 {x:(x-p)·n=0} 与轴对齐盒 [lo,hi] 的相交多边形（有序 (K,3)）。

    无交或退化为点/线时返回 (0,3) 空数组。
    """
    lo = np.asarray(lo, float).reshape(3)
    hi = np.asarray(hi, float).reshape(3)
    p = np.asarray(point, float).reshape(3)
    nrm = np.asarray(normal, float).reshape(3)
    corners = np.asarray([[hi[d] if (m >> d) & 1 else lo[d]
                           for d in range(3)] for m in range(8)], float)
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") != 1:
                continue
            di = float((corners[i] - p) @ nrm)
            dj = float((corners[j] - p) @ nrm)
            if di == 0.0:
                pts.append(corners[i])
            if di * dj < 0.0:
                t = di / (di - dj)
                pts.append(corners[i] + t * (corners[j] - corners[i]))
    if len(pts) < 3:
        return np.zeros((0, 3), float)
    P = np.asarray(pts, float)
    if _safe_norm(nrm) > EPS:
        return _order_polygon(P, nrm)
    return P


def _sig(v):
    if isinstance(v, np.ndarray):
        return "arr:" + ",".join("%.9g" % x
                                 for x in np.asarray(v, float).reshape(-1))
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(_sig(x) for x in v) + ")"
    return repr(v)


class DerivedCache:
    """派生零件缓存：按 (名称, 参数签名) 存储，命中/未命中计数。"""

    def __init__(self, fv=None):
        self.fv = fv
        self._store = {}
        self.hits = 0
        self.misses = 0

    def key(self, name, params=None):
        items = tuple(sorted((str(k), _sig(v))
                             for k, v in (params or {}).items()))
        return (str(name), items)

    def get_or_compute(self, name, params, factory):
        k = self.key(name, params)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        val = factory()
        self._store[k] = val
        return val

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._store)


# ===========================================================================
# V4 绘图数据
# ===========================================================================
def xy_series(x, y):
    """XY 序列：打包 x/y 及范围统计（保留原始长度，附有效点数）。"""
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    n = min(len(x), len(y))
    x, y = x[:n].copy(), y[:n].copy()
    good = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[good], y[good]
    return {"x": x, "y": y, "n": int(n),
            "valid": int(np.count_nonzero(good)),
            "x_range": ((float(xv.min()), float(xv.max()))
                        if len(xv) else (0.0, 0.0)),
            "y_range": ((float(yv.min()), float(yv.max()))
                        if len(yv) else (0.0, 0.0))}


def histogram(values, bins=DEFAULT_BINS, range=None, density=False):
    """直方图：计数/边界/中心/带宽/均值；density=True 时附 "pdf"。"""
    v = np.asarray(values, float).reshape(-1)
    v = v[np.isfinite(v)]
    bins = max(1, int(bins))
    if len(v) == 0:
        edges = np.linspace(0.0, 1.0, bins + 1)
        return {"counts": np.zeros(bins, np.int64), "edges": edges,
                "centers": 0.5 * (edges[:-1] + edges[1:]),
                "bin_width": 1.0 / bins, "n": 0, "range": (0.0, 0.0),
                "mean": float("nan"), "density": bool(density)}
    lo, hi = ((float(range[0]), float(range[1])) if range
              else (float(v.min()), float(v.max())))
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    counts = counts.astype(np.int64)
    out = {"counts": counts, "edges": edges,
           "centers": 0.5 * (edges[:-1] + edges[1:]),
           "bin_width": float(edges[1] - edges[0]), "n": int(len(v)),
           "range": (lo, hi), "mean": float(v.mean()),
           "density": bool(density)}
    if density:
        tot = int(counts.sum())
        out["pdf"] = (counts / (tot * out["bin_width"]) if tot
                      else counts * 0.0)
    return out


def cumulative_distribution(values, bins=DEFAULT_BINS, range=None):
    """累积分布曲线：返回 {"x","cdf","counts","edges","n"}。"""
    h = histogram(values, bins=bins, range=range)
    counts = h["counts"].astype(float)
    tot = counts.sum()
    cdf = np.cumsum(counts) / tot if tot > 0.0 else np.zeros_like(counts)
    return {"x": h["centers"], "cdf": cdf, "counts": h["counts"],
            "edges": h["edges"], "n": h["n"]}


def decimate_series(x, y, max_points=DEFAULT_MAX_PTS):
    """曲线抽稀：按 x 排序后分桶保留每桶首/尾与 y 极值，至多约 max_points。"""
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n == 0:
        return {"x": x.copy(), "y": y.copy(), "n": 0, "original": 0}
    max_points = max(2, int(max_points))
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    if n <= max_points:
        return {"x": xs, "y": ys, "n": int(n), "original": int(n)}
    nb = max(1, max_points // 2)
    edges = np.linspace(0, n, nb + 1).astype(np.int64)
    keep = []
    for b in range(nb):
        s, e = int(edges[b]), int(edges[b + 1])
        if e <= s:
            continue
        seg = ys[s:e]
        keep.extend([s, s + int(np.argmin(seg)), s + int(np.argmax(seg)),
                     e - 1])
    keep = np.unique(np.asarray(keep, np.int64))
    return {"x": xs[keep], "y": ys[keep], "n": int(len(keep)),
            "original": int(n)}


class MonitorBuffer:
    """监视器实时缓冲：滚动保存 (iteration, 各名称值) 供 XY 图刷新。"""

    def __init__(self, names, capacity=1024):
        self.names = list(names)
        self.capacity = max(2, int(capacity))
        self.iterations = []
        self._data = {nm: [] for nm in self.names}

    def append(self, iteration, values):
        vals = dict(values)
        self.iterations.append(float(iteration))
        for nm in self.names:
            self._data[nm].append(float(vals.get(nm, np.nan)))
        if len(self.iterations) > self.capacity:
            drop = len(self.iterations) - self.capacity
            self.iterations = self.iterations[drop:]
            for nm in self.names:
                self._data[nm] = self._data[nm][drop:]

    def extend(self, iterations, series):
        it = np.asarray(iterations, float).reshape(-1)
        for i, v in enumerate(it):
            self.append(v, {nm: np.asarray(series[nm], float).reshape(-1)[i]
                            for nm in self.names})

    def series(self, name):
        if name not in self._data:
            raise KeyError(name)
        return xy_series(self.iterations, self._data[name])

    def arrays(self):
        it = np.asarray(self.iterations, float)
        return it, {nm: np.asarray(self._data[nm], float) for nm in self.names}

    def clear(self):
        self.iterations = []
        self._data = {nm: [] for nm in self.names}

    def __len__(self):
        return len(self.iterations)


# ===========================================================================
# V5 注记 / 图例 / 色标尺 / 动画帧
# ===========================================================================
def colorbar_ticks(lo, hi, n=6, fmt="%.4g"):
    """色标尺刻度：返回 {"values","labels","lo","hi","n"}。"""
    n = max(2, int(n))
    vals = np.linspace(float(lo), float(hi), n)
    return {"values": vals, "labels": [fmt % v for v in vals],
            "lo": float(lo), "hi": float(hi), "n": n}


def colorbar_strip(cmap=None, samples=DEFAULT_CMAP_SAMPLES):
    """色标尺色带：0→1 采样得到 (samples,4) RGBA。"""
    samples = max(2, int(samples))
    t = np.linspace(0.0, 1.0, samples)
    return {"rgba": map_scalars(t, cmap=cmap, lo=0.0, hi=1.0),
            "samples": samples, "height": samples}


def legend_items(labels, colors=None, cmap=None):
    """图例项：标签→颜色。colors 缺省时按 cmap 在 [0,1] 均匀取样。"""
    labels = list(labels)
    if colors is None:
        n = max(1, len(labels))
        t = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
        rgba = map_scalars(t, cmap=cmap, lo=0.0, hi=1.0)
    else:
        rgba = np.asarray(colors, float)
        if rgba.ndim == 1:
            rgba = np.tile(rgba[None, :], (len(labels), 1))
    return [{"label": str(lb), "rgba": rgba[i]}
            for i, lb in enumerate(labels)]


def annotation(text, position=(0.02, 0.95), color=(1.0, 1.0, 1.0, 1.0),
               size=14, align="left"):
    """注记：返回可渲染的文本描述字典。"""
    return {"text": str(text), "position": tuple(float(x) for x in position),
            "color": tuple(float(c) for c in color), "size": int(size),
            "align": str(align)}


def frame_times(n, dt, start=0.0):
    """n 帧时间轴 t = start + i·dt。"""
    return float(start) + float(dt) * np.arange(max(0, int(n)))


def frame_indices(n, start=0, step=1):
    """n 帧序号（可自定义起点/步长）。"""
    return int(start) + int(step) * np.arange(max(0, int(n)), dtype=np.int64)


def frame_name(prefix, index, digits=4, ext="png"):
    """动画帧文件名：prefix_0001.png。"""
    return "%s_%0*d.%s" % (str(prefix), int(digits), int(index), str(ext))


def _to_rgb8(image):
    a = np.asarray(image, float)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("图像需 (H,W,3) 或 (H,W,4)")
    if a.size and float(a.max()) <= 1.0 + 1.0e-9:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def write_ppm(path, image):
    """写二进制 PPM(P6)；image 为 (H,W,3)/(H,W,4)，float[0,1] 或 uint8。"""
    a = _to_rgb8(image)
    h, w = a.shape[:2]
    header = ("P6\n%d %d\n255\n" % (w, h)).encode("ascii")
    with open(path, "wb") as f:
        f.write(header + a.tobytes())
    return path


def _png_chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png(path, image, compression=6):
    """写 PNG(RGB8)，纯 zlib/struct 实现，无第三方依赖。"""
    a = _to_rgb8(image)
    h, w = a.shape[:2]
    flat = a.reshape(h, w * 3)
    raw = bytearray()
    for r in range(h):
        raw.append(0)
        raw += flat[r].tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), int(compression)))
    png += _png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
    return path


def write_gif(path, frames, duration=100, loop=0):
    """用 PIL 将帧序列写 GIF；Pillow 不可用时抛 RuntimeError（诚实降级）。"""
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("write_gif 需要 Pillow: %s" % exc)
    imgs = [Image.fromarray(_to_rgb8(fr), "RGB") for fr in frames]
    if not imgs:
        raise ValueError("write_gif 需要至少一帧")
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(duration), loop=int(loop))
    return path


def export_animation(frames, out_dir, prefix="frame", fps=10, fmt="png",
                     gif=None, mp4=None):
    """导出动画帧序列。

    始终写 PNG（或 ppm）序列；gif 给定时（需 Pillow）额外写 GIF；
    mp4 需 ffmpeg，本环境缺失时返回中标注 "mp4": "unavailable(ffmpeg)"。
    """
    os.makedirs(out_dir, exist_ok=True)
    low = str(fmt).lower()
    files = []
    for i, fr in enumerate(frames):
        p = os.path.join(out_dir, frame_name(prefix, i, ext=low))
        (write_ppm(p, fr) if low == "ppm" else write_png(p, fr))
        files.append(p)
    out = {"frames": files, "count": len(files), "fps": int(fps),
           "gif": None, "mp4": None}
    if gif:
        write_gif(gif, frames, duration=max(20, int(1000.0 / max(1, fps))))
        out["gif"] = gif
    if mp4:
        out["mp4"] = "unavailable(ffmpeg)"
    return out


# ===========================================================================
# V6 数据写出 CSV / EnSight / CGNS
# ===========================================================================
def write_csv(path, columns, headers=None):
    """写 CSV：columns 为 {名称: 1维数组} 或二维数组；返回 path。"""
    if isinstance(columns, dict):
        names = list(columns.keys())
        cols = [np.asarray(columns[n], float).reshape(-1) for n in names]
    else:
        arr = np.asarray(columns, float)
        if arr.ndim == 1:
            arr = arr[:, None]
        names = (list(headers) if headers
                 else ["col%d" % i for i in range(arr.shape[1])])
        cols = [arr[:, i] for i in range(arr.shape[1])]
    n = min([len(c) for c in cols]) if cols else 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(names) + "\n")
        for i in range(n):
            f.write(",".join("%.16g" % c[i] for c in cols) + "\n")
    return path


def export_csv(path, fv, fields, include_geometry=True):
    """导出单元中心场 CSV（含单元索引与质心坐标）。"""
    cols = {}
    if include_geometry:
        cols["cell"] = np.arange(fv.n_cells, dtype=float)
        cen = np.asarray(fv.centroids, float)
        cols["x"], cols["y"], cols["z"] = cen[:, 0], cen[:, 1], cen[:, 2]
    for name, arr in (fields or {}).items():
        a = np.asarray(arr, float)
        if a.ndim == 1:
            cols[str(name)] = a
        else:
            for d, suf in enumerate("xyz"[:a.shape[1]]):
                cols["%s_%s" % (name, suf)] = a[:, d]
    return write_csv(path, cols)


def _ensight_vals(arr):
    a = np.asarray(arr, float).reshape(-1)
    out = []
    for i in range(0, len(a), 6):
        out.append(" ".join("%.9g" % x for x in a[i:i + 6]))
    return "\n".join(out)


def write_ensight(out_dir, fv, fields=None, base_name="mesh"):
    """写 EnSight Gold ASCII（.case/.geo/.dat）非结构四面体网格。

    fields: {名称: 1维单元中心标量 或 (ncell,3) 矢量}。
    返回 {"case","geo","dat","variables"}。
    """
    os.makedirs(out_dir, exist_ok=True)
    geo_path = os.path.join(out_dir, base_name + ".geo")
    V = np.asarray(fv.vertices, float)
    C = np.asarray(fv.cells, np.int64)
    gl = ["extradata", "stardecoding tetra mesh",
          "generated by postprocess.py",
          "node id given", "element id given", "coordinates",
          str(fv.n_vertices)]
    for i in range(len(V)):
        p = V[i]
        gl.append("%d %.9g %.9g %.9g" % (i + 1, p[0], p[1], p[2]))
    gl.append("tetra4")
    gl.append(str(fv.n_cells))
    for i in range(len(C)):
        c = C[i]
        gl.append("%d %d %d %d %d"
                  % (i + 1, c[0] + 1, c[1] + 1, c[2] + 1, c[3] + 1))
    with open(geo_path, "w", encoding="ascii") as f:
        f.write("\n".join(gl) + "\n")

    entries = []
    for name, arr in (fields or {}).items():
        a = np.asarray(arr, float)
        var_file = "%s_%s.dat" % (base_name, name)
        var_path = os.path.join(out_dir, var_file)
        body = [str(name), "part", "1"]
        if a.ndim == 1:
            body.append(_ensight_vals(a))
            kind = "scalar per element"
        else:
            for d in range(a.shape[1]):
                body.append(_ensight_vals(a[:, d]))
            kind = "vector per element"
        with open(var_path, "w", encoding="ascii") as f:
            f.write("\n".join(body) + "\n")
        entries.append((kind, str(name), var_file))

    case_path = os.path.join(out_dir, base_name + ".case")
    cl = ["FORMAT", "type: ensight gold", "", "GEOMETRY",
          "model: %s.geo" % base_name]
    if entries:
        cl.extend(["", "VARIABLE"])
        for kind, name, vf in entries:
            cl.append("%s: %s %s" % (kind, name, vf))
    with open(case_path, "w", encoding="ascii") as f:
        f.write("\n".join(cl) + "\n")
    return {"case": case_path, "geo": geo_path,
            "dat": [os.path.join(out_dir, vf) for _, _, vf in entries],
            "variables": [n for _, n, _ in entries]}


def write_cgns(path, fv, fields=None, base_name="Base", zone_name="Zone",
               solution_name="FlowSolution"):
    """写 CGNS/HDF5 非结构四面体网格 + 单元中心解场（h5py 必需）。

    采用 CGNS SIDS 的 HDF5 布局（CGNSBase_t / Zone_t / GridCoordinates_t /
    Elements_t / FlowSolution_t，四面体元素类型 10），返回 path；
    h5py 缺失时抛 RuntimeError（诚实降级）。
    """
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("write_cgns 需要 h5py: %s" % exc)
    V = np.asarray(fv.vertices, float)
    C = np.asarray(fv.cells, np.int64) + 1
    nv, nc = int(fv.n_vertices), int(fv.n_cells)

    def tagged(obj, name, label, type_=None):
        obj.attrs["name"] = np.bytes_(name)
        obj.attrs["label"] = np.bytes_(label)
        if type_ is not None:
            obj.attrs["type"] = np.asarray(type_, np.int32)
        return obj

    with h5py.File(path, "w") as h:
        d = h.create_dataset("CGNSLibraryVersion",
                             data=np.array([4.0], np.float32))
        tagged(d, "CGNSLibraryVersion", "CGNSLibraryVersion_t")
        base = tagged(h.create_group(base_name), base_name, "CGNSBase_t",
                      [3, 3])
        zone = tagged(base.create_group(zone_name), zone_name, "Zone_t",
                      [nv, nc, 0])
        tagged(zone.create_dataset("ZoneType",
                                   data=np.bytes_("Unstructured")),
               "ZoneType", "ZoneType_t")
        gc = tagged(zone.create_group("GridCoordinates"), "GridCoordinates",
                    "GridCoordinates_t")
        for d_, nm in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
            tagged(gc.create_dataset(nm, data=np.ascontiguousarray(V[:, d_])),
                   nm, "DataArray_t")
        el = tagged(zone.create_group("Elements"), "Elements", "Elements_t",
                    [10, 0])
        tagged(el.create_dataset("ElementRange",
                                 data=np.array([1, nc], np.int32)),
               "ElementRange", "IndexRange_t")
        tagged(el.create_dataset("ElementConnectivity",
                                 data=C.astype(np.int32)),
               "ElementConnectivity", "DataArray_t")
        if fields:
            fs = tagged(zone.create_group(solution_name), solution_name,
                        "FlowSolution_t")
            tagged(fs.create_dataset("GridLocation",
                                     data=np.array([1], np.int32)),
                   "GridLocation", "DataArray_t")
            for nm, arr in fields.items():
                a = np.asarray(arr, float)
                if a.ndim == 1:
                    tagged(fs.create_dataset(str(nm), data=a), str(nm),
                           "DataArray_t")
                else:
                    for d_ in range(a.shape[1]):
                        vn = "%s%s" % (nm, "XYZ"[d_])
                        tagged(fs.create_dataset(vn, data=a[:, d_]), vn,
                               "DataArray_t")
    return path


# ===========================================================================
# 门面：字段抽取 + PostProcessor + 工厂
# ===========================================================================
def _solver_fv(solver):
    """从求解器对象解析其 FVM 网格（兼容 fv / fvm / _fv 三种属性名）。"""
    for attr in ("fv", "fvm", "_fv"):
        cand = getattr(solver, attr, None)
        if cand is not None:
            return cand
    return None


def fields_from_solver(solver, fv=None, names=None):
    """从求解器对象抽取后处理常用场 {名称: 数组}。

    支持 PressureSolver（velocity/pressure/rho）、CompressibleSolver
    （额外 T/mach）等；names 给定时仅返回其中的可用名称。
    fv 缺省取 solver 的 FVM 网格（fv / fvm / _fv，用于标量物性广播）。
    """
    fvm = fv if fv is not None else _solver_fv(solver)
    if fvm is None:
        raise ValueError("fields_from_solver 需要 fv 或 solver.fv")
    raw = {}

    def add(name, val):
        if val is None:
            return
        try:
            arr = np.asarray(val, float)
        except Exception:
            return
        if arr.ndim == 0:
            arr = np.full(fvm.n_cells, float(arr))
        raw[name] = arr

    vel = None
    if hasattr(solver, "velocity"):
        try:
            vel = np.asarray(solver.velocity(), float)
        except Exception:
            vel = None
    if vel is None:
        u = getattr(solver, "_u", None)
        v = getattr(solver, "_v", None)
        w = getattr(solver, "_w", None)
        if u is not None and v is not None and w is not None:
            vel = np.stack([np.asarray(u, float), np.asarray(v, float),
                            np.asarray(w, float)], axis=1)
    if vel is not None:
        add("velocity", vel)
        add("speed", vector_magnitude(vel))

    p = getattr(solver, "pressure", None)
    add("pressure", p() if callable(p) else p)
    for name in ("rho", "T", "temperature", "mach"):
        val = getattr(solver, name, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        add(name, val)
    for attr in ("_rho", "_T", "_mach"):
        if attr[1:] not in raw:
            add(attr[1:], getattr(solver, attr, None))
    if names is not None:
        return {k: raw[k] for k in names if k in raw}
    return raw


class PostProcessor:
    """后处理门面：绑定 (fv, fields) 并暴露 V1–V6 全部能力。"""

    def __init__(self, fv, fields=None, cache=None):
        self.fv = fv
        self.fields = {k: np.asarray(v, float)
                       for k, v in (fields or {}).items()}
        self.cache = cache if cache is not None else DerivedCache(fv)

    def set_field(self, name, values):
        self.fields[str(name)] = np.asarray(values, float)
        return self

    def update_fields(self, fields):
        for k, v in (fields or {}).items():
            self.fields[str(k)] = np.asarray(v, float)
        return self

    @classmethod
    def from_solver(cls, solver, fv=None, names=None):
        fvm = fv if fv is not None else _solver_fv(solver)
        return cls(fvm, fields_from_solver(solver, fv=fvm, names=names))

    def _field(self, name):
        if name not in self.fields:
            raise KeyError("场 %r 不存在" % (name,))
        return self.fields[name]

    def _resolve(self, names):
        if names is None:
            return None
        return {n: self._field(n) for n in names}

    # -- V1 色彩 -------------------------------------------------
    def color(self, name, **kw):
        return color_by(self._field(name), **kw)

    # -- V2 显示器 -----------------------------------------------
    def streamline(self, seed, **kw):
        return streamline(self.fv, self._field("velocity"), seed, **kw)

    def streamlines(self, seeds, **kw):
        return streamlines(self.fv, self._field("velocity"), seeds, **kw)

    def isosurface(self, name, iso, fields=None):
        fld = np.asarray(self._field(name), float).reshape(-1)
        if fld.size == self.fv.n_vertices and fld.size != self.fv.n_cells:
            nodal = fld
        else:
            nodal = cell_to_vertex(self.fv, fld)
        return marching_tets(self.fv, nodal, float(iso),
                             fields=self._resolve(fields))

    def section(self, point, normal, fields=None):
        return section_plane(self.fv, point, normal,
                             fields=self._resolve(fields))

    def clip(self, point, normal, keep="negative", fields=None):
        return clip_plane(self.fv, point, normal, keep=keep,
                          fields=self._resolve(fields))

    def threshold(self, name, lo=None, hi=None, fields=None):
        return threshold_part(self.fv, self._field(name), lo=lo, hi=hi,
                              fields=self._resolve(fields))

    def mirror(self, surface, **kw):
        return mirror_geometry(surface, **kw)

    def glyphs(self, name, **kw):
        return vector_glyphs(self.fv, self._field(name), **kw)

    def derived(self, name, params, factory):
        """经缓存获取/计算派生零件（同一名称+参数只算一次）。"""
        return self.cache.get_or_compute(name, params, factory)

    # -- V3 派生零件 ---------------------------------------------
    def _subset(self, names):
        names = list(names) if names else list(self.fields.keys())
        return {k: self.fields[k] for k in names if k in self.fields}

    def probe(self, points, names=None, vectors=None):
        return probe(self.fv, self._subset(names), points, vectors=vectors)

    def line(self, a, b, n=100, names=None):
        return line_sample(self.fv, self._subset(names), a, b, n=n)

    def plane(self, point, normal, n=40, names=None):
        return plane_sample(self.fv, self._subset(names), point, normal, n=n)

    def iso_volume(self, name, iso, above=True):
        return iso_volume(self.fv, self._field(name), float(iso), above=above)

    # -- V4 绘图数据 ---------------------------------------------
    def xy(self, x, y):
        return xy_series(x, y)

    def histogram(self, name, **kw):
        return histogram(self._field(name), **kw)

    def cdf(self, name, **kw):
        return cumulative_distribution(self._field(name), **kw)

    # -- V5 注记 / 图例 / 动画 -----------------------------------
    def colorbar(self, name, n=6, fmt="%.4g"):
        lo, hi = scalar_range(self._field(name))
        return colorbar_ticks(lo, hi, n=n, fmt=fmt)

    def legend(self, labels, **kw):
        return legend_items(labels, **kw)

    def animate(self, frames, out_dir, **kw):
        return export_animation(frames, out_dir, **kw)

    # -- V6 数据写出 ---------------------------------------------
    def export_csv(self, path, names=None, include_geometry=True):
        return export_csv(path, self.fv, self._subset(names),
                          include_geometry=include_geometry)

    def export_ensight(self, out_dir, names=None, base_name="mesh"):
        return write_ensight(out_dir, self.fv, self._subset(names),
                             base_name=base_name)

    def export_cgns(self, path, names=None, **kw):
        return write_cgns(path, self.fv, self._subset(names), **kw)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_POSTPROCESS_MODELS = {
    "postprocess": PostProcessor,
    "post_processor": PostProcessor,
    "postprocessor": PostProcessor,
    "post_processing": PostProcessor,
    "post": PostProcessor,
    "display": PostProcessor,
    "visualization": PostProcessor,
    "visualisation": PostProcessor,
}


def make_postprocessor(fv=None, fields=None, solver=None,
                       model="postprocess", **kwargs):
    """按字符串名实例化后处理器（大小写/连字符/下划线不敏感）。

    可传 fv+fields，或直接传 solver（自动 fv= solver.fv 并抽取字段）。
    """
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _POSTPROCESS_MODELS:
        names = ", ".join(sorted(set(_POSTPROCESS_MODELS.keys())))
        raise ValueError("未知后处理器 %r（支持：%s）" % (model, names))
    if fv is None and solver is not None:
        fv = _solver_fv(solver)
    if fv is None:
        raise ValueError("make_postprocessor 需要 fv 或 solver")
    if fields is None and solver is not None:
        fields = fields_from_solver(solver, fv=fv)
    return _POSTPROCESS_MODELS[key](fv, fields=fields, **kwargs)