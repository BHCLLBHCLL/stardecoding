# -*- coding: utf-8 -*-
"""N 波 N5（三）：extruder（线性/旋转）+ directed 推进网格器（纯 numpy）。

STAR-CCM+ 三种"推"式网格器（相对 prism 边界层的法向碰撞推进）：
  - `extrude_mesh`（extruder 线性）：源 2D 表面网格沿固定方向按给定层深
    （显式 distances 或几何级数 t0·r^k）平移提升，层间 prism 列；
  - `extrude_rotate`（extruder 旋转）：源网格绕轴按总角 n 层旋进
    （Rodrigues 旋转，层间 warped prism 列），体积对标 Pappus
    V = A·θ·d（扫掠角 θ × 面积 × 质心到轴距）；
  - `directed_mesh`（directed）：源网格沿**固定方向**逐顶点射线到目标
    表面取 h_v，几何级数层深逐顶点按 h_v/总厚 比例缩放贯通到目标
    （fill_to_target=True；False 则固定层深、不足处截断到目标），
    未命中目标的顶点抛 ValueError（STAR-CCM+ directed 同样拒绝）。

三者提升点均按 (顶点, 层) 共享 → 相邻列侧墙共享、拼合无缝；单元统一
prism 6 顶点 [a0,b0,c0,a1,b1,c1]，质量走 mesh_prism.prism_quality。

与 N4 prism_layers 的分工：prism = 法向推进 + 碰撞降层；thin = 法向
一次贯通全厚；directed = 定向推进到目标；extrude = 无目标按给定深度/
角度推。

验收要点（tests/test_mesh_extrude.py）：
  线性：平板 distances 精确回显、体积=面积×总厚 精确、几何级数便捷
  参数与显式 distances 等价；非法 distances/方向 ValueError；
  旋转：跨 z 方形面绕 z 轴 2π@16 层 体积≈π（Pappus V=A·θ·d，5%）、
  π/2@8 层 ≈π/4（2%）、层间角度均分；
  directed：平行板 0.3 间距 fill 模式体积=0.3 精确且顶面恰在目标面、
  固定模式体积=几何级数总厚、目标未覆盖源足迹 → ValueError。
"""
import numpy as np

from mesh_prism import ray_hit_distance, prism_quality


def _prism_cells(F, lift, n_layers):
    """按 (v, j) 提升点表组装 prism 单元 [a0,b0,c0,a1,b1,c1]。"""
    cells = []
    for a, b, c in F:
        if (a, 1) not in lift or (b, 1) not in lift or (c, 1) not in lift:
            continue
        for j in range(1, n_layers + 1):
            a0 = int(a) if j == 1 else lift[(int(a), j - 1)]
            b0 = int(b) if j == 1 else lift[(int(b), j - 1)]
            c0 = int(c) if j == 1 else lift[(int(c), j - 1)]
            cells.append([a0, b0, c0, lift[(int(a), j)], lift[(int(b), j)],
                          lift[(int(c), j)]])
    return cells


def _geometric_levels(n_layers, first_thickness, stretch):
    """几何级数累计层厚 d_k = t0·(r^k−1)/(r−1)（r=1 时 = k·t0）。"""
    levels = []
    d = 0.0
    for kk in range(int(n_layers)):
        d += first_thickness * (stretch ** kk)
        levels.append(d)
    return levels


def _finish(newV, cells, extra):
    cells_arr = (np.asarray(cells, np.int64) if cells
                 else np.zeros((0, 6), np.int64))
    quality = prism_quality(newV, cells_arr)
    out = {"ok": len(cells) > 0,
           "vertices": np.asarray(newV, float),
           "cells": cells_arr,
           "n_cells": len(cells),
           "quality": quality,
           "volume_total": quality["volume_total"]}
    out.update(extra)
    return out


def _check_surface(V, F):
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or len(V) == 0 or len(F) == 0:
        raise ValueError("需要非空源表面 (vertices Nx3, faces Mx3)")
    return V, F


# --------------------------------------------------------------- extruder 线性
def extrude_mesh(vertices, faces, distances=None, direction=(0.0, 0.0, 1.0),
                 n_layers=3, first_thickness=None, stretch=1.5):
    """extruder 线性：源表面沿固定方向按层深平移提升 prism 列。

    distances：显式累计层深（正且严格递增，≥1 层）；缺省用几何级数
    （first_thickness t0 缺省 = 包围盒对角×1%，stretch r，n_layers 层）。
    direction：推进方向（非零，单位化后使用）。
    """
    V, F = _check_surface(vertices, faces)
    d = np.asarray(direction, float)
    dn = float(np.linalg.norm(d))
    if dn < 1e-300:
        raise ValueError("direction 须为非零向量")
    d = d / dn
    if distances is not None:
        levels = [float(x) for x in distances]
        if not levels or any(not (x > 0.0) for x in levels):
            raise ValueError("distances 须为正数列表")
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError("distances 须严格递增")
        n = len(levels)
    else:
        n = max(1, int(n_layers))
        if first_thickness is None:
            diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
            first_thickness = diag * 0.01
        t0 = max(float(first_thickness), 1e-12)
        levels = _geometric_levels(n, t0, float(stretch))

    newV = [list(map(float, p)) for p in V]
    lift = {}
    for v in range(len(V)):
        for j in range(1, n + 1):
            lift[(v, j)] = len(newV)
            newV.append((V[v] + d * levels[j - 1]).tolist())
    cells = _prism_cells(F, lift, n)
    return _finish(newV, cells, {"levels": levels, "direction": d})


# --------------------------------------------------------------- extruder 旋转
def _rotate_about(points, origin, axis, angle):
    """Rodrigues 旋转：points 绕过 origin 的单位轴 axis 转 angle（弧度）。"""
    P = np.asarray(points, float) - np.asarray(origin, float)
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    eu = u * (P @ u)                      # (u·P)u，保持与 P 同形 (3,)
    return (np.asarray(origin, float)
            + P * c + np.cross(u, P) * s + eu * (1.0 - c))


def extrude_rotate(vertices, faces, axis_point, axis_dir, angle, n_layers):
    """extruder 旋转：源网格绕轴按总角 n 层旋进 prism 列。

    每层累计角 φ_j = j·angle/n_layers（Rodrigues），层间 warped prism。
    体积对标 Pappus：V = 面积 × θ × 质心到轴距（细分收敛）。
    """
    V, F = _check_surface(vertices, faces)
    u = np.asarray(axis_dir, float)
    un = float(np.linalg.norm(u))
    if un < 1e-300:
        raise ValueError("axis_dir 须为非零向量")
    ang = float(angle)
    if abs(ang) < 1e-12:
        raise ValueError("angle 须为非零弧度")
    n = max(1, int(n_layers))
    o = np.asarray(axis_point, float)

    newV = [list(map(float, p)) for p in V]
    lift = {}
    for v in range(len(V)):
        for j in range(1, n + 1):
            lift[(v, j)] = len(newV)
            newV.append(_rotate_about(V[v], o, u, ang * j / n).tolist())
    cells = _prism_cells(F, lift, n)
    angles = [ang * j / n for j in range(1, n + 1)]
    return _finish(newV, cells, {"angles": angles, "n_layers": n})


# --------------------------------------------------------------- directed
def directed_mesh(vertices, faces, direction, target_vertices, target_faces,
                  n_layers=3, first_thickness=None, stretch=1.5,
                  fill_to_target=True):
    """directed：源网格沿固定方向逐顶点射线到目标表面、贯通生成 prism 列。

    每顶点 h_v = 沿 direction 到目标的最近命中；几何级数层深（t0 缺省 =
    源包围盒对角×1%）逐顶点按 h_v/总厚 缩放：
      fill_to_target=True：比例 = h_v/总厚（首尾都贴目标，层厚整体缩放）；
      False：比例 = min(1, h_v/总厚)（固定层深，不足处截断到目标）。
    任一顶点未命中目标 → ValueError（报未命中数）。
    """
    V, F = _check_surface(vertices, faces)
    TV = np.asarray(target_vertices, float)
    TF = np.asarray(target_faces, np.int64)
    if (TV.ndim != 2 or TV.shape[1] != 3 or len(TV) == 0 or len(TF) == 0):
        raise ValueError("directed_mesh 需要非空目标表面")
    d = np.asarray(direction, float)
    dn = float(np.linalg.norm(d))
    if dn < 1e-300:
        raise ValueError("direction 须为非零向量")
    d = d / dn

    n = max(1, int(n_layers))
    if first_thickness is None:
        diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
        first_thickness = diag * 0.01
    t0 = max(float(first_thickness), 1e-12)
    levels = _geometric_levels(n, t0, float(stretch))
    total = levels[-1]

    h = np.array([ray_hit_distance(V[v], d, TV, TF) for v in range(len(V))])
    n_miss = int(np.sum(~np.isfinite(h)))
    if n_miss:
        raise ValueError("directed_mesh：%d 个源顶点沿方向未命中目标表面"
                         % n_miss)
    scale = h / total if fill_to_target else np.minimum(1.0, h / total)

    newV = [list(map(float, p)) for p in V]
    lift = {}
    for v in range(len(V)):
        for j in range(1, n + 1):
            lift[(v, j)] = len(newV)
            newV.append((V[v] + d * (levels[j - 1] * scale[v])).tolist())
    cells = _prism_cells(F, lift, n)
    return _finish(newV, cells,
                   {"levels": levels, "direction": d,
                    "h_stats": (float(h.min()), float(h.max())),
                    "fill_to_target": bool(fill_to_target)})
