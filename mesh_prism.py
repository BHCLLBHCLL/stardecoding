# -*- coding: utf-8 -*-
"""N 波 N4：prism 边界层推进 + 碰撞处理（纯 numpy，双环境可用）。

STAR-CCM+ prism mesher 的核心思想：从边界面向体内逐层推进三棱柱单元，
层厚按几何级数增长（首层厚度 t0、增长率 r），总厚 = t0·(r^n−1)/(r−1)；
窄通道处两侧边界层相互挤压 → 碰撞处理（自动降层/塌缩）。

实现：
  - `vertex_normals`：面积加权顶点法向（与面片绕向一致，不预设外向）；
  - `ray_hit_distance`：Möller–Trumbore 最近三角命中距离（含边界容差，
    t≈0 的共点命中跳过——射线自表面顶点出发的安全前提）；
  - `prism_layers`：边界层推进主入口——
      · 每顶点沿推进方向射线到最近表面的距离 h_v，允许深度 =
        gap_fraction·h_v（gap_fraction 即「边界层最多占据通道的比例」，
        ≤0.5 时两侧边界层在通道内永不重叠）；
      · 面层高 L_f = 三顶点均满足累计层厚 ≤ 允许深度的最大层数
        （提升点按顶点共享 → 邻面列自动保形一致）→ 超限自动降层
        （碰撞处理）、极端时塌缩为 0；
      · side="inward"/"outward" 用表面内外探针自动定向（需闭合表面）；
        开放表面用 "+n"/"-n" 沿绕向法向显式定向；
      · cells 每单元 6 顶点 [a0,b0,c0,a1,b1,c1]（底面三角 + 顶面三角）。
  - `prism_quality`：单元数/翻转数（符号少数派）/体积统计；
    `first_layer_thickness` 首层厚度均值（y+ 控制：t0 直接决定首层单元
    高度，平壁首层质心高 = t0/2）。

验收要点（tests/test_mesh_prism.py）：
  平板全层：层数/几何级数总厚/体积=面积×总厚 精确、首层质心高=t0/2；
  立方体满层：12 面全部满层、推进点全部留在体内；
  窄通道：gap_fraction=0.9 自动降层、=0.5 两侧无重叠、=0.01 全塌缩。
"""
import numpy as np

from mesh_tet import point_in_mesh


# --------------------------------------------------------------- 法向与射线
def vertex_normals(V, F):
    """面积加权顶点法向（单位化，与面片绕向一致；不预设内外向）。"""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    p, q, r = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(q - p, r - p)              # 模长 = 2×面积
    acc = np.zeros_like(V)
    for k in range(3):
        np.add.at(acc, F[:, k], fn)
    ln = np.linalg.norm(acc, axis=1)
    out = np.zeros_like(acc)
    ok = ln > 1e-300
    out[ok] = acc[ok] / ln[ok][:, None]
    return out


def ray_hit_distance(p, d, V, F, t_eps=1e-9, bnd=1e-9):
    """p 沿 d 的最近三角命中距离 t（无命中返回 +inf）。

    Möller–Trumbore：barycentric 含边界容差 bnd（对角/棱上的退化命中
    不丢）；t > t_eps 跳过出发面上的共点命中（射线自表面顶点出发）。
    """
    p = np.asarray(p, float)
    d = np.asarray(d, float)
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if float(np.linalg.norm(d)) < 1e-300:
        return float("inf")
    best = float("inf")
    for a, b, c in F:
        A, B, C = V[a], V[b], V[c]
        E1 = B - A
        E2 = C - A
        P = np.cross(d, E2)
        det = float(E1 @ P)
        if abs(det) < 1e-300:
            continue                          # 射线与三角平行/共面
        inv = 1.0 / det
        T = p - A
        u = float(T @ P) * inv
        if not (-bnd <= u <= 1.0 + bnd):
            continue
        Q = np.cross(T, E1)
        v = float(d @ Q) * inv
        if not (-bnd <= v <= 1.0 + bnd) or u + v > 1.0 + bnd:
            continue
        t = float(E2 @ Q) * inv
        if t > t_eps and t < best:
            best = t
    return best


def _probe_inward(V, F):
    """探针判定绕向法向是否指向体内（需闭合表面）。

    取首个非退化面：质心 c、绕向面法向 n=(B−A)×(C−A)；c±εn 两个探针点经
    even-odd 判定一内一外 → 可定向；两点同类（开放表面）→ 返回 None。
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    eps = max(diag * 1e-6, 1e-12)
    for tri in F:
        A, B, C = V[tri[0]], V[tri[1]], V[tri[2]]
        n = np.cross(B - A, C - A)
        ln = float(np.linalg.norm(n))
        if ln < 1e-300:
            continue                          # 退化面
        n = n / ln
        c = (A + B + C) / 3.0
        pin = point_in_mesh(c + eps * n, V, F)
        pout = point_in_mesh(c - eps * n, V, F)
        if pin and not pout:
            return True                       # 绕向法向 = 内向
        if pout and not pin:
            return False                      # 绕向法向 = 外向
        return None                           # 开放表面，无法判定
    return None


# --------------------------------------------------------------- 边界层推进
def prism_layers(vertices, faces, n_layers=3, first_thickness=None,
                 stretch=1.5, gap_fraction=0.5, side="inward",
                 target_faces=None):
    """从边界面向体内/外推进 prism 边界层（含碰撞降层）。

    参数：
      n_layers：目标层数；first_thickness：首层厚度 t0（缺省 = 包围盒对角×1%）；
      stretch：层厚增长率 r（r=1 等厚）；gap_fraction：允许占据最近通道的比例；
      side："inward"/"outward"（闭合表面自动定向）或 "+n"/"-n"（沿/逆绕向法向，
        开放表面用）；target_faces：仅这些面索引出层（缺省全部，N5 尺寸控制复用）。

    返回 {ok, vertices, cells, levels, face_layers, normals, n_cells, n_full,
    n_reduced, n_collapsed, total_thickness, first_layer_thickness, quality}；
    空输入抛 ValueError。
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or len(V) == 0 or len(F) == 0:
        raise ValueError("prism_layers 需要非空表面 (vertices Nx3, faces Mx3)")
    N = len(V)
    n_layers = max(1, int(n_layers))
    if first_thickness is None:
        diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
        first_thickness = diag * 0.01
    t0 = max(float(first_thickness), 1e-12)
    r = float(stretch)

    # 累计层厚 d_k = t0·(r^k−1)/(r−1)（r=1 时 = k·t0）
    levels = []
    d = 0.0
    for k in range(n_layers):
        d += t0 * (r ** k)
        levels.append(d)

    nrm = vertex_normals(V, F)
    if side in ("inward", "outward"):
        inward_winding = _probe_inward(V, F)
        if inward_winding is None:
            raise ValueError("无法判定内外向（表面可能开放），请用 side='+n'/'-n'")
        along = (side == "inward") == inward_winding
    elif side in ("+n", "-n"):
        along = (side == "+n")
    else:
        raise ValueError("side 须为 inward/outward/+n/-n")
    dirs = nrm if along else -nrm

    # 每顶点允许深度 → 顶点层高 K_v（碰撞处理：超通道比例自动降层）
    K = np.zeros(N, dtype=int)
    for v in range(N):
        if float(np.linalg.norm(dirs[v])) < 1e-12:
            continue
        h = ray_hit_distance(V[v], dirs[v], V, F)
        if not np.isfinite(h):
            K[v] = n_layers                   # 开放方向无阻碍
            continue
        allowed = gap_fraction * h
        kv = 0
        for k in range(1, n_layers + 1):
            if levels[k - 1] <= allowed + 1e-12:
                kv = k
            else:
                break
        K[v] = kv

    # 提升点（按顶点共享）：level j 的顶点 = v + dirs[v]·d_j
    newV = [list(map(float, p)) for p in V]
    lift = {}
    for v in range(N):
        for k in range(1, int(K[v]) + 1):
            lift[(v, k)] = len(newV)
            newV.append((V[v] + dirs[v] * levels[k - 1]).tolist())

    tgt = set(int(t) for t in target_faces) if target_faces is not None \
        else set(range(len(F)))
    cells = []
    face_layers = []
    first_h = []
    for fi, (a, b, c) in enumerate(F):
        L = min(int(K[a]), int(K[b]), int(K[c])) if fi in tgt else 0
        face_layers.append(L)
        if L >= 1:
            a1, b1, c1 = lift[(a, 1)], lift[(b, 1)], lift[(c, 1)]
            bot = (V[a] + V[b] + V[c]) / 3.0
            top = (np.asarray(newV[a1]) + np.asarray(newV[b1])
                   + np.asarray(newV[c1])) / 3.0
            first_h.append(float(np.linalg.norm(top - bot)))
        for k in range(1, L + 1):
            a0 = a if k == 1 else lift[(a, k - 1)]
            b0 = b if k == 1 else lift[(b, k - 1)]
            c0 = c if k == 1 else lift[(c, k - 1)]
            cells.append([a0, b0, c0, lift[(a, k)], lift[(b, k)],
                          lift[(c, k)]])

    cells_arr = (np.asarray(cells, np.int64) if cells
                 else np.zeros((0, 6), np.int64))
    n_full = sum(1 for L in face_layers if L == n_layers)
    n_red = sum(1 for L in face_layers if 0 < L < n_layers)
    n_col = sum(1 for L in face_layers if L == 0)
    return {
        "ok": len(cells) > 0,
        "vertices": np.asarray(newV, float),
        "cells": cells_arr,
        "levels": levels,
        "face_layers": face_layers,
        "normals": dirs,
        "n_cells": len(cells),
        "n_full": n_full,
        "n_reduced": n_red,
        "n_collapsed": n_col,
        "total_thickness": levels[-1],
        "first_layer_thickness": (float(np.mean(first_h)) if first_h
                                  else 0.0),
        "quality": prism_quality(newV, cells_arr),
    }


# --------------------------------------------------------------- 质量
def _prism_signed_volumes(V, C):
    """棱柱有向体积（标准棋盘三分四面体分解，绕向一致时同号）。"""
    a0, b0, c0 = V[C[:, 0]], V[C[:, 1]], V[C[:, 2]]
    a1, b1, c1 = V[C[:, 3]], V[C[:, 4]], V[C[:, 5]]
    t1 = np.einsum("ij,ij->i", np.cross(b0 - a0, c0 - a0), a1 - a0)
    t2 = np.einsum("ij,ij->i", np.cross(c0 - b0, a1 - b0), b1 - b0)
    t3 = np.einsum("ij,ij->i", np.cross(a1 - c0, b1 - c0), c1 - c0)
    return (t1 + t2 + t3) / 6.0


def prism_quality(vertices, cells):
    """prism 单元质量统计：单元数/翻转数（符号少数派）/体积与符号一致性。"""
    V = np.asarray(vertices, float)
    C = np.asarray(cells, np.int64)
    if len(C) == 0:
        return {"n_cells": 0, "n_inverted": 0, "volume_total": 0.0,
                "volume_min": 0.0, "sign_consistent": True, "ok": False}
    signed = _prism_signed_volumes(V, C)
    vols = np.abs(signed)
    n_pos = int((signed > 1e-12).sum())
    n_neg = int((signed < -1e-12).sum())
    return {"n_cells": int(len(C)),
            "n_inverted": min(n_pos, n_neg),
            "volume_total": float(vols.sum()),
            "volume_min": float(vols.min()),
            "sign_consistent": n_pos == 0 or n_neg == 0,
            "ok": bool(vols.min() > 0.0)}
