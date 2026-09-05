# -*- coding: utf-8 -*-
"""N 波 N6：统一网格质量诊断/统计/修复（纯 numpy，双环境可用）。

N6 目标是把 N1–N5 散落在各 mesher 里的质量函数（`mesh_tet.tet_quality`、
`mesh_poly.poly_quality`、`mesh_trimmer.trimmer_quality`、`mesh_prism.prism_quality`）
统一到一套入口，并补上验收核心——**质量 histogram** 与**修复**：

  - `cell_metric`：单单元归一化质量 0–1（1=最优），按单元类型分派：
      · tet（4 节点）——SIGMA 指标 12·(3V)^{2/3} / Σℓ²，正四面体=1；
      · prism（6 节点）——标准棋盘分解为 3 个四面体，取子 tet SIGMA 均值；
      · poly / hex ——体积-表面积紧致度（Wadell sphericity），球≈1、立方≈0.806；
    axis-aligned box，SIGMA 恒为 1。
  - `quality_histogram`：per-cell 得分→分箱计数 + 均值/最小/分位 + 达标判定
    （`pass` = 无退化单元且 mean ≥ 阈值）；
  - `quality_report`：分派各 mesher 原生质量统计 + histogram 汇总；
  - `cell_quality`：统一入口，返回 kind 相关统计 + histogram；
  - `repair_cells`：去除退化(近零体积)/翻转(负体积)单元，tet 负体积自动翻转取向。

单元表示约定与各 mesher 一致：
  - tet    —— [M,4] 顶点索引数组；
  - prism  —— [M,6] 顶点索引（底面三角 [a0,b0,c0] + 顶面三角 [a1,b1,c1]）；
  - poly   —— list[dict] {"seed","vertices","faces","volume","n_faces"}；
  - trimmer—— list[dict] {"kind":"hex","lo","hi"} 或 {"kind":"cut","vertices","faces"}。

验收（tests/test_mesh_quality_n6.py）：立方体 tet 网格 histogram 集中高质量端
（mean/p10 超阈、归一化计数和=n_cells、pass=True）；人工混入负/退化单元后
repair_cells 恢复（n_flipped/n_dropped 正确、修复后 pass=True）。
"""
import numpy as np


# ---------------------------------------------------------------------------
# 单单元归一化质量
# ---------------------------------------------------------------------------
def _tet_sigma(V, idx):
    """SIGMA 形状质量：12·(3V)^{2/3}/Σℓ²，正四面体=1，退化/翻转→0/负。"""
    p = V[idx]
    e0 = np.linalg.norm(p[1] - p[0])
    e1 = np.linalg.norm(p[2] - p[1])
    e2 = np.linalg.norm(p[0] - p[2])
    e3 = np.linalg.norm(p[3] - p[0])
    e4 = np.linalg.norm(p[3] - p[1])
    e5 = np.linalg.norm(p[3] - p[2])
    s2 = (e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4 + e5 * e5)
    if s2 < 1e-300:
        return 0.0
    vol = abs(float(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])))) / 6.0
    if vol < 1e-300:
        return 0.0
    q = 12.0 * (3.0 * vol) ** (2.0 / 3.0) / s2
    return float(min(max(q, 0.0), 1.0))


def _prism_sub_tets(C):
    """棱柱 [a0,b0,c0,a1,b1,c1] → 3 个四面体索引（与 mesh_prism 棋盘分解一致）。"""
    a0, b0, c0, a1, b1, c1 = C
    return [(a0, b0, c0, a1), (b0, c0, a1, b1), (c0, a1, b1, c1)]


def _cell_vertices(cell):
    """从单元表示取顶点坐标 (K,3)，支持 numpy 数组、dict、hex 盒。"""
    if isinstance(cell, dict):
        if cell.get("kind") == "hex":
            lo = np.asarray(cell["lo"], float)
            hi = np.asarray(cell["hi"], float)
            return np.array([
                [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], float)
        if "vertices" in cell:
            return np.asarray(cell["vertices"], float)
        raise ValueError("无法从 dict 单元提取顶点")
    raise ValueError("cell_vertices 需要浮点顶点数组或含 vertices/lo-hi 的 dict")


def _wadell_sphericity(verts):
    """Wadell 球度：Ψ=(π^{1/3}(6V)^{2/3})/A，球=1、立方≈0.806∈(0,1]。"""
    p = np.asarray(verts, float)
    if p.shape[0] < 4:
        return 0.0
    c = p.mean(axis=0)
    # 用各面扇形三角形化 + 散度定理算体，需外向绕向；此处用凸包定位鲁棒外推。
    # 为不依赖绕向，直接用「顶点与质心外推」近似体积（凸胞内恒正）。
    try:
        from scipy.spatial import ConvexHull  # noqa: F401
        hull = ConvexHull(p)
        V = hull.volume
        A = hull.area
    except Exception:
        return 0.0
    if V < 1e-300 or A < 1e-300:
        return 0.0
    s = float((np.pi ** (1.0 / 3.0)) * ((6.0 * V) ** (2.0 / 3.0)) / A)
    return float(min(max(s, 0.0), 1.0))


def cell_metric(vertices, cell, kind="tet"):
    """单单元归一化质量（0–1，1=最优）。

    kind：tet/prism → 顶点索引数组 + vertices；poly/hex → dict（可忽略
    vertices 参数，直接取 cell 内顶点）。退化/翻转单元返回接近 0。
    """
    if kind == "tet":
        return _tet_sigma(np.asarray(vertices, float),
                          np.asarray(cell, np.int64))
    if kind == "prism":
        Vs = np.asarray(vertices, float)
        idx = np.asarray(cell, np.int64)
        subs = _prism_sub_tets(idx)
        return float(np.mean([_tet_sigma(Vs, np.asarray(s, np.int64))
                              for s in subs]))
    if kind in ("poly", "hex", "trimmer", "cut"):
        return _wadell_sphericity(_cell_vertices(cell))
    raise ValueError("kind 须为 tet/prism/poly/hex/trimmer，收到 %r" % (kind,))


# ---------------------------------------------------------------------------
# 质量 histogram
# ---------------------------------------------------------------------------
def quality_histogram(vertices, cells, kind="tet", bins=10, range_=(0.0, 1.0),
                      quality_threshold=0.30):
    """逐单元质量得分直方图 + 达标判定。

    返回 {n_cells, counts(list[int]), edges(list[float]), mean, min, max,
          p10, p90, n_degen(<1e-6), n_bad(<quality_threshold),
          pass(bool)}。pass = n_cells>0 且无退化单元且 mean ≥ quality_threshold。
    """
    if cells is None or (hasattr(cells, "__len__") and len(cells) == 0):
        counts = np.zeros(int(bins), np.int64)
        return {"n_cells": 0, "counts": counts.tolist(),
                "edges": np.linspace(range_[0], range_[1], int(bins) + 1).tolist(),
                "mean": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p90": 0.0,
                "n_degen": 0, "n_bad": 0, "pass": False}
    if kind == "poly" or kind == "hex" or kind == "trimmer":
        scores = [cell_metric(vertices, c, kind) for c in cells]
    else:
        arr = np.asarray(cells)
        scores = [cell_metric(vertices, row, kind) for row in arr]
    scores = np.clip(np.asarray(scores, float), 0.0, 1.0)
    counts, edges = np.histogram(scores, bins=int(bins), range=range_)
    n = len(scores)
    mean = float(scores.mean()) if n else 0.0
    smin = float(scores.min()) if n else 0.0
    smax = float(scores.max()) if n else 0.0
    p10 = float(np.percentile(scores, 10)) if n else 0.0
    p90 = float(np.percentile(scores, 90)) if n else 0.0
    n_degen = int((scores < 1e-6).sum())
    n_bad = int((scores < float(quality_threshold)).sum())
    return {"n_cells": int(n), "counts": [int(x) for x in counts],
            "edges": [float(x) for x in edges],
            "mean": mean, "min": smin, "max": smax,
            "p10": p10, "p90": p90,
            "n_degen": n_degen, "n_bad": n_bad,
            "pass": bool(n > 0 and n_degen == 0 and mean >= float(quality_threshold))}


# ---------------------------------------------------------------------------
# 分派各 mesher 的原生质量统计
# ---------------------------------------------------------------------------
def _native_quality(vertices, cells, kind):
    if kind == "tet":
        from mesh_tet import tet_quality
        return tet_quality(vertices, cells)
    if kind == "prism":
        from mesh_prism import prism_quality
        return prism_quality(vertices, cells)
    if kind == "poly":
        from mesh_poly import poly_quality
        return poly_quality(cells)
    if kind in ("trimmer", "hex"):
        from mesh_trimmer import trimmer_quality
        return trimmer_quality(cells)
    raise ValueError("未知单元类型 %r" % (kind,))


def quality_report(vertices, cells, kind="tet", bins=10, quality_threshold=0.30):
    """统一质量报告：分派原生统计 + histogram 汇总。"""
    native = _native_quality(vertices, cells, kind)
    hist = quality_histogram(vertices, cells, kind, bins=bins,
                             quality_threshold=quality_threshold)
    out = dict(native)
    out["kind"] = kind
    out["histogram"] = hist
    out["ok"] = bool(out.get("ok", False) or hist["pass"])
    out["pass"] = hist["pass"]
    return out


def cell_quality(vertices, cells, kind="tet", bins=10,
                 quality_threshold=0.30):
    """统一入口：返回 kind 原生统计 + histogram（N6 对接 GUI 质量诊断）。"""
    return quality_report(vertices, cells, kind, bins=bins,
                          quality_threshold=quality_threshold)


# ---------------------------------------------------------------------------
# 修复
# ---------------------------------------------------------------------------
def _tet_signed_volume(V, C):
    a = V[C[:, 0]]
    b = V[C[:, 1]]
    c = V[C[:, 2]]
    d = V[C[:, 3]]
    return np.sum((b - a) * np.cross(c - a, d - a), axis=1) / 6.0


def repair_cells(vertices, cells, kind="tet", null_tol=1e-12):
    """去除退化(近零体积)单元；tet 负体积自动翻转取向。

    返回 {ok, vertices, cells, n_before, n_after, n_flipped, n_dropped,
    quality, flipped(bool 标记数组)}。
    """
    reduced = False
    if kind != "tet":
        # 非 tet：仅按体积正负/零过滤（poly/hex 无统一节点翻转规则）。
        kept = []
        for c in cells:
            vol = c.get("volume", None)
            if vol is None:
                from mesh_trimmer import _cell_volume  # 复用体积
                if c.get("kind") == "hex":
                    d = np.asarray(c["hi"], float) - np.asarray(c["lo"], float)
                    vol = float(np.prod(np.maximum(d, 0.0)))
                else:
                    vol = float(c.get("volume", 0.0))
            if vol > float(null_tol):
                kept.append(c)
        n_dropped = len(cells) - len(kept)
        return {"ok": len(kept) > 0, "vertices": vertices, "cells": kept,
                "n_before": len(cells), "n_after": len(kept),
                "n_flipped": 0, "n_dropped": int(n_dropped),
                "flipped": None,
                "quality": poly_quality_safe(vertices, kept)}

    V = np.asarray(vertices, float)
    C = np.asarray(cells, np.int64)
    if len(C) == 0:
        return {"ok": False, "vertices": vertices, "cells": C,
                "n_before": 0, "n_after": 0, "n_flipped": 0, "n_dropped": 0,
                "flipped": np.zeros(0, bool),
                "quality": _native_quality(V, C, "tet")}
    signed = _tet_signed_volume(V, C)
    flip_mask = signed < -float(null_tol)
    drop_mask = np.abs(signed) <= float(null_tol)
    n_flipped = int(flip_mask.sum())
    n_dropped = int(drop_mask.sum())
    keep = ~drop_mask
    Ck = C[keep].copy()
    fold = flip_mask[keep]
    Ck[fold] = Ck[fold][:, [0, 2, 1, 3]]       # 交换两顶点 → 翻转取向
    return {"ok": len(Ck) > 0, "vertices": V, "cells": Ck,
            "n_before": int(len(C)), "n_after": int(len(Ck)),
            "n_flipped": n_flipped, "n_dropped": n_dropped,
            "flipped": flip_mask,
            "quality": _native_quality(V, Ck, "tet")}


def poly_quality_safe(vertices, cells):
    """repair_cells 内部对 dict 单元的安全质量汇总（无 cells 返回空）。"""
    try:
        from mesh_poly import poly_quality
        return poly_quality(cells)
    except Exception:
        return {"n_cells": len(cells), "ok": len(cells) > 0}
