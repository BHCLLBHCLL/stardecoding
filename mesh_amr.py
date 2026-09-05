# -*- coding: utf-8 -*-
"""N 波 N6：AMR 运行时细化钩子（质量驱动，P 波联调预留）。

AMR 在 N6 落地**运行时钩子**（标记 + 回调注册），真正的求解循环集成放到
P 波；本模块提供求解器可消费的三件套：

  - `amr_marks`：质量驱动的细化 **标记**——逐单元 `mesh_quality.cell_metric`
    得分 < threshold（或低于指定百分位）→ 布尔掩码，并给出份额/直方图。这是
    运行时触发的核心判断器；
  - `refine_tets`：把标记的四面体做 1:8「红细化」（六条边中点 + 中央八面体
    沿对边对角线拆 4）——边中点按边键去重（标记集内共享边一致），邻居未标记
    面产生悬挂节点（非协调，P 波做 green 闭合时消费 `hanging` 统计）；
  - `register_amr_hook` / `run_amr`：回调注册表——求解循环每步 `run_amr(
    vertices, cells, threshold)` 计算标记并分发到已注册钩子，返回汇总
    （n_cells/n_marks/fraction/直方图）。P 波把细化算子注册进来即可联调。

用法：
    register_amr_hook(refine_tets)          # 把细化算子挂进运行时
    out = run_amr(V, C, kind="tet", threshold=0.30)
"""
import numpy as np

from mesh_quality import cell_metric, quality_histogram


# ---------------------------------------------------------------------------
# 质量驱动标记
# ---------------------------------------------------------------------------
def amr_marks(vertices, cells, kind="tet", threshold=0.30,
              percentile=None, bins=10):
    """质量驱动细化标记：返回 score<threshold（或低于 percentile 分位）的单元。

    返回 {mask, n_cells, n_marks, fraction, mean, histogram}；mask 为 bool 数组
    标记待细化单元。percentile 给定时按较劣比例标记覆盖 threshold（二选一）。
    """
    if cells is None or (hasattr(cells, "__len__") and len(cells) == 0):
        n = 0
        mask = np.zeros(0, bool)
        return {"mask": mask, "n_cells": 0, "n_marks": 0, "fraction": 0.0,
                "mean": 0.0,
                "histogram": quality_histogram(vertices, cells, kind, bins=bins)}
    if kind in ("poly", "hex", "trimmer"):
        scores = np.array([cell_metric(vertices, c, kind) for c in cells])
    else:
        arr = np.asarray(cells)
        scores = np.array([cell_metric(vertices, row, kind) for row in arr])
    scores = np.clip(scores, 0.0, 1.0)
    n = len(scores)
    if percentile is not None:
        cut = float(np.percentile(scores, percentile))
        mask = scores <= cut if percentile < 50 else scores < cut
    else:
        mask = scores < float(threshold)
    return {"mask": mask, "n_cells": n, "n_marks": int(mask.sum()),
            "fraction": float(mask.mean()) if n else 0.0,
            "mean": float(scores.mean()) if n else 0.0,
            "histogram": quality_histogram(vertices, cells, kind, bins=bins,
                                           quality_threshold=float(threshold))}


# ---------------------------------------------------------------------------
# 1:8 红细化（tet 基元，非协调）
# ---------------------------------------------------------------------------
def _edge_mp(Vx, a, b, cache, vlist):
    """共享边中点：a<b 规范化边键去重，返回新顶点索引。"""
    key = (a, b) if a < b else (b, a)
    if key not in cache:
        cache[key] = len(vlist)
        vlist.append(((Vx[a] + Vx[b]) * 0.5).tolist())
    return cache[key]


def refine_tets(vertices, cells, mask=None, threshold=None):
    """标记四面体 1:8 红细化（非协调；邻居未标记产生悬挂节点）。

    返回 {ok, vertices, cells, n_before, n_after, n_refined, n_hanging,
    hanging(bool)}。mask 为 bool 数组（None 表示全部细化；threshold 表示质量
    低于阈值的单元细化）。空输入抛 ValueError。
    """
    V = np.asarray(vertices, float)
    C = np.asarray(cells, np.int64)
    if len(C) == 0:
        raise ValueError("refine_tets 需要非空四面体")
    if mask is None:
        mask = np.ones(len(C), bool)
    else:
        mask = np.asarray(mask, bool)
    if threshold is not None:
        scores = np.array([cell_metric(V, row, "tet") for row in C])
        mask = scores < float(threshold)

    vlist = [p.tolist() for p in V]
    cache = {}
    keep_rows = []
    for row in C[~mask]:
        keep_rows.append([int(x) for x in row])

    for row in C[mask]:
        n0, n1, n2, n3 = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        e01 = _edge_mp(V, n0, n1, cache, vlist)
        e02 = _edge_mp(V, n0, n2, cache, vlist)
        e03 = _edge_mp(V, n0, n3, cache, vlist)
        e12 = _edge_mp(V, n1, n2, cache, vlist)
        e13 = _edge_mp(V, n1, n3, cache, vlist)
        e23 = _edge_mp(V, n2, n3, cache, vlist)
        # 4 角 tets
        keep_rows += [[n0, e01, e02, e03], [n1, e01, e12, e13],
                      [n2, e02, e12, e23], [n3, e03, e13, e23]]
        # 中央八面体沿 e01–e23 对角线拆 4
        keep_rows += [[e01, e02, e12, e23], [e01, e12, e13, e23],
                      [e01, e13, e03, e23], [e01, e03, e02, e23]]

    newV = np.asarray(vlist, float)
    newC = np.asarray(keep_rows, np.int64)
    split = int(mask.sum())
    # 悬挂节点计数：未细化单元共享被细化单元新建的中点边（非协调界面）
    hang_count = 0
    for row in C[~mask]:
        r = [int(x) for x in row]
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = r[i], r[j]
                key = (a, b) if a < b else (b, a)
                if key in cache:
                    hang_count += 1
    return {"ok": len(newC) > 0, "vertices": newV, "cells": newC,
            "n_before": int(len(C)), "n_after": int(len(newC)),
            "n_refined": split,
            "n_hanging": hang_count,
            "hanging": None}


# ---------------------------------------------------------------------------
# 运行时回调注册表（P 波联调入口）
# ---------------------------------------------------------------------------
_hooks = []


def register_amr_hook(fn):
    """注册 AMR 细化钩子（P 波求解循环调用 run_amr 时被分派）。"""
    if callable(fn) and fn not in _hooks:
        _hooks.append(fn)
    return fn


def unregister_amr_hook(fn):
    """从注册表移除钩子。"""
    if fn in _hooks:
        _hooks.remove(fn)


def run_amr(vertices, cells, kind="tet", threshold=0.30, percentile=None):
    """运行时 AMR：计算质量标记 → 分派钩子 → 返回汇总。

    返回 {n_cells, n_marks, fraction, mean, histogram, hooks_called,
    results}；无注册钩子时仅返回标记统计（hooks_called=0）。
    """
    marks = amr_marks(vertices, cells, kind=kind, threshold=threshold,
                      percentile=percentile)
    results = []
    for fn in list(_hooks):
        try:
            res = fn(vertices, cells, marks["mask"])
            results.append({"hook": getattr(fn, "__name__", repr(fn)),
                            "result": res})
        except Exception as e:  # 单钩子失败不中断其余
            results.append({"hook": getattr(fn, "__name__", repr(fn)),
                            "error": str(e)})
    return {"n_cells": marks["n_cells"], "n_marks": marks["n_marks"],
            "fraction": marks["fraction"], "mean": marks["mean"],
            "histogram": marks["histogram"],
            "hooks_called": len(results), "results": results}
