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
# S4：协调（无悬挂节点）细化 + 场传递 —— AMR 入求解环的前置条件
# ---------------------------------------------------------------------------
def _cell_faces(cells):
    """(排序顶点三元组) → 共享该面的单元索引列表。"""
    faces = {}
    for ci, row in enumerate(cells):
        r = [int(x) for x in row]
        for tri in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            k = tuple(sorted((r[tri[0]], r[tri[1]], r[tri[2]])))
            faces.setdefault(k, []).append(ci)
    return faces


def mesh_conformity(vertices, cells):
    """协调性硬判据：面最多被两个单元共享，且 count==1 的"单侧面"构成闭合曲面。

    1:8 红细化若邻接未细化单元会产生悬挂节点：粗面(3 顶点)与 4 个细分面同时
    只出现一次，粗面的边只被 1 个单侧面使用 → n_open_edges>0。FVM 的面匹配按
    顶点集合配对，悬挂界面会被误判为边界（内部连通丢失），所以细化结果入环前
    必须过这道门槛。返回 {conforming, n_faces, n_boundary, n_dup_faces,
    n_open_edges, n_vertices}。
    """
    V = np.asarray(vertices, float)
    faces = _cell_faces(cells)
    bnd = [k for k, cs in faces.items() if len(cs) == 1]
    dup = [k for k, cs in faces.items() if len(cs) > 2]
    # ① 无向边键规范化后统计使用次数（(9,0) 与 (0,9) 是同一条边）
    use, edge_faces = {}, {}
    for k in bnd:
        a, b, c = k
        for e in ((a, b), (b, c), (c, a)):
            ek = e if e[0] < e[1] else (e[1], e[0])
            use[ek] = use.get(ek, 0) + 1
            edge_faces.setdefault(ek, []).append(k)
    bad = [e for e, n in use.items() if n != 2]
    # ② T 型节点（悬挂节点）检测：某条单侧面的边的**内部**存在别的顶点
    #    （该顶点被共用这条边的其它单侧面使用）。角落单元的孤立细化正是这种
    #    形态：粗面退化边被真实边界边掩盖，只能靠 T 型检测抓到。
    hanging = []
    scale = max(1.0, float(np.abs(V).max()) if V.size else 1.0)
    tol = 1e-9 * scale
    for ek, ks in edge_faces.items():
        a, b = int(ek[0]), int(ek[1])
        d = V[b] - V[a]
        L2 = float(d @ d)
        if L2 <= 0.0:
            continue
        cand = {int(v) for k in ks for v in k} - {a, b}
        for v in cand:
            t = float((V[v] - V[a]) @ d) / L2
            if not (1e-9 < t < 1.0 - 1e-9):
                continue
            if float(np.linalg.norm(V[v] - (V[a] + t * d))) <= tol:
                hanging.append((ek, v))
                break
    return {"conforming": (not bad) and (not dup) and (not hanging),
            "n_faces": len(faces), "n_boundary": len(bnd),
            "n_dup_faces": len(dup), "n_open_edges": len(bad),
            "n_hanging_edges": len(hanging),
            "n_vertices": int(len(V))}


def refine_conformity(mask, cells):
    """红细化的协调性**精确前置判据**：任何被两个单元共享的面，其两侧要么
    同细化、要么同不细化；且网格本身面配对合法（无 >2 共享面）。

    1:8 红细化在父单元内部是协调的，因此"无混合面"等价于细化后协调 —— 这是
    比事后几何检测更强、也更便宜的判据（O(面数)）。返回 {conforming,
    n_mixed_faces, n_dup_faces, n_faces, n_masked}。
    """
    faces = _cell_faces(cells)
    mask = np.asarray(mask, bool)
    mixed = [k for k, cs in faces.items()
             if len(cs) == 2 and bool(mask[cs[0]]) != bool(mask[cs[1]])]
    dup = [k for k, cs in faces.items() if len(cs) > 2]
    return {"conforming": (not mixed) and (not dup),
            "n_mixed_faces": len(mixed), "n_dup_faces": len(dup),
            "n_faces": len(faces), "n_masked": int(mask.sum())}


def refine_tets_conforming(vertices, cells, mask=None, threshold=None,
                           max_rounds=8, verify=True):
    """协调 1:8 红细化：**边闭包**（标记单元每条边上的单元一并细化）+ 父子映射。

    规则：任一被标记单元的每条边，其上所有单元都必须细化（标准 edge-closure），
    迭代至多 max_rounds 轮；随后细分并复核：`conforming` = 红细化前置判据
    （`refine_conformity`，精确）**且** 结果网格结构诊断（`mesh_conformity`：
    重复面 / 开口边 / T 型节点）皆通过。任一不过都如实返回 conforming=False ——
    调用方应**跳过**本次细化，而不是把非协调网格交给求解器。

    返回 {ok, vertices, cells, parent, n_before, n_after, n_refined, n_closed,
    rounds, conforming, conformity}；`parent[k]` = 新单元 k 的**旧单元索引**
    （未细化单元指向自身），供 transfer_fields 场传递使用。
    """
    V = np.asarray(vertices, float)
    C = np.asarray(cells, np.int64)
    if len(C) == 0:
        raise ValueError("refine_tets_conforming 需要非空四面体")
    if mask is None:
        mask = np.ones(len(C), bool)
    else:
        mask = np.asarray(mask, bool).copy()
    if threshold is not None:
        scores = np.array([cell_metric(V, row, "tet") for row in C])
        mask = scores < float(threshold)
    edge_cells = {}
    for ci, row in enumerate(C):
        r = [int(x) for x in row]
        for i in range(4):
            for j in range(i + 1, 4):
                k = (r[i], r[j]) if r[i] < r[j] else (r[j], r[i])
                edge_cells.setdefault(k, []).append(ci)
    rounds, n_closed = 0, 0
    for _ in range(max(1, int(max_rounds))):
        grow = set()
        for _k, cs in edge_cells.items():
            if any(mask[c] for c in cs):
                for c in cs:
                    if not mask[c]:
                        grow.add(c)
        if not grow:
            break
        for c in grow:
            mask[c] = True
        n_closed += len(grow)
        rounds += 1
    ref = refine_tets(V, C, mask=mask)
    parents = [ci for ci in range(len(C)) if not mask[ci]]
    parents += [ci for ci in range(len(C)) if mask[ci] for _ in range(8)]
    parent = np.asarray(parents, np.int64)
    if parent.size != len(ref["cells"]):
        raise ValueError("父子映射长度 %d != 新单元数 %d"
                         % (parent.size, len(ref["cells"])))
    pre = refine_conformity(mask, C)
    mesh_conf = mesh_conformity(ref["vertices"], ref["cells"]) if verify else None
    conforming = bool(pre["conforming"]) and (mesh_conf is None
                                               or bool(mesh_conf["conforming"]))
    return {"ok": bool(ref["ok"]), "vertices": ref["vertices"],
            "cells": ref["cells"], "parent": parent,
            "n_before": int(len(C)), "n_after": int(len(ref["cells"])),
            "n_refined": int(mask.sum()), "n_closed": int(n_closed),
            "rounds": int(rounds),
            "conforming": conforming,
            "conformity": pre, "mesh_conformity": mesh_conf}


def transfer_fields(parent, values):
    """AMR 场传递（注入式）：新网格单元值 = 其父单元值（`parent` 来自
    `refine_tets_conforming`）。

    常数场逐点精确；1:8 红细分的 8 个子单元恰好填满父单元，故体积加权均值
    ∑φV 保持不变（同一父单元的子单元同值，∑φV_new = ∑_parent φ_parent V_parent）。
    标量 (n,) 与矢量 (n,3) 均可。
    """
    V = np.asarray(values)
    return V[np.asarray(parent, np.int64)]


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
