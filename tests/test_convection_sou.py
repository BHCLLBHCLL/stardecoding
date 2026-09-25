# -*- coding: utf-8 -*-
"""S2 第十一步：隐式二阶上风（`convection="upwind2"`）的验证。

动机（第八/九/十步实测）：二阶信息若只作**显式延迟修正**（central/limited），其有效性依赖
内迭代把二阶项迭代出来 —— 实测耗散随 ni 只在 1→2 有改善（−1.71→−0.88/s）后饱和（−0.77/−0.95），
而官方求解器在同一张网格上不衰减（St=0.1689）。故把二阶上风做成**真正的隐式矩阵**：
面值 φ_f = 1.5φ_U − 0.5φ_UU（U=上风、UU=二阶上风），系数直接进矩阵。

本文件验证：① UU 表满足定义（上风单元邻居中排除下游、**严格位于上游**且最靠前者）；
② 入口侧无合格候选 → 退化 −1（该面回到一阶上风）；③ 选项确实改变矩阵且装配系统适定可解；
④ 小网格瞬态可跑不发散。（大算例 S2 验证待机器空闲 —— 见 s_wave_plan 待执行清单。）
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from official_diff import channel_tet_mesh_cartesian  # noqa: E402
from pressure_solver import PressureSolver, solve_linear  # noqa: E402


def _mesh():
    return channel_tet_mesh_cartesian(0.04, length_D=8.0, height_D=4.0,
                                      thickness_D=0.5, h_factor=2.0)


def _solver(mesh, convection="upwind2"):
    return PressureSolver(np.asarray(mesh["vertices"], float),
                          np.asarray(mesh["cells"], np.int64), rho=1.0, mu=1e-5,
                          inlet_axis=0, inlet_side="min",
                          inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
                          convection=convection)


def _adjacency(fv):
    o = fv.owner[fv.neighbor >= 0]
    nb = fv.neighbor[fv.neighbor >= 0]
    adj = {}
    for a, b in zip(o.tolist(), nb.tolist()):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def test_second_upwind_satisfies_argmin_property():
    """UU 必须是"上风单元 U 的邻居中（排除下游 D）、严格位于上游且最靠前者"。"""
    mesh = _mesh()
    s = _solver(mesh)
    fidx, uu_pos, uu_neg = s._second_upwind_tables()
    fv = s.fvm
    cen = np.asarray(fv.centroids, float)
    o = fv.owner[fidx]
    nb = fv.neighbor[fidx]
    d = cen[nb] - cen[o]
    L = np.linalg.norm(d, axis=1)
    dhat = d / np.where(L > 0, L, 1.0)[:, None]
    adj = _adjacency(fv)
    checked = 0
    for i in range(0, fidx.size, max(1, fidx.size // 300)):
        for anchor, other, uu, direction in (
                (int(o[i]), int(nb[i]), int(uu_pos[i]), dhat[i]),
                (int(nb[i]), int(o[i]), int(uu_neg[i]), -dhat[i])):
            projs = {x: float((cen[x] - cen[anchor]) @ direction)
                     for x in adj.get(anchor, ()) if x != other}
            neg = {x: v for x, v in projs.items() if v < 0.0}
            if not neg:
                assert uu == -1, "无上游候选时应退化 −1，实得 %d" % uu
            else:
                # 允许**精确并列**（对称构型下多个候选项投影完全相同，取任一都合法）
                best = min(neg.values())
                assert uu in neg and abs(projs[uu] - best) < 1e-12, \
                    "UU 不是最靠上游的合格候选（face %d，proj %.3e vs best %.3e）" \
                    % (i, projs.get(uu, float("nan")), best)
            checked += 1
    assert checked > 100, checked


def test_second_upwind_falls_back_when_no_upstream_candidate():
    """最小合成网格（两个四面体、各只有一个邻居）→ 无上游候选，必须退化 −1。

    实测教训：在 tet 网格上"入口侧就有 −1"的假设是错的 —— 同一 hex 拆出的 tet
    互相之间也有沿流向为负的邻居（对角 tet），因此退化分支要靠**单邻居/极端拓扑**
    才能触发。这里用显式构造的最小网格把该分支钉死。
    """
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], float)
    C = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], np.int64)   # 共享面 (0,1,2)
    s = PressureSolver(V, C, rho=1.0, mu=1e-5, inlet_axis=0, inlet_side="min",
                       inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
                       convection="upwind2")
    fidx, uu_pos, uu_neg = s._second_upwind_tables()
    assert fidx.size == 1, "两 tet 应只有一个内部面"
    assert uu_pos[0] == -1 and uu_neg[0] == -1, \
        "单邻居单元无上游候选，UU 必须为 −1（实得 %d/%d）" % (uu_pos[0], uu_neg[0])


def test_upwind2_changes_matrix_and_is_well_posed():
    """选项必须真正改矩阵；对角为正；装配出的系统可解且残差小。"""
    mesh = _mesh()
    s1 = _solver(mesh, "upwind")
    s2 = _solver(mesh, "upwind2")
    r1, c1, v1, _b1, ap1 = s1._assemble_momentum(0)
    r2, c2, v2, b2, ap2 = s2._assemble_momentum(0)
    n = s1.fvm.n_cells
    assert (r1.size != r2.size) or not np.allclose(np.sort(v1), np.sort(v2)), \
        "upwind2 未改变矩阵"
    diag = np.bincount(r2[r2 == c2], weights=v2[r2 == c2], minlength=n)
    assert (diag > 0).all(), "存在非正对角"
    assert np.isfinite(ap1).all() and np.isfinite(ap2).all()
    x = solve_linear(r2, c2, v2, b2, n, tol=1e-8, system="convection")
    assert np.isfinite(x).all()
    ax = np.bincount(r2, weights=v2 * x[c2], minlength=n)
    res = float(np.linalg.norm(ax - b2) / max(np.linalg.norm(b2), 1e-300))
    assert res < 1e-4, "upwind2 装配系统未解出（rel resid %.2e）" % res


def test_upwind2_transient_runs_stable_on_small_hybrid():
    """小 hybrid 网格上跑几步瞬态：有限、有界、不发散。"""
    from mesh_hybrid import hybrid_channel_mesh
    m = hybrid_channel_mesh(D=0.04, length_D=10.0, height_D=6.0, thickness_D=0.25,
                            center_x_D=3.0, h_factor=2.0, m=8, n_r=6, a_D=2.0,
                            stretch=1.3)
    s = PressureSolver(np.asarray(m["vertices"], float),
                       np.asarray(m["cells"], np.int64), rho=1.0, mu=1e-5,
                       inlet_axis=0, inlet_side="min",
                       inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
                       convection="upwind2", wall_slip_axes=(1, 2),
                       piso_correctors=1, nonorth_corrected=True, corr_limit=0.33)
    s.enable_transient(0.01, snapshot=True)
    r = None
    for _ in range(8):
        r = s.advance(dt=0.01, n_inner=1)
    vel = s.velocity()
    assert np.isfinite(vel).all()
    assert float(np.linalg.norm(vel, axis=1).max()) < 1.0
    assert float(r["residual"]) < 1.0