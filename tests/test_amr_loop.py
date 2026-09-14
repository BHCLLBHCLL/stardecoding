# -*- coding: utf-8 -*-
"""S 波 S4：AMR 入求解环 —— 协调细化（边闭包）+ 守恒场传递。

背景（本轮实测）：AMR 原实现（N6 钩子）在 `run_amr` 返回新网格后直接
`solver.set_mesh(...)` + `_initialize_field()` —— **把已发展的解清零**；且
`refine_tets` 是**非协调**红细化（邻接未细化单元处产生悬挂节点）。FVM 按顶点
集合配对单元面，悬挂界面会被误判为边界（内部连通丢失），所以非协调网格不能
进求解器。

本轮落地：
  - `mesh_conformity`：面最多被 2 单元共享 + count==1 的"单侧面"构成闭合曲面；
  - `refine_tets_conforming`：边闭包（标记单元每条边上的单元一并细化，迭代至多
    max_rounds 轮）→ 1:8 红细化 → 复核；不协调则如实返回 conforming=False；
  - `transfer_fields`：按父子映射注入式传场（常数场精确、体积加权均值保持）；
  - `PressureSolver.set_fields`：直接设定 u/v/w/p 并可重取 φⁿ 参考；
  - `SolverBackend.set_amr`：协调细化 + 场传递；无传递能力的求解器保持 N6 行为
    但如实记录 note；未协调时跳过并记 skipped。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fvm_core import cube_tet_mesh  # noqa: E402
from mesh_amr import (amr_marks, mesh_conformity, refine_conformity,  # noqa: E402
                      refine_tets, refine_tets_conforming, transfer_fields)
from mesh_tet import _tet_volumes  # noqa: E402
from pressure_solver import PressureSolver  # noqa: E402
from solver_run import (DemoDiffusionSolver, SolverBackend,  # noqa: E402
                        demo_mesh)


def _vols(V, C):
    return np.abs(_tet_volumes(np.asarray(V, float), np.asarray(C, np.int64)))


# ---------------- 协调性判据 ----------------

def test_mesh_conformity_kuhn_mesh_is_conforming():
    """Kuhn 6-tet 结构化网格：面全配对，单侧面恰好是 6 个外表面的三角。"""
    for nx in (2, 3):
        V, C = cube_tet_mesh(nx)
        conf = mesh_conformity(V, C)
        assert conf["conforming"] is True, conf
        assert conf["n_dup_faces"] == 0 and conf["n_open_edges"] == 0
        assert conf["n_boundary"] == 6 * nx * nx * 2   # 每面 nx^2 个四边形 → 2 三角
        assert conf["n_faces"] == 4 * len(C) - 2 * len(C) + conf["n_boundary"] * 0 \
            or conf["n_faces"] > conf["n_boundary"]


def test_mesh_conformity_flags_five_tet_demo_mesh():
    """`demo_mesh` 是 5-tet 扇形测试基元（体积和 5/6、跨 hex 面三角剖分不重合）→
    判据必须判为非协调（这正是它不能进 AMR 入环路径的原因）。"""
    V, C = demo_mesh(nx=3)
    conf = mesh_conformity(V, C)
    assert conf["conforming"] is False
    assert conf["n_open_edges"] > 0
    assert conf["n_dup_faces"] == 0            # 无重复面：是"对不上"而非"重叠"
    assert conf["n_boundary"] > 6 * 3 * 3 * 2  # 单侧面多于真实外表面（含内部悬挂面）


def test_plain_refinement_of_one_cell_is_nonconforming():
    """单单元细化 → 与邻居的共享面"一侧细化、一侧没细化" → 精确前置判据必须拒绝。

    （几何事后检测在角落单元上会被真实边界边掩盖 —— 所以入环用的是
    `refine_conformity` 这个**精确**前置判据。）"""
    V, C = cube_tet_mesh(2)
    mask = np.zeros(len(C), bool)
    mask[0] = True
    pre = refine_conformity(mask, C)
    assert pre["conforming"] is False and pre["n_mixed_faces"] > 0
    ref = refine_tets(V, C, mask=mask)
    assert len(ref["cells"]) == (len(C) - 1) + 8      # 仅 1 个单元被 1:8 细化
    full = refine_conformity(np.ones(len(C), bool), C)
    assert full["conforming"] is True and full["n_mixed_faces"] == 0


# ---------------- 边闭包协调细化 ----------------

def test_conforming_closure_refines_neighbours_and_stays_conforming():
    V, C = cube_tet_mesh(3)
    n0, vol0 = len(C), _vols(V, C).sum()
    mask = np.zeros(n0, bool)
    mask[0] = True
    ref = refine_tets_conforming(V, C, mask=mask, max_rounds=8)
    assert ref["ok"] and ref["conforming"] is True, ref.get("conformity")
    assert ref["conformity"]["n_mixed_faces"] == 0      # 精确前置判据
    assert ref["mesh_conformity"]["conforming"] is True  # 结构诊断
    assert ref["n_refined"] > 1                 # 闭包把共享边的邻居一并细化
    assert ref["n_closed"] >= 1 and ref["rounds"] >= 1
    assert ref["n_after"] == 8 * ref["n_refined"] + (n0 - ref["n_refined"])
    p = ref["parent"]
    assert p.size == ref["n_after"] and int(p.min()) >= 0 and int(p.max()) < n0
    counts = np.bincount(p, minlength=n0)
    assert set(np.unique(counts)).issubset({1, 8})   # 未细化→自身；细化→8 子同父
    assert _vols(ref["vertices"], ref["cells"]).sum() == pytest.approx(vol0, rel=1e-12)
    ref_all = refine_tets_conforming(V, C, mask=np.ones(n0, bool))
    assert ref_all["conforming"] is True and ref_all["n_after"] == 8 * n0


def test_transfer_fields_exact_and_volume_weighted_mean_preserved():
    V, C = cube_tet_mesh(3)
    rng = np.random.default_rng(7)
    f = rng.normal(size=(len(C), 3))
    ref = refine_tets_conforming(V, C, mask=amr_marks(V, C, threshold=0.99)["mask"])
    new = transfer_fields(ref["parent"], f)
    assert new.shape == (ref["n_after"], 3)
    const = np.full((len(C), 3), 2.5)
    assert np.allclose(transfer_fields(ref["parent"], const), 2.5)
    w0, w1 = _vols(V, C), _vols(ref["vertices"], ref["cells"])
    m0 = (f * w0[:, None]).sum(axis=0) / w0.sum()
    m1 = (new * w1[:, None]).sum(axis=0) / w1.sum()
    assert np.allclose(m0, m1, rtol=1e-12, atol=1e-14)


# ---------------- 入环：场不被清零 ----------------

def _pressure_backend(nx=2):
    V, C = cube_tet_mesh(nx)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                       alpha_momentum=0.7, alpha_pressure=0.3)
    return SolverBackend(s), s


def test_backend_amr_transfers_solution_exactly():
    """AMR 后解必须是传递后的场（旧实现会跳回均匀初值 → std≈0）。

    注意 `run_loop` 每次调用都会重置初值（solver_run.py:808），所以必须在
    **单次** run_loop 内完成：interval == max_iter 让 AMR 恰好落在最后一步、
    且是该步的最后动作 → 出循环时场就是传递结果（可逐位比对）。
    """
    V0, C0 = cube_tet_mesh(2)
    s = PressureSolver(V0, C0, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                       alpha_momentum=0.7, alpha_pressure=0.3)
    be = SolverBackend(s)
    be.initialize()
    caps = []
    be.set_amr(interval=6, threshold=0.99)
    be.run_loop(max_iter=6,
                iter_callback=lambda it, pl: caps.append(s.velocity().copy()))
    assert len(caps) == 6
    assert float(np.std(caps[-1][:, 0])) > 0.0     # 解已发展成非均匀场
    marks = amr_marks(V0, C0, kind="tet", threshold=0.99)
    assert marks["n_marks"] > 0
    ref = refine_tets_conforming(V0, C0, mask=marks["mask"])
    assert ref["conforming"] is True
    amr = be.metrics()["amr"]
    assert amr["times"] == 1 and amr["transferred"] is True
    assert amr["n_before"] == len(C0)
    assert amr["n_after"] == ref["n_after"]
    u_after = s.velocity()
    assert u_after.shape[0] == ref["n_after"]
    expected = transfer_fields(ref["parent"], caps[-1])
    assert np.allclose(u_after, expected, rtol=0.0, atol=1e-12)  # 原样传递
    assert float(np.std(u_after[:, 0])) > 0.0                    # 未被清零
    assert s.pressure().shape[0] == ref["n_after"]


def test_backend_amr_honest_skip_when_unable_to_transfer():
    """节点场求解器 + 非协调 5-tet 网格 → 跳过并如实记录原因（不塞给求解器）；
    协调网格上的节点场求解器 → 细化发生但传递如实标注"非单元量"。"""
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    be.set_amr(interval=5, threshold=0.99)
    be.run_loop(max_iter=10)
    amr = be.metrics()["amr"]
    assert amr is not None and amr["times"] == 0
    assert "未协调" in (amr["skipped"] or "")
    assert len(be.solver.cells) == len(C)            # 网格未被替换
    V2, C2 = cube_tet_mesh(2)
    be2 = SolverBackend(DemoDiffusionSolver(V2, C2))
    be2.initialize()
    be2.set_amr(interval=5, threshold=0.99)
    be2.run_loop(max_iter=10)
    amr2 = be2.metrics()["amr"]
    assert amr2["times"] >= 1 and amr2["n_after"] > amr2["n_before"]
    assert amr2["transferred"] is False and "非单元量" in (amr2["note"] or "")
