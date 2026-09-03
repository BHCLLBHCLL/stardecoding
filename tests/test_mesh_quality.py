# -*- coding: utf-8 -*-
"""N 波网格内核数值验收（parity_100pct_plan.md 第 6 节验收条目）。

覆盖：
  N1  mesh_pipeline 流水线执行引擎（空流水线/进度/取消/四段真实流水线）
  N2  refine_surface 表面细分（面数 4^n、水密保持）
  N3  体网格 tet_mesh（scipy Delaunay 路由；gmsh 路由按环境门控）
  质量诊断 tet_quality（正体积/无负/边长纵横比）
  几何判定 point_in_mesh（espresso even-odd 内部/外部/表面）

唯一数值验收门：体网格全部单元正体积、水密输入、紧致重编号。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_io import cube_mesh
from mesh_pipeline import (MeshPipeline, PipelineCanceled, refine_surface,
                           default_volume_mesh_pipeline)
from mesh_tet import tet_mesh, tet_quality, point_in_mesh


def _watertight(v, f):
    from occ_repair import boundary_edges
    b, _ = boundary_edges(v, f)
    return len(b) == 0


def _has(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- N1 引擎
def test_pipeline_empty_runs_ok():
    pipe = MeshPipeline("空")
    out = pipe.run()
    assert out["ok"]
    assert not out["cancelled"]
    assert out["stages_ran"] == 0
    assert out["results"] == []
    assert out["previews"] == {}


def test_pipeline_progress_order_and_cancel():
    pipe = MeshPipeline()
    call = []
    ctx = {"x": 1}

    def make(v):
        def run(c):
            c["y"] = (c["x"] if "x" in c else 0) + v
            return v
        return run

    pipe.add("a", make(1))
    pipe.add("b", make(2))
    out = pipe.run(ctx=ctx, progress=lambda i, s, t: call.append((i, s)))
    assert out["ok"]
    assert [c[0] for c in call] == [1, 2]
    assert [c[1] for c in call] == ["a", "b"]
    assert out["ctx"]["y"] == 3

    # 运行中途取消
    state = {"n": 0}
    pipe2 = MeshPipeline()
    pipe2.add("s1", lambda c: None)

    def canceled():
        return True
    out2 = pipe2.run(canceled=canceled)
    assert not out2["ok"] and out2["cancelled"]


def test_pipeline_preview_and_weight():
    pipe = MeshPipeline()
    pipe.add("m", lambda c: c.__setitem__("mesh", ([1], [2])),
             preview=lambda c: c.get("mesh"))
    out = pipe.run()
    assert out["previews"]["m"] == ([1], [2])


# ---------------------------------------------------------------- N2 表面细分
def test_refine_surface_counts_and_watertight():
    V, F = cube_mesh(1.0)
    V, F = refine_surface(V, F, levels=1)
    assert len(F) == 12 * 4        # 每三角 → 4
    assert len(V) == 8 + 18        # 8 顶点 + 18 条共享边中点
    assert _watertight(V, F)


def test_refine_surface_multilevel_watertight():
    V, F = cube_mesh(1.0)
    V, F = refine_surface(V, F, levels=3)
    assert len(F) == 12 * (4 ** 3)
    assert _watertight(V, F)


# ---------------------------------------------------------------- 点判
def test_point_in_mesh_inside_outside():
    V, F = cube_mesh(1.0)
    assert point_in_mesh([0.5, 0.5, 0.5], V, F)          # 内部
    assert not point_in_mesh([2.0, 2.0, 2.0], V, F)      # 外部
    assert point_in_mesh([0.5, 0.5, 1.0], V, F)          # 顶面上


# ---------------------------------------------------------------- N3 体网格
@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_tet_cube_scipy_positive_volume():
    V, F = cube_mesh(1.0)
    out = tet_mesh(V, F, spacing=0.25, method="scipy")
    assert out["ok"]
    assert out["method"].startswith("scipy")
    assert out["n_cells"] > 0
    q = out["quality"]
    assert q["n_negative"] == 0
    assert q["volume"]["min"] > 0
    assert not np.isclose(q["edge_aspect"]["max"], float("inf"))


def test_tet_quality_metrics():
    V, F = cube_mesh(1.0)
    out = tet_mesh(V, F, spacing=0.4, method=("scipy" if _has("scipy") else None))
    if not out.get("ok"):
        pytest.skip("无 scipy 也无法生成")
    q = tet_quality(out["vertices"], out["cells"])
    assert q["n_cells"] == out["n_cells"]
    assert q["n_negative"] == 0
    assert 0 < q["volume"]["mean"] <= 1.0
    assert q["edge_aspect"]["mean"] >= 1.0


def test_tet_reject_open_surface():
    V, F = cube_mesh(1.0)
    F_open = F[:-1]   # 移除一面对开放
    with pytest.raises(ValueError):
        tet_mesh(V, F_open, method=("scipy" if _has("scipy") else None))


# ---------------------------------------------------------------- 默认流水线
def test_default_volume_mesh_pipeline_full():
    V, F = cube_mesh(1.0)
    pipe = default_volume_mesh_pipeline(V, F, cell_size=0.5, refine_levels=1)
    ran = ["生成表面网格", "生成体网格", "质量诊断", "单元重编号"]
    seen = []
    out = pipe.run(progress=lambda i, s, t: seen.append(s))
    assert out["ok"]
    assert seen == ran
    ctx = out["ctx"]
    assert ctx["volume"] is not None
    Vn, Cn = ctx["volume"]
    assert Vn.shape[1] == 3
    assert Cn.shape[0] > 0
    # 重编号紧致：使用节点恰为 0..max 连续区间
    used = set(int(x) for x in Cn.ravel())
    assert used == set(range(len(used)))
    assert int(Cn.min()) == 0
    assert ctx["quality"]["volume"]["n_negative"] == 0


@pytest.mark.skipif(not _has("gmsh"), reason="无 gmsh")
def test_tet_cube_gmsh_route():
    V, F = cube_mesh(1.0)
    out = tet_mesh(V, F, spacing=0.4, method="gmsh")
    assert out["ok"]
    assert out["method"] == "gmsh"
    assert out["n_cells"] > 0
    assert out["quality"]["n_negative"] == 0