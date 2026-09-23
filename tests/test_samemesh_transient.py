# -*- coding: utf-8 -*-
"""S6/S2：同网格**瞬态**官方对照的宏与驱动（API 链护栏 + 参数契约）。

背景：S2 长期悬置的问题是「阻断在网格还是求解器」。做法是让官方求解器在我方
同一张 CGNS 网格上跑瞬态（四个外壁用 SymmetryBoundary = 滑移，对齐我方
wall_slip_axes=(1,2)），再离线用本仓库同一份 force_coefficients 算 CL(t)。

本文件只做**离线**护栏（不跑官方）：宏模板必须渲染出经 Javadoc 核对过的调用链 ——
这条链踩过三个真坑，每个都留断言防回归：
  ① 时间步要经 **Solver**（`sim.getSolverManager().getSolver(Class)`）拿
     `SpecifiedTimestepUnsteadySolver`，不能走 `ModelManager.getModel`
     （后者泛型上界是 Model，编译期报错）；
  ② 推进用 `it.step(n)`（`run(n)` 粒度依赖求解器状态、`run()` 会被自带停止准则截断）；
  ③ 外壁必须是 `SymmetryBoundary`（滑移）才能与我方准二维设置可比。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import star_bridge as sb  # noqa: E402


def _render(**over):
    kw = dict(cgns="C:/tmp/a.cgns", out="C:/tmp/o", dt=0.01, total_time=12.0,
              sample_dt=0.5, rho=1.0, mu=1e-5, u_in=0.05, poll=50)
    kw.update(over)
    return sb.MESH_CASE_TRANSIENT_MACRO % kw


def test_transient_macro_renders_with_verified_api_chain():
    src = _render()
    # 非稳态模型 + 时间步（Solver 路径，不是 ModelManager）
    assert "pc.enable(ImplicitUnsteadyModel.class)" in src
    assert "sim.getSolverManager()" in src
    assert "sm.getSolver(SpecifiedTimestepUnsteadySolver.class)" in src
    assert "setTimeStep(dt)" in src
    assert "getModelManager().getModel(SpecifiedTimestepUnsteadySolver.class)" not in src
    # 推进用 step(int)；不得回退到 run(int)/时间轮询
    assert "it.step(chunk)" in src
    assert "it.run()" not in src
    assert "isIterating()" not in src
    # 四个外壁 → 对称（滑移），圆柱 → 壁面
    assert "SymmetryBoundary.class" in src and "WallBoundary.class" in src
    assert 'n.equals("Cylinder")' in src
    # 分段存场 + 标记
    assert "saveState(outp)" in src and "SAMEMESH save|" in src
    assert "SAMEMESH_DONE " in src


def test_transient_macro_parameters_are_injected():
    src = _render(total_time=20.0, sample_dt=0.25, u_in=0.07, mu=2e-5)
    assert "double totalTime = 20.0" in src
    assert "double sampleDt = 0.25" in src
    assert "double uIn = 0.07" in src
    assert "double mu = 2e-05" in src or "double mu = 2e-5" in src


def test_transient_driver_signature_and_honesty():
    import inspect
    sig = inspect.signature(sb.official_mesh_case_transient)
    for p in ("dt", "total_time", "sample_dt", "rho", "mu", "u_in", "timeout"):
        assert p in sig.parameters, p
    # 缺 CGNS 时必须诚实失败（不编造）
    res = sb.official_mesh_case_transient("x.sim", "no_such.cgns", "out",
                                          total_time=1.0)
    assert res["ok"] is False and "CGNS" in res["reason"]


def test_samemesh_transient_cli_stages_exist():
    src = open(os.path.join(ROOT, "s6_samemesh.py"), encoding="utf-8").read()
    assert "def do_official_transient" in src
    assert "def do_transient_series" in src
    assert "official_cl_series.json" in src
    # 命令行两段可独立重跑
    assert 'cmd == "transient"' in src and 'cmd == "series"' in src