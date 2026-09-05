# -*- coding: utf-8 -*-
"""P 波 P10：求解运行闭环（监视器 / 报告 / 停止准则 / Update Events + 残差实时曲线）。

覆盖：
  DemoDiffusionSolver：纯 numpy 图拉普拉斯显式扩散（网格/步进/残差/监视器载荷）
  Monitor / MonitorManager：采样/聚合/曲线元组/文本表/降采样
  StopCriterion：步数 / 残差 / 算时判停
  UpdateEvents：注册 / 触发 / 注销 / 单回调异常隔离
  SolverBackend 状态机 + run_loop：初始化 / 运行 / 暂停 / 单步 / 恢复 / 停止
  AMR 运行时接入：set_amr → run_amr 细化 → set_mesh 重建
  线程化 Run / Pause / Step / Stop 闭环（P10 验收核心）

验收核心（P10 行）：Run/暂停/步进/停止全按钮生效 —— 后端状态机 + run_loop 感知
GUI 线程写入的控制标志（暂停/单步/停止即时响应）。
"""
import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_amr import refine_tets, register_amr_hook, unregister_amr_hook
from solver_run import (DemoDiffusionSolver, Monitor, MonitorManager,
                        Report, SolverBackend, SolverState, StopCriterion,
                        UpdateEvents, demo_mesh)


def _wait_until(fn, timeout=1.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.01)
    return False


def _tet_volumes(V, C):
    a = V[C[:, 0]] - V[C[:, 1]]
    b = V[C[:, 0]] - V[C[:, 2]]
    c = V[C[:, 0]] - V[C[:, 3]]
    return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


# ---------------------------------------------------------------- demo 网格 / 求解器
def test_demo_mesh_shape():
    V, C = demo_mesh(nx=3)
    assert V.ndim == 2 and V.shape[1] == 3
    assert C.ndim == 2 and C.shape[1] == 4
    assert len(V) == (3 + 1) ** 3
    assert len(C) == 3 * 3 * 3 * 5


def test_demo_mesh_tet_positive_volume():
    V, C = demo_mesh(nx=2)
    assert (_tet_volumes(V, C) > 0).all()


def test_diffusion_solver_initialization():
    V, C = demo_mesh(nx=3)
    s = DemoDiffusionSolver(V, C)
    assert s.field().shape == (len(V),)
    assert s.source_nodes().size > 0
    assert np.isnan(s.residual())


def test_diffusion_solver_step_monotone_residual():
    V, C = demo_mesh(nx=3)
    s = DemoDiffusionSolver(V, C)
    prev = None
    for _ in range(10):
        p = s.step()
        assert set(p) == {"residual", "u_min", "u_max", "u_mean"}
        if prev is not None:
            assert p["residual"] <= prev * 1.05     # 单调不减（容忍数值小波动）
        prev = p["residual"]
    assert s.iteration == 10
    assert s.monitor_payload()["residual"] == pytest.approx(prev)


def test_diffusion_solver_set_mesh_rebuilds():
    V, C = demo_mesh(nx=3)
    s = DemoDiffusionSolver(V, C)
    n0 = len(s.vertices)
    s.set_mesh(*demo_mesh(nx=4))
    assert len(s.vertices) != n0                     # 24→64 节点（nx+1 立方）
    assert s.field().shape == (len(s.vertices),)


# ---------------------------------------------------------------- Monitor / MonitorManager
def test_monitor_sample_aggregates():
    m = Monitor("residual", kind="residual")
    for i in range(5):
        m.sample(i, float(i) / 10.0, x=i)
    assert m.n == 5
    assert np.allclose(m.values(), [0.0, 0.1, 0.2, 0.3, 0.4])
    assert np.allclose(m.xs(), [0.0, 1.0, 2.0, 3.0, 4.0])
    assert m.last == pytest.approx(0.4)
    assert m.y_min == pytest.approx(0.0)
    assert m.y_max == pytest.approx(0.4)
    assert m.mean == pytest.approx(0.2)
    assert "residual" in m.line()


def test_monitor_manager_sample_and_table():
    mgr = MonitorManager()
    mgr.add("residual", Monitor("residual", kind="residual"))
    mgr.add("u_mean", Monitor("u_mean", kind="field"))
    mgr.sample(1, {"residual": 0.1, "u_mean": 0.5, "u_extra": 0.3})
    mgr.sample(2, {"residual": 0.05, "u_mean": 0.6, "u_extra": 0.4})
    assert "residual" in mgr
    assert mgr.names() == ["residual", "u_mean", "u_extra"]
    assert mgr.current("residual") == pytest.approx(0.05)
    tbl = mgr.table()
    assert len(tbl) == 3
    row = next(r for r in tbl if r["name"] == "residual")
    assert row["n"] == 2 and row["last"] == pytest.approx(0.05)


def test_monitor_manager_items_frame():
    mgr = MonitorManager()
    mgr.add("residual", Monitor("residual", kind="residual"))
    for i in range(10):
        mgr.sample(i, {"residual": float(i)}, x=i)
    items = mgr.items(max_pts=512)
    assert len(items) == 1
    name, tag, ys, mid, xs = items[0]
    assert name == "residual" and tag == "P10" and mid == 0
    assert len(ys) >= 2 and xs is not None and len(xs) == len(ys)


def test_monitor_items_downsampled():
    mgr = MonitorManager()
    mgr.add("residual", Monitor("residual", kind="residual"))
    for i in range(1000):
        mgr.sample(i, {"residual": float(i)}, x=i)
    items = mgr.items(max_pts=100)
    ys = items[0][2]
    assert 2 <= len(ys) <= 101
    assert ys[0] == pytest.approx(0.0)
    assert ys[-1] == pytest.approx(999.0)


# ---------------------------------------------------------------- StopCriterion
def test_stop_max_iterations():
    c = StopCriterion(max_iterations=10)
    assert c.evaluate(10, 0.5) == (True, "已到最大迭代 10")
    assert c.evaluate(9, 0.5) == (False, "")


def test_stop_residual_tol():
    c = StopCriterion(residual_tol=0.05)
    stop, reason = c.evaluate(1, 0.03)
    assert stop and "残差" in reason and "0.03" in reason
    assert c.evaluate(1, 0.09) == (False, "")
    assert c.evaluate(1, None) == (False, "")       # residual None 时该准则跳过


def test_stop_wall_time():
    c = StopCriterion(wall_time=0.05)
    c.start_timer()
    time.sleep(0.06)
    stop, reason = c.evaluate(1, 0.5)
    assert stop and "算时" in reason


# ---------------------------------------------------------------- UpdateEvents
def test_update_events_fire_and_unregister():
    ev = UpdateEvents()
    called = []

    def on_iterate(**kw):
        called.append(kw)
        return "ok"

    ev.register("iterate", on_iterate)
    assert ev.count("iterate") == 1
    assert ev.fire("iterate", iteration=3, payload={"residual": 0.1}) == ["ok"]
    assert called[0]["iteration"] == 3
    ev.unregister("iterate", on_iterate)
    assert ev.count("iterate") == 0
    assert ev.fire("iterate", iteration=4) == []


def test_update_events_registry_covers_all_names():
    ev = UpdateEvents()
    assert set(ev.EVENT_NAMES) >= {"start", "iterate", "pause", "resume",
                                   "step", "stop", "finish", "error", "amr"}


def test_update_events_bad_name_raises():
    ev = UpdateEvents()
    with pytest.raises(KeyError):
        ev.register("nope", lambda **kw: True)


def test_update_events_single_callback_error_isolated():
    ev = UpdateEvents()

    def bad(**kw):
        raise ValueError("boom")

    ev.register("iterate", bad)
    ev.register("iterate", lambda **kw: "fine")
    res = ev.fire("iterate", iteration=1)
    assert any(isinstance(r, dict) and r.get("error") == "boom" for r in res)
    assert "fine" in res


# ---------------------------------------------------------------- SolverBackend 状态机
def test_backend_initialize_state():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    assert be.state() == SolverState.IDLE
    assert be.initialize()
    assert be.state() == SolverState.INITIALIZED


def test_backend_pause_requires_running():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    assert not be.pause()                    # 未运行不可暂停
    assert be.run()
    assert be.state() == SolverState.RUNNING
    assert be.pause()
    assert be.state() == SolverState.PAUSED
    assert not be.run()                      # PAUSED 不可直接 run（用 resume）


def test_backend_resume_only_from_paused():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    assert not be.resume()                   # INITIALIZED 非 PAUSED
    assert be.run()
    assert not be.resume()                   # RUNNING 非 PAUSED
    be.pause()
    assert be.resume()
    assert be.state() == SolverState.RUNNING


def test_backend_step_requires_paused_or_idle():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    assert be.step()                         # IDLE → PAUSED 单步请求
    assert be.state() == SolverState.PAUSED
    assert be.step()                         # PAUSED 下可再请求（连续单步）
    be2 = SolverBackend(DemoDiffusionSolver(V, C))
    be2.initialize()
    be2.run()
    assert not be2.step()                    # RUNNING 下忽略单步


def test_backend_stop_terminal():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    assert be.stop()
    assert be.state() == SolverState.STOPPED
    assert not be.stop()                     # 终态不再停


# ---------------------------------------------------------------- run_loop
def test_run_loop_completes_by_max_iter():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    r = be.run_loop(max_iter=10)
    assert r["state"] == SolverState.COMPLETED
    assert r["iteration"] == 10
    assert r["residual"] is not None
    assert be.metrics()["state"] == SolverState.COMPLETED
    assert be.metrics()["iteration"] == 10
    assert len(be.curve_items()) == 4


def test_run_loop_stops_by_residual_tol():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C),
                       stop_criteria=[StopCriterion(residual_tol=0.05)])
    be.initialize()
    r = be.run_loop(max_iter=500)
    assert r["state"] == SolverState.COMPLETED
    assert "残差" in r["reason"]
    assert r["iteration"] < 500


def test_run_loop_report_and_events():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    be.run_loop(max_iter=5)
    lines = be.report_lines()
    assert any("监视器" in ln for ln in lines)
    assert any("residual" in ln for ln in lines)
    assert any("运行时事件" in ln for ln in lines)


def test_run_loop_amr_refines_mesh():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    register_amr_hook(refine_tets)
    try:
        be.set_amr(interval=5, threshold=0.99)
        be.run_loop(max_iter=20)
        amr = be.metrics()["amr"]
        assert amr is not None
        assert amr["times"] >= 1
        assert amr["n_after"] > amr["n_before"]
    finally:
        unregister_amr_hook(refine_tets)


# ---------------------------------------------------------------- 线程化 Run/Pause/Step/Stop（P10 验收核心）
def test_run_loop_threaded_pause_step_resume_stop():
    V, C = demo_mesh(nx=3)
    be = SolverBackend(DemoDiffusionSolver(V, C))
    be.initialize()
    holder = {}

    def worker():
        holder["r"] = be.run_loop(max_iter=100000)

    t = threading.Thread(target=worker)
    t.start()
    assert _wait_until(lambda: be.state() in (SolverState.RUNNING,
                                              SolverState.PAUSED))
    assert be.pause()
    assert _wait_until(lambda: be.state() == SolverState.PAUSED)
    it0 = be.metrics()["iteration"]
    assert be.step()
    assert _wait_until(lambda: be.metrics()["iteration"] > it0)
    assert be.state() == SolverState.PAUSED       # 单步后仍停留 PAUSED
    assert be.resume()
    assert _wait_until(lambda: be.state() in (SolverState.RUNNING,
                                              SolverState.PAUSED))
    assert be.stop()
    t.join(timeout=3)
    assert not t.is_alive()
    assert holder["r"]["state"] in (SolverState.STOPPED, SolverState.COMPLETED)
