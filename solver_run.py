# -*- coding: utf-8 -*-
"""P 波 P10：求解运行闭环（监视器 / 报告 / 停止准则 / Update Events + 残差实时曲线）。

不依赖 CCM 许可：A 路线提供 `DemoDiffusionSolver`（图拉普拉斯显式扩散，纯 numpy）
作为残差/监视器的**真实运行时**来源，配合 `SolverBackend` 状态机驱动
Run / Pause / Step / Stop 闭环；`MonitorManager` 采样残差与场统计，
`Report` 生成文本表，`StopCriterion` 判定收敛/步数/算时，`UpdateEvents`
做运行时回调（start/iterate/pause/resume/step/stop/finish/error/amr）。

运行时闭环线程模型：`SolverBackend.run_loop()` 在**工作线程**阻塞迭代，每步
检查状态标志（RUNNING/PAUSED/STOPPED）以对 GUI 命令即时响应；GUI 从另一线程
调 `pause()/step()/stop()` 写入标志，`run_loop()` 感知后即暂停/单步/退出。
`metrics()` 提供线程安全快照（状态 + 迭代 + 残差 + 监视器曲线元组），
`curve_items()` 产出兼容 star_gui_plots.SeriesCanvas 的 (name, tag, ys, id, xs)。

用法（无 GUI）：
    from solver_run import SolverBackend, DemoDiffusionSolver, demo_mesh
    be = SolverBackend(DemoDiffusionSolver(*demo_mesh(nx=3)))
    be.initialize()
    be.run_loop(max_iter=40)          # 阻塞至 stop/步数到；GUI 线程可 pause/stop
    print(be.metrics()); print("\\n".join(be.report_lines()))
"""
import threading
import time

import numpy as np


def _safe_norm(a, axis=None):
    """np.linalg.norm 的替代实现。

    occ 环境中 numpy/BLAS 对部分数组长度调用 np.linalg.norm / np.dot 会触发
    Windows fatal exception (0xc06d007f)，因此统一改用 sqrt(sum(x*x))。
    axis=None 返回浮点标量（用于 1-D 向量）；axis=1 返回逐行范数（2-D (n,3)）。
    """
    a = np.asarray(a)
    sq = a * a
    if axis is None:
        return float(np.sqrt(np.sum(sq)))
    return np.sqrt(np.sum(sq, axis=axis))


# ---------------------------------------------------------------------------
# 运行状态
# ---------------------------------------------------------------------------
class SolverState:
    """P10 运行状态（字符串常量，便于 GUI/CLI/log 直接显示）。"""
    IDLE = "IDLE"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

    # 终态（run_loop 退出后不再自走）
    TERMINAL = frozenset({STOPPED, COMPLETED, ERROR})


# ---------------------------------------------------------------------------
# 停止准则
# ---------------------------------------------------------------------------
class StopCriterion:
    """停止准则：步数 / 残差 / 算时 任一达到即判停。

    同一准则可组合多个条件（全部为 None 则永不触发）。evaluate 由 run_loop
    每步调用，返回 (stop, reason)。
    """

    def __init__(self, max_iterations=None, residual_tol=None, wall_time=None,
                 name="criterion"):
        self.max_iterations = max_iterations
        self.residual_tol = residual_tol
        self.wall_time = wall_time
        self.name = name
        self._t0 = None

    def start_timer(self):
        self._t0 = time.monotonic()

    def _elapsed(self):
        if self._t0 is None:
            return 0.0
        return time.monotonic() - self._t0

    def evaluate(self, iteration, residual=None):
        """返回 (stop, reason)。residual 可 None（残差准则跳过）。"""
        if self.max_iterations is not None and iteration >= int(self.max_iterations):
            return True, "已到最大迭代 %d" % self.max_iterations
        if (self.residual_tol is not None and residual is not None
                and float(residual) <= float(self.residual_tol)):
            return True, "残差 %.3g 低于阈值 %.3g" % (float(residual),
                                                    float(self.residual_tol))
        if self.wall_time is not None and self._elapsed() >= float(self.wall_time):
            return True, "算时 %.2fs 达到上限" % self._elapsed()
        return False, ""


# ---------------------------------------------------------------------------
# 监视器（单指标时间序列）
# ---------------------------------------------------------------------------
class Monitor:
    """命名标量时间序列：sample(iteration, value, x=None)。

    - `history[(iter, x, value)]` 逐点保留；`values()` 返回 y 向量（np）。
    - `last / n / y_min / y_max / mean` 提供报告与曲线用聚合。
    - `xs()` 返回 x 向量（迭代号或物理时间）；None 时曲线按索引定位。
    """

    def __init__(self, name, kind="monitor", x_label="Iteration",
                 y_label=None, _id=0):
        self.name = name
        self.kind = kind
        self.x_label = x_label
        self.y_label = y_label or name
        self.id = _id
        self.history = []      # [(iter, x, value)]

    def sample(self, iteration, value, x=None):
        self.history.append((int(iteration), None if x is None else float(x),
                             float(value)))

    def clear(self):
        self.history.clear()

    @property
    def n(self):
        return len(self.history)

    def values(self):
        if not self.history:
            return np.array([], float)
        return np.array([h[2] for h in self.history], float)

    def xs(self):
        if not self.history:
            return None
        if all(h[1] is not None for h in self.history):
            return np.array([h[1] for h in self.history], float)
        return None

    @property
    def last(self):
        return float(self.history[-1][2]) if self.history else float("nan")

    @property
    def y_min(self):
        v = self.values()
        return float(v.min()) if v.size else float("nan")

    @property
    def y_max(self):
        v = self.values()
        return float(v.max()) if v.size else float("nan")

    @property
    def mean(self):
        v = self.values()
        return float(v.mean()) if v.size else float("nan")

    def line(self):
        return ("%-32s %-20s n=%-7d y[%+.4g .. %+.4g] last=%+.4g" % (
            self.name, self.kind.split(".")[-1], self.n,
            self.y_min if self.n else 0.0, self.y_max if self.n else 0.0,
            self.last if self.n else 0.0))


# ---------------------------------------------------------------------------
# 监视器集合（多指标 + 曲线元组）
# ---------------------------------------------------------------------------
class MonitorManager:
    """持有多个 Monitor，按名字访问；sample() 批量喂入一次解的指标字典。

    产出供 SeriesCanvas / 报告 / CLI 三种视角：
      - `items(max_pts)`：(name, "P10", ys, id, xs) 降采样元组（曲线）；
      - `lines()`：每监视器一行的文本表（报告）；
      - `table()`：[{name, kind, n, last, y_min, y_max}]（dict 视图）。
    """

    def __init__(self):
        self._monitors = {}
        self._order = []

    def add(self, name, monitor=None, **kw):
        if name in self._monitors:
            return self._monitors[name]
        mon = monitor or Monitor(name, _id=len(self._order))
        if mon.name not in self._monitors:
            self._monitors[name] = mon
            self._order.append(name)
        return self._monitors[name]

    def get(self, name):
        return self._monitors.get(name)

    def __contains__(self, name):
        return name in self._monitors

    def names(self):
        return list(self._order)

    def clear(self):
        for m in self._monitors.values():
            m.clear()

    def sample(self, iteration, payload, x=None):
        """payload: {monitor_name: value}。x 为公共 X（如迭代号）。"""
        for name, val in payload.items():
            if name in self._monitors:
                self._monitors[name].sample(iteration, val, x)
            else:
                self.add(name).sample(iteration, val, x)

    def current(self, name):
        m = self._monitors.get(name)
        return m.last if m else None

    def items(self, max_pts=512):
        out = []
        for name in self._order:
            mon = self._monitors[name]
            y = mon.values()
            if y.size < 2:
                continue
            xs = mon.xs()
            step = max(1, y.size // max_pts)
            idxs = list(range(0, y.size, step))
            if idxs[-1] != y.size - 1:
                idxs.append(y.size - 1)
            ys = [float(y[i]) for i in idxs]
            xs_l = ([float(xs[i]) for i in idxs]
                    if xs is not None and xs.size == y.size else None)
            out.append((mon.name, "P10", ys, mon.id, xs_l))
        return out

    def lines(self):
        return [m.line() for m in self._monitors.values() if m.n]

    def table(self):
        return [{"name": m.name, "kind": m.kind, "n": m.n, "last": m.last,
                 "y_min": m.y_min, "y_max": m.y_max}
                for m in self._monitors.values() if m.n]


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
class Report:
    """把监视器 + 运行元信息格式化为文本报告（输出窗 / log / CLI 共用）。"""

    def __init__(self, title="Solution Report"):
        self.title = title

    def lines(self, monitors, meta=None):
        meta = meta or {}
        out = [self.title, "-" * 72]
        out.append("状态: %-10s  迭代: %-6d  残差: %s" % (
            meta.get("state", "-"), meta.get("iteration", 0),
            _fmt(last(meta.get("residual"))) if meta.get("residual") is not None
            else "-"))
        reason = meta.get("reason")
        if reason:
            out.append("停止: %s" % reason)
        out.append("")
        out.append("监视器:")
        rows = monitors.lines()
        if rows:
            out.extend(rows)
        else:
            out.append("（尚未采样）")
        out.append("")
        out.append("运行时事件: %d" % meta.get("events", 0))
        if meta.get("amr"):
            a = meta["amr"]
            out.append("AMR: %d 次细化（%d→%d 单元）" % (
                a.get("times", 0), a.get("n_before", 0), a.get("n_after", 0)))
        return out


def last(value):
    if isinstance(value, (list, tuple)) and value:
        return value[-1]
    return value


def _fmt(v):
    try:
        return "%+.4g" % float(v)
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Update Events 运行时回调
# ---------------------------------------------------------------------------
class UpdateEvents:
    """运行时回调注册表：start / iterate / pause / resume / step / stop /
    finish / error / amr。fire 返回各回调结果；单回调异常不中断其余。

    GUI 把每类回调接到 Qt 信号（跨线程自动排队），或接绘图/日志函数。
    """

    EVENT_NAMES = ("start", "iterate", "pause", "resume", "step",
                   "stop", "finish", "error", "amr")

    def __init__(self):
        self._cb = {n: [] for n in self.EVENT_NAMES}

    def register(self, name, fn):
        if name not in self._cb:
            raise KeyError("未知事件: %s" % name)
        if callable(fn) and fn not in self._cb[name]:
            self._cb[name].append(fn)
        return fn

    def unregister(self, name, fn):
        if fn in self._cb.get(name, []):
            self._cb[name].remove(fn)

    def fire(self, name, **kwargs):
        results = []
        for fn in list(self._cb.get(name, [])):
            try:
                results.append(fn(**kwargs))
            except Exception as exc:   # noqa: BLE001
                results.append({"event": name, "error": str(exc)})
        return results

    def count(self, name=None):
        if name is not None:
            return len(self._cb.get(name, []))
        return sum(len(v) for v in self._cb.values())


# ---------------------------------------------------------------------------
# Demo 扩散求解器（纯 numpy 图拉普拉斯显式扩散）
# ---------------------------------------------------------------------------
class DemoDiffusionSolver:
    """最小可运行的 demo 求解器：四面体网格上的标量扩散（纯 numpy）。

    构造边混合拉普拉斯（conductivity / |edge| 权重）→ 每步显式前向欧拉
    扩散，端面（min-x）固定 Dirichlet 定值，残差 = ||Δu||/max(1,||u_old||)
    单调衰减；产出残差 + 场统计（U_min/max/mean）监视器采样。

    `set_mesh(V, C)`（AMR 运行时接入）重建拉普拉斯并重算 Dirichlet 源。
    """

    def __init__(self, vertices, cells, conductivity=1.0, source_value=1.0,
                 source_axis=0, source_side="min", edge_source_frac=0.05,
                 name="DemoDiffusion"):
        self.name = name
        self.conductivity = float(conductivity)
        self.source_value = float(source_value)
        self.source_axis = int(source_axis)
        self.source_side = source_side
        self.edge_source_frac = float(edge_source_frac)
        self.vertices = None
        self.cells = None
        self.iteration = 0
        self._field = None
        self._source_mask = None
        self._degree = None
        self._edges_src = None
        self._edges_dst = None
        self._weights = None
        self._dt = None
        self._last_residual = float("nan")
        self.set_mesh(vertices, cells)

    # -- 网格 / 拉普拉斯 ------------------------------------------------
    def set_mesh(self, vertices, cells):
        V = np.asarray(vertices, float)
        C = np.asarray(cells, np.int64)
        if C.ndim != 2 or C.shape[1] < 4:
            raise ValueError("DemoDiffusionSolver 需要四面体单元(≥4 顶点)")
        if len(V) < 8 or len(C) < 1:
            raise ValueError("网格过小，无法扩散")
        self.vertices = V
        self.cells = C
        self._rebuild_laplacian()
        self._initialize_field()

    def _rebuild_laplacian(self):
        V = self.vertices
        C = self.cells
        # 四面体 6 条边（去重：a<b 规范化后再合并到稀疏数组）
        e = np.concatenate([
            np.stack([C[:, 0], C[:, 1]], 1),
            np.stack([C[:, 0], C[:, 2]], 1),
            np.stack([C[:, 0], C[:, 3]], 1),
            np.stack([C[:, 1], C[:, 2]], 1),
            np.stack([C[:, 1], C[:, 3]], 1),
            np.stack([C[:, 2], C[:, 3]], 1),
        ], axis=0)
        lo = np.minimum(e[:, 0], e[:, 1])
        hi = np.maximum(e[:, 0], e[:, 1])
        n = len(V)
        key = lo.astype(np.int64) * np.int64(n) + hi
        _, first = np.unique(key, return_index=True)
        lo, hi, key = lo[first], hi[first], key[first]
        w = self.conductivity / np.maximum(
            _safe_norm(V[lo] - V[hi], axis=1), 1e-12)
        deg = np.zeros(n, float)
        np.add.at(deg, lo, w)
        np.add.at(deg, hi, w)
        self._edges_src = lo
        self._edges_dst = hi
        self._weights = w
        self._degree = deg
        dmax = float(deg.max()) if len(deg) else 1.0
        self._dt = 0.8 / max(dmax, 1e-12)

    def _source_nodes(self):
        V = self.vertices
        ax = V[:, self.source_axis]
        diag = _safe_norm(V.max(axis=0) - V.min(axis=0)) or 1.0
        tol = self.edge_source_frac * diag
        if self.source_side == "min":
            cut = float(ax.min())
            return np.where(ax <= cut + tol)[0]
        cut = float(ax.max())
        return np.where(ax >= cut - tol)[0]

    def _initialize_field(self):
        self._source_mask = self._source_nodes()
        self._field = np.zeros(len(self.vertices), float)
        self._field[self._source_mask] = self.source_value
        self.iteration = 0

    # -- 场 / 迭代 ------------------------------------------------------
    def field(self):
        return self._field

    def source_nodes(self):
        return self._source_mask

    def residual(self):
        return float("nan") if self._field is None else self._last_residual

    def _diffusion_step(self):
        """一步显式扩散：返回新场（未应用到实例，供残差计算）。"""
        f = self._field
        acc = np.zeros(len(f), float)
        w = self._weights
        np.add.at(acc, self._edges_src, w * f[self._edges_dst])
        np.add.at(acc, self._edges_dst, w * f[self._edges_src])
        lape = acc - self._degree * f
        new = f + self._dt * lape
        new[self._source_mask] = self.source_value
        return new

    def step(self):
        """执行一步扩散，更新时间/迭代/残差；返回本步指标字典。"""
        new = self._diffusion_step()
        old = self._field
        denom = max(_safe_norm(old), 1e-12)
        res = _safe_norm(new - old) / denom
        self._field = new
        self.iteration += 1
        self._last_residual = res
        return {"residual": res,
                "u_min": float(new.min()),
                "u_max": float(new.max()),
                "u_mean": float(new.mean())}

    def monitor_payload(self):
        """本步指标的监视器映射：残差 + 场统计（与 step() 返回键一致）。"""
        return {"residual": self._last_residual,
                "u_min": float(self._field.min()),
                "u_max": float(self._field.max()),
                "u_mean": float(self._field.mean())}


# ---------------------------------------------------------------------------
# 求解后端（状态机 + 闭环驱动器）
# ---------------------------------------------------------------------------
class SolverBackend:
    """运行闭环控制器：状态机 + DemoDiffusionSolver + 监视器 + 停止准则 + 事件。

    线程模型：
      - `run_loop()` 在工作线程阻塞迭代，逐点检查状态标志以响应 GUI 命令；
      - `pause()/resume()/step()/stop()/initialize()` 从任意线程写入控制标志；
      - `metrics()/curve_items()/report_lines()` 线程安全快照（GUI 轮询/绘制）。

    AMR（N6 联调）接入：`set_amr(interval, threshold)` 开启运行时细化，
    run_loop 每 interval 步回调 mesh_amr.run_amr；若返回新网格则 solver.set_mesh
    重建并 fire "amr" 事件。
    """

    def __init__(self, solver=None, stop_criteria=None, monitors=None,
                 events=None):
        self.solver = solver
        self.monitors = monitors or MonitorManager()
        self.events = events or UpdateEvents()
        self.stop_criteria = list(stop_criteria) if stop_criteria else []
        self._lock = threading.RLock()
        self._state = SolverState.IDLE
        self._iteration = 0
        self._stop_reason = ""
        self._result = None
        self._amr_interval = None
        self._amr_threshold = 0.30
        self._amr_times = 0
        self._amr_n_before = 0
        self._amr_n_after = 0
        self._event_count = 0
        self._pause_sleep = 0.02
        self._step_requested = False
        self._active_criteria = []
        # 默认监视器（初始化时注册）
        self._ensure_default_monitors()

    # -- 初始化 / 监视器 ------------------------------------------------
    def _ensure_default_monitors(self):
        for name, kind in (("residual", "residual"),
                           ("u_min", "field"), ("u_max", "field"),
                           ("u_mean", "field")):
            self.monitors.add(name, Monitor(name, kind=kind))

    def set_amr(self, interval, threshold=0.30):
        """开启 AMR 运行时细化（interval 步一次）；None 关闭。"""
        self._amr_interval = int(interval) if interval else None
        self._amr_threshold = float(threshold)

    # -- 命令（任意线程） ------------------------------------------------
    def initialize(self):
        if self.solver is None:
            return False
        with self._lock:
            if self._state not in (SolverState.IDLE, SolverState.INITIALIZED,
                                   SolverState.COMPLETED, SolverState.STOPPED):
                return False
            try:
                self.solver._initialize_field()
            except Exception:
                if hasattr(self.solver, "initialize_public"):
                    self.solver.initialize_public()
            self.monitors.clear()
            self._iteration = 0
            self._stop_reason = ""
            self._state = SolverState.INITIALIZED
            self._step_requested = False
        return True

    def run(self):
        with self._lock:
            if self._state not in (SolverState.IDLE, SolverState.INITIALIZED):
                return False
            self._state = SolverState.RUNNING
        self.events.fire("resume", state="RUNNING")
        return True

    def pause(self):
        with self._lock:
            if self._state != SolverState.RUNNING:
                return False
            self._state = SolverState.PAUSED
        self.events.fire("pause", state="PAUSED")
        return True

    def resume(self):
        with self._lock:
            if self._state != SolverState.PAUSED:
                return False
            self._state = SolverState.RUNNING
        self.events.fire("resume", state="RUNNING")
        return True

    def step(self):
        """暂停态下单步（若运行中则忽略）；写一个单步请求标志。"""
        with self._lock:
            if self._state not in (SolverState.PAUSED, SolverState.INITIALIZED,
                                   SolverState.IDLE):
                return False
            self._state = SolverState.PAUSED
            self._step_requested = True
        return True

    def stop(self):
        with self._lock:
            if self._state in SolverState.TERMINAL:
                return False
            self._state = SolverState.STOPPED
        self.events.fire("stop", state="STOPPED")
        return True

    # -- 闭环线程 ------------------------------------------------
    def _do_one_step(self):
        payload = self.solver.step()
        with self._lock:
            self._iteration += 1
            self.monitors.sample(self._iteration, payload, x=self._iteration)
            self._event_count += 1
        self.events.fire("iterate", iteration=self._iteration,
                         payload=payload, state="working")
        return payload

    def _should_stop(self):
        residual = None
        try:
            residual = self.solver.residual()
        except Exception:
            residual = None
        for crit in self._active_criteria:
            stop, reason = crit.evaluate(self._iteration, residual)
            if stop:
                return True, reason
        return False, ""

    def run_loop(self, max_iter=None, iter_callback=None):
        """阻塞迭代驱动直至终态。/ G 命令见 pause/stop/step。

        每次迭代检查状态标志 —— RUNNING 步进，PAUSED 按 step_requested 单步或
        睡眠等待，STOPPED/COMPLETED/ERROR 退出。max_iter 作为运行时附加停止准则。
        """
        self.solver._initialize_field()
        self.monitors.clear()
        self._iteration = 0
        crits = list(self.stop_criteria)
        if max_iter is not None:
            from copy import copy
            run_crit = StopCriterion(max_iterations=int(max_iter),
                                     name="run_loop")
            run_crit.start_timer()
            crits.insert(0, run_crit)
        for c in crits:
            if hasattr(c, "start_timer"):
                c.start_timer()
        self._active_criteria = crits
        self._step_requested = False
        with self._lock:
            self._state = SolverState.RUNNING
        self.events.fire("start", state="RUNNING")

        loop = True
        while loop:
            with self._lock:
                state = self._state
                step_req = bool(getattr(self, "_step_requested", False))
                self._step_requested = False
            if state in SolverState.TERMINAL:
                break
            if state == SolverState.PAUSED:
                if step_req:
                    payload = self._do_one_step()
                    self.events.fire("step", iteration=self._iteration,
                                     payload=payload)
                    if iter_callback:
                        iter_callback(self._iteration, payload)
                    # 单步后仍停留 PAUSED（继续等待 GUI 命令）
                else:
                    time.sleep(self._pause_sleep)
                continue
            # RUNNING：先判停再步进
            stop, reason = self._should_stop()
            if stop:
                with self._lock:
                    self._state = SolverState.COMPLETED
                    self._stop_reason = reason
                self.events.fire("finish", state="COMPLETED", reason=reason)
                break
            payload = self._do_one_step()
            if iter_callback:
                iter_callback(self._iteration, payload)
            self._maybe_amr()

        with self._lock:
            state = self._state
        self._result = {"state": state, "iteration": self._iteration,
                        "residual": self._safe_residual(),
                        "reason": self._stop_reason}
        return self._result

    def _safe_residual(self):
        try:
            return self.solver.residual()
        except Exception:
            return None

    def _maybe_amr(self):
        if not self._amr_interval or self._iteration % int(self._amr_interval):
            return
        try:
            from mesh_amr import run_amr
            n_before = int(getattr(self.solver, "cells", None) is not None and
                           len(self.solver.cells))
            out = run_amr(self.solver.vertices, self.solver.cells, kind="tet",
                          threshold=self._amr_threshold)
            res = None
            for r in out.get("results", []):
                res = r.get("result") or res
            if isinstance(res, dict) and res.get("ok") and res.get("vertices") is not None:
                n_after = int(len(res["cells"]))
                if n_after != n_before:
                    self.solver.set_mesh(res["vertices"], res["cells"])
                    self.solver._initialize_field()
                    self._amr_times += 1
                    self._amr_n_before = n_before
                    self._amr_n_after = n_after
                    self.events.fire("amr", n_before=n_before, n_after=n_after,
                                     fraction=out.get("fraction", 0.0))
        except Exception:
            # AMR 失败不阻断主循环
            pass

    # -- 只读快照 ------------------------------------------------
    def state(self):
        with self._lock:
            return self._state

    def metrics(self):
        with self._lock:
            return {"state": self._state,
                    "iteration": self._iteration,
                    "residual": self._safe_residual(),
                    "monitors": self.monitors.table(),
                    "reason": self._stop_reason,
                    "amr": {"times": self._amr_times,
                            "n_before": self._amr_n_before,
                            "n_after": self._amr_n_after} if self._amr_times else None}

    def curve_items(self, max_pts=512):
        with self._lock:
            return self.monitors.items(max_pts)

    def report_lines(self):
        with self._lock:
            meta = {"state": self._state, "iteration": self._iteration,
                    "residual": self._safe_residual(),
                    "reason": self._stop_reason,
                    "events": self._event_count,
                    "amr": {"times": self._amr_times,
                            "n_before": self._amr_n_before,
                            "n_after": self._amr_n_after} if self._amr_times else None}
            return Report("%s — 运行报告" % self._state.upper()).lines(
                self.monitors, meta)


# ---------------------------------------------------------------------------
# demo 网格（GUI 回退 / 测试基元）
# ---------------------------------------------------------------------------
def demo_mesh(nx=3, ny=None, nz=None):
    """生成单位立方体结构化四面体网格（每 hex 5-tet 分解，正体积）。

    返回 (vertices[N,3], cells[M,4])。nx/ny/nz 为三向单元数（缺省 = nx）。
    """
    ny = nx if ny is None else ny
    nz = nx if nz is None else nz
    vx = np.linspace(0.0, 1.0, nx + 1)
    vy = np.linspace(0.0, 1.0, ny + 1)
    vz = np.linspace(0.0, 1.0, nz + 1)
    # 节点网格
    node = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(
        nx + 1, ny + 1, nz + 1)
    verts = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                verts.append([vx[i], vy[j], vz[k]])
    V = np.asarray(verts, float)
    # 每 hex 8 角（局部序与 cube_5tet 一致）→ 5 tet
    TRI = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
           [0, 7, 4, 6], [0, 4, 1, 6]]
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n000 = node[i, j, k]
                n100 = node[i + 1, j, k]
                n110 = node[i + 1, j + 1, k]
                n010 = node[i, j + 1, k]
                n001 = node[i, j, k + 1]
                n101 = node[i + 1, j, k + 1]
                n111 = node[i + 1, j + 1, k + 1]
                n011 = node[i, j + 1, k + 1]
                local = [n000, n100, n110, n010, n001, n101, n111, n011]
                for tet in TRI:
                    cells.append([local[t] for t in tet])
    C = np.asarray(cells, np.int64)
    return V, C
