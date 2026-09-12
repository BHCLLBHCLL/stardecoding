# -*- coding: utf-8 -*-
"""star_gui_postprocess.py — V 波后处理 GUI 接线（纯逻辑桥，无 Qt）。

把 `postprocess.py`（V1–V6 纯逻辑后处理内核）接入 `star_gui.py` 的菜单与处理器：
GUI 侧只做动作注册、视口渲染与消息提示，本模块负责

- 从求解器解析 FVM 网格（`fv`/`fvm`/`_fv`，或由 `vertices`+`cells` 现场构造 FVM）；
- 会话级后处理状态（当前场 / 颜色映射 / 标量范围 / 导出目录）；
- 菜单动作 → 后处理操作的统一派发（`run_action`），返回可渲染的载荷字典。

诚实边界：无 FVM 网格、无可用标量/矢量场、无 x/y 序列、CGNS 缺 h5py、
GIF 缺 Pillow、动画无帧序列时统一返回 `{"ok": False, "message": ...}`，
不抛异常、不假装成功。

本模块为纯逻辑（无 Qt），便于无头单测；GUI 侧只做菜单接线与视口渲染。
"""

import os
import tempfile

import numpy as np

from postprocess import (DEFAULT_COLORMAP_VALUES, PostProcessor,
                         export_animation, fields_from_solver)

DEFAULT_FIELD_ORDER = ("pressure", "speed", "rho", "temperature", "T", "mach")
DEFAULT_VECTOR_ORDER = ("velocity",)
DEFAULT_GLYPHS_MAX = 2000
DEFAULT_EXPORT_DIR = "stardecoding_post"

ACTION_SPECS = {
    "Post>ColorBy": {"op": "color", "label": "标量着色", "icon": "field"},
    "Post>IsoSurface": {"op": "isosurface", "label": "等值面", "icon": "part"},
    "Post>Section": {"op": "section", "label": "剖面", "icon": "part"},
    "Post>Clip": {"op": "clip", "label": "平面裁剪", "icon": "part"},
    "Post>Threshold": {"op": "threshold", "label": "阈值零件", "icon": "part"},
    "Post>Mirror": {"op": "mirror", "label": "镜像几何", "icon": "part"},
    "Post>Glyphs": {"op": "glyphs", "label": "矢量箭头", "icon": "field"},
    "Post>Probe": {"op": "probe", "label": "探针取样", "icon": "field"},
    "Post>Line": {"op": "line", "label": "线取样", "icon": "plot"},
    "Post>PlaneSample": {"op": "plane", "label": "面取样", "icon": "field"},
    "Post>IsoVolume": {"op": "iso_volume", "label": "等值体积", "icon": "part"},
    "Post>XY": {"op": "xy", "label": "XY 曲线", "icon": "plot"},
    "Post>Histogram": {"op": "histogram", "label": "直方图", "icon": "plot"},
    "Post>Colorbar": {"op": "colorbar", "label": "色标尺", "icon": "field"},
    "Post>Legend": {"op": "legend", "label": "图例", "icon": "field"},
    "Post>ExportCSV": {"op": "export_csv", "label": "导出 CSV", "icon": "plot"},
    "Post>ExportEnSight": {"op": "export_ensight", "label": "导出 EnSight",
                           "icon": "plot"},
    "Post>ExportCGNS": {"op": "export_cgns", "label": "导出 CGNS", "icon": "plot"},
    "Post>Animate": {"op": "animate", "label": "导出动画", "icon": "plot"},
}

POST_MENU_KEYS = [
    "Post>ColorBy", "Post>IsoSurface", None,
    "Post>Section", "Post>Clip", "Post>Threshold", "Post>Mirror",
    "Post>Glyphs", None,
    "Post>Probe", "Post>Line", "Post>PlaneSample", "Post>IsoVolume", None,
    "Post>XY", "Post>Histogram", "Post>Colorbar", "Post>Legend", None,
    "Post>ExportCSV", "Post>ExportEnSight", "Post>ExportCGNS", "Post>Animate",
]


# ===========================================================================
# FVM 解析 / 几何缺省
# ===========================================================================
def solver_fv(solver):
    """按 `fv` / `fvm` / `_fv` 顺序解析求解器的 FVM 网格；无则 None。

    语义与 `postprocess._solver_fv` 一致（此处本地实现，保持桥模块自足）。
    """
    if solver is None:
        return None
    for attr in ("fv", "fvm", "_fv"):
        cand = getattr(solver, attr, None)
        if cand is not None:
            return cand
    return None


def resolve_fv(solver=None, fv=None, vertices=None, cells=None):
    """解析 FVM 网格：显式 fv > 求解器属性 > 由 vertices+cells 现场构造。

    构造路径要求严格四面体（4 列）；非四面体或网格过小返回 None。
    """
    if fv is not None:
        return fv
    got = solver_fv(solver)
    if got is not None:
        return got
    V = vertices if vertices is not None else getattr(solver, "vertices", None)
    C = cells if cells is not None else getattr(solver, "cells", None)
    if V is None or C is None:
        return None
    C = np.asarray(C, np.int64)
    if C.ndim != 2 or C.shape[1] != 4 or len(C) < 1:
        return None
    try:
        from fvm_core import FVM
        return FVM(np.asarray(V, float), C)
    except Exception:
        return None


def bbox(fv):
    """网格顶点包围盒 (lo, hi)。"""
    V = np.asarray(fv.vertices, float)
    return V.min(axis=0), V.max(axis=0)


def default_point(fv):
    """缺省取样点：包围盒中心。"""
    lo, hi = bbox(fv)
    return 0.5 * (lo + hi)


def default_normal():
    """缺省剖面/裁剪法向 +x。"""
    return np.array([1.0, 0.0, 0.0])


def match_location(fv, size):
    """标量数组长度对应的绑定位置：True=逐顶点 / False=逐单元 / None=不吻合。"""
    size = int(size)
    nv, nc = int(fv.n_vertices), int(fv.n_cells)
    if size == nv and size != nc:
        return True
    if size == nc and size != nv:
        return False
    if size == nv:
        return True
    return None


def official_colormap():
    """官方风格蓝→黄→红 4n 断点（供 GUI `lut_from_colormap` 使用）。"""
    return list(DEFAULT_COLORMAP_VALUES)


def _finite_range(arr):
    a = np.asarray(arr, float).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo, hi = float(a.min()), float(a.max())
    return lo, (hi if hi > lo else lo + 1.0)


def _ok(op, message, payload=None):
    return {"ok": True, "op": op, "message": message, "payload": payload}


def _fail(op, message):
    return {"ok": False, "op": op, "message": message, "payload": None}


# ===========================================================================
# 会话级状态
# ===========================================================================
class PostProcessSession(object):
    """V 波后处理会话：绑定 (fv, fields) 并暴露菜单动作派发。"""

    def __init__(self, solver=None, fv=None, fields=None, names=None,
                 export_dir=None):
        self.solver = solver
        self.fv = resolve_fv(solver=solver, fv=fv)
        self.names = list(names) if names else None
        self.export_dir = export_dir or os.path.join(
            tempfile.gettempdir(), DEFAULT_EXPORT_DIR)
        self.fields = {}
        if fields is not None:
            self.update_fields(fields)
        elif solver is not None and self.fv is not None:
            try:
                self.update_fields(
                    fields_from_solver(solver, fv=self.fv, names=self.names))
            except Exception:
                self.fields = {}
        if not self.fields:
            self._augment_from_solver(solver)
        self.processor = (PostProcessor(self.fv, self.fields)
                          if self.fv is not None else None)
        self.active_field = self.pick_field()
        self.active_vector = self.pick_vector()
        self.colormap = None
        self.range = None
        self.last = None

    # -- 场 / 状态 -------------------------------------------------
    def available(self):
        return self.fv is not None and self.processor is not None

    def field_names(self):
        return sorted(self.fields.keys())

    def scalar_names(self):
        return sorted(k for k, v in self.fields.items()
                      if np.asarray(v).ndim == 1)

    def vector_names(self):
        return sorted(k for k, v in self.fields.items()
                      if np.asarray(v).ndim == 2)

    def pick_field(self):
        for name in DEFAULT_FIELD_ORDER:
            if name in self.fields and np.asarray(self.fields[name]).ndim == 1:
                return name
        for name in self.scalar_names():
            return name
        return None

    def pick_vector(self):
        for name in DEFAULT_VECTOR_ORDER:
            if name in self.fields and np.asarray(self.fields[name]).ndim == 2:
                return name
        for name in self.vector_names():
            return name
        return None

    def _augment_from_solver(self, solver):
        """求解器仅暴露标量场方法（如 `field()`）时补一个 "field" 标量。

        接受逐单元或逐顶点场（GUI 默认 `DemoDiffusionSolver.field()` 为节点场）；
        节点场在阈值等仅支持单元中心的操作中由 `_cell_field` 折算。
        """
        fn = getattr(solver, "field", None)
        if not callable(fn) or self.fv is None:
            return
        try:
            arr = np.asarray(fn(), float)
        except Exception:
            return
        if arr.ndim != 1:
            return
        if arr.size in (self.fv.n_cells, self.fv.n_vertices):
            self.update_fields({"field": arr})

    def set_field(self, name, values):
        self.update_fields({name: values})
        return self

    def update_fields(self, fields):
        for key, val in (fields or {}).items():
            self.fields[str(key)] = np.asarray(val, float)
        if getattr(self, "processor", None) is not None:
            self.processor.update_fields(fields)
        self.active_field = self.pick_field()
        self.active_vector = self.pick_vector()
        return self

    def set_colormap(self, values):
        """设置会话颜色映射断点（4n 组）；None 用内核默认官方色表。"""
        self.colormap = list(values) if values else None
        return self

    def refresh(self, solver=None):
        """求解推进后重取 FVM 与场（保持当前颜色映射/范围）。"""
        solver = solver if solver is not None else self.solver
        if solver is None:
            return self
        self.solver = solver
        new_fv = resolve_fv(solver=solver)
        if new_fv is not None:
            self.fv = new_fv
            self.processor = PostProcessor(self.fv, self.fields)
        try:
            self.update_fields(
                fields_from_solver(solver, fv=self.fv, names=self.names))
        except Exception:
            pass
        if not self.fields:
            self._augment_from_solver(solver)
        return self

    # -- 渲染提示 --------------------------------------------------
    def color_payload(self, res):
        """把 color_by 结果转成 GUI 渲染提示（on_points / 标量 / 范围）。"""
        scalars = np.asarray(res["scalars"], float).reshape(-1)
        return {"scalars": scalars, "range": res["range"], "rgba": res["rgba"],
                "on_points": match_location(self.fv, scalars.size)}

    def summary(self):
        if not self.available():
            return "后处理不可用：无 FVM 网格"
        return "后处理：%d 单元 / %d 顶点，场 %s" % (
            int(self.fv.n_cells), int(self.fv.n_vertices),
            ",".join(self.field_names()) or "（无）")


# ===========================================================================
# 动作派发
# ===========================================================================
def _field_name(session, params, prefer_vector=False):
    name = params.get("field") or params.get("name")
    if name is None:
        name = session.active_vector if prefer_vector else session.active_field
    if name is None or name not in session.fields:
        return None
    return name


def _cell_field(session, arr):
    """把 1 维场折算为单元中心；顶点场取单元顶点均值。"""
    a = np.asarray(arr, float).reshape(-1)
    if a.size == session.fv.n_cells:
        return a
    if a.size == session.fv.n_vertices:
        return np.asarray([float(np.mean(a[session.fv.cells[c]]))
                           for c in range(session.fv.n_cells)], float)
    return None


def _op_color(session, params):
    name = _field_name(session, params)
    if name is None:
        return _fail("color", "无可用标量场")
    try:
        res = session.processor.color(
            name, kind=params.get("kind", "scalar"), comp=params.get("comp"),
            cmap=session.colormap, lo=params.get("lo"), hi=params.get("hi"))
    except ValueError as exc:
        return _fail("color", str(exc))
    session.active_field = name
    session.range = res["range"]
    lo, hi = res["range"]
    payload = session.color_payload(res)
    return _ok("color", "标量着色 %s ∈ [%.4g, %.4g]" % (name, lo, hi), payload)


def _op_isosurface(session, params):
    name = _field_name(session, params)
    if name is None:
        return _fail("isosurface", "无可用标量场")
    fld = np.asarray(session.fields[name], float).reshape(-1)
    lo, hi = _finite_range(fld)
    iso = params.get("iso")
    iso = 0.5 * (lo + hi) if iso is None else float(iso)
    res = session.processor.isosurface(name, iso, fields=params.get("over"))
    out = {"vertices": res["vertices"], "triangles": res["triangles"],
           "scalars": res.get("scalars", {}), "iso": res["iso"]}
    return _ok("isosurface", "等值面 %s=%.4g（%d 三角形）"
               % (name, iso, len(res["triangles"])), out)


def _op_section(session, params):
    point = params.get("point")
    normal = params.get("normal")
    point = default_point(session.fv) if point is None else point
    normal = default_normal() if normal is None else normal
    res = session.processor.section(point, normal, fields=params.get("over"))
    return _ok("section", "剖面（%d 三角形）"
               % len(res["triangles"]), res)


def _op_clip(session, params):
    point = params.get("point")
    normal = params.get("normal")
    point = default_point(session.fv) if point is None else point
    normal = default_normal() if normal is None else normal
    keep = params.get("keep", "negative")
    res = session.processor.clip(point, normal, keep=keep,
                                 fields=params.get("over"))
    return _ok("clip", "平面裁剪 keep=%s（%d 三角形）"
               % (keep, len(res["triangles"])), res)


def _op_threshold(session, params):
    name = _field_name(session, params)
    if name is None:
        return _fail("threshold", "无可用标量场")
    cell_field = _cell_field(session, session.fields[name])
    if cell_field is None:
        return _fail("threshold", "阈值需要单元中心场")
    lo, hi = params.get("lo"), params.get("hi")
    if lo is None and hi is None:
        rlo, rhi = _finite_range(cell_field)
        hi = 0.5 * (rlo + rhi)
    over = params.get("over")
    resolved = None if over is None else {
        n: session.fields[n] for n in over if n in session.fields}
    from postprocess import threshold_part
    res = threshold_part(session.fv, cell_field, lo=lo, hi=hi, fields=resolved)
    return _ok("threshold", "阈值零件 %s ∈ [%s, %s]（%d 单元）"
               % (name, lo, hi, res["count"]), res)


def _op_mirror(session, params):
    cells = np.arange(session.fv.n_cells, dtype=np.int64)
    from postprocess import extract_surface
    surf = extract_surface(session.fv, cells)
    point = params.get("point")
    normal = params.get("normal")
    normal = default_normal() if normal is None else normal
    res = session.processor.mirror(surf, point=point, normal=normal,
                                   merge=params.get("merge", True))
    return _ok("mirror", "镜像几何（%d 三角形）"
               % len(res["triangles"]), res)


def _op_glyphs(session, params):
    name = _field_name(session, params, prefer_vector=True)
    if name is None:
        return _fail("glyphs", "无可用矢量场")
    res = session.processor.glyphs(
        name, stride=params.get("stride", 1),
        max_glyphs=params.get("max_glyphs", DEFAULT_GLYPHS_MAX),
        cmap=session.colormap)
    return _ok("glyphs", "矢量箭头 %s（%d 个，scale=%.4g）"
               % (name, res["count"], res["scale"]), res)


def _op_probe(session, params):
    points = params.get("points")
    points = [default_point(session.fv)] if points is None else points
    res = session.processor.probe(points, names=params.get("names"))
    inside = int(np.sum(res["inside"]))
    return _ok("probe", "探针 %d 点（域内 %d）" % (len(res["points"]), inside), res)


def _op_line(session, params):
    a, b = params.get("a"), params.get("b")
    if a is None or b is None:
        lo, hi = bbox(session.fv)
        a = lo if a is None else a
        b = hi if b is None else b
    res = session.processor.line(a, b, n=params.get("n", 100),
                                 names=params.get("names"))
    return _ok("line", "线取样（%d 点，长 %.4g）"
               % (len(res["points"]), res["length"]), res)


def _op_plane(session, params):
    point = params.get("point")
    normal = params.get("normal")
    point = default_point(session.fv) if point is None else point
    normal = default_normal() if normal is None else normal
    res = session.processor.plane(point, normal, n=params.get("n", 40),
                                  names=params.get("names"))
    return _ok("plane", "面取样 %dx%d" % res["grid_shape"], res)


def _op_iso_volume(session, params):
    name = _field_name(session, params)
    if name is None:
        return _fail("iso_volume", "无可用标量场")
    fld = np.asarray(session.fields[name], float).reshape(-1)
    lo, hi = _finite_range(fld)
    iso = params.get("iso")
    iso = 0.5 * (lo + hi) if iso is None else float(iso)
    res = session.processor.iso_volume(name, iso, above=params.get("above", True))
    return _ok("iso_volume", "等值体积 %s %s %.4g：占比 %.4g（%d 单元）"
               % (name, "≥" if res["above"] else "≤", iso,
                  res["fraction"], res["count"]), res)


def _op_xy(session, params):
    x, y = params.get("x"), params.get("y")
    if x is None or y is None:
        return _fail("xy", "XY 曲线需要 x/y 序列")
    res = session.processor.xy(x, y)
    return _ok("xy", "XY 曲线（%d 有效点）" % res["valid"], res)


def _op_histogram(session, params):
    name = _field_name(session, params)
    if name is None:
        return _fail("histogram", "无可用标量场")
    res = session.processor.histogram(
        name, bins=params.get("bins", 20), range=params.get("range"))
    return _ok("histogram", "直方图 %s（%d 桶，均值 %.4g）"
               % (name, res["n"] and len(res["counts"]), res["mean"]), res)


def _op_colorbar(session, params):
    name = _field_name(session, params)
    if name is None:
        return _fail("colorbar", "无可用标量场")
    res = session.processor.colorbar(name, n=params.get("n", 6))
    return _ok("colorbar", "色标尺 %s [%.4g, %.4g]"
               % (name, res["lo"], res["hi"]), res)


def _op_legend(session, params):
    labels = params.get("labels")
    labels = session.field_names() if labels is None else labels
    res = session.processor.legend(labels, cmap=session.colormap)
    return _ok("legend", "图例 %d 项" % len(res), res)


def _op_export_csv(session, params):
    path = params.get("path")
    path = (os.path.join(session.export_dir, "post_fields.csv")
            if path is None else path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = session.processor.export_csv(path, names=session.names)
    return _ok("export_csv", "已导出 CSV：%s" % out, out)


def _op_export_ensight(session, params):
    out_dir = params.get("out_dir")
    out_dir = (os.path.join(session.export_dir, "ensight")
               if out_dir is None else out_dir)
    res = session.processor.export_ensight(out_dir, names=session.names)
    return _ok("export_ensight", "已导出 EnSight：%s" % res["case"], res)


def _op_export_cgns(session, params):
    path = params.get("path")
    path = (os.path.join(session.export_dir, "post.cgns")
            if path is None else path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        out = session.processor.export_cgns(path, names=session.names)
    except RuntimeError as exc:
        return _fail("export_cgns", "CGNS 导出不可用：%s" % exc)
    return _ok("export_cgns", "已导出 CGNS：%s" % out, out)


def _op_animate(session, params):
    frames = params.get("frames")
    if not frames:
        return _fail("animate", "动画需要帧序列（GUI 帧渲染待接）")
    out_dir = params.get("out_dir")
    out_dir = (os.path.join(session.export_dir, "animation")
               if out_dir is None else out_dir)
    gif = params.get("gif")
    try:
        res = export_animation(frames, out_dir, prefix=params.get("prefix", "frame"),
                               fps=params.get("fps", 10), fmt=params.get("fmt", "png"),
                               gif=gif, mp4=params.get("mp4"))
    except RuntimeError as exc:
        return _fail("animate", "动画导出降级：%s" % exc)
    return _ok("animate", "已写 %d 帧 → %s" % (res["count"], out_dir), res)


_OP_HANDLERS = {
    "color": _op_color, "isosurface": _op_isosurface, "section": _op_section,
    "clip": _op_clip, "threshold": _op_threshold, "mirror": _op_mirror,
    "glyphs": _op_glyphs, "probe": _op_probe, "line": _op_line,
    "plane": _op_plane, "iso_volume": _op_iso_volume, "xy": _op_xy,
    "histogram": _op_histogram, "colorbar": _op_colorbar, "legend": _op_legend,
    "export_csv": _op_export_csv, "export_ensight": _op_export_ensight,
    "export_cgns": _op_export_cgns, "animate": _op_animate,
}


def action_op(key):
    """动作键 → 操作名（未知返回 None）。"""
    spec = ACTION_SPECS.get(key)
    return spec["op"] if spec else None


def run_action(session, key, **params):
    """执行后处理动作，返回 {"ok","op","message","payload"}。

    无 FVM 网格或无对应处理器时诚实返回 `ok=False`。
    """
    spec = ACTION_SPECS.get(key)
    if spec is None:
        return _fail(None, "未知后处理动作：%r" % (key,))
    op = spec["op"]
    if session is None or not session.available():
        return _fail(op, "后处理不可用：请加载四面体网格或用 FVM/压力求解器")
    handler = _OP_HANDLERS.get(op)
    if handler is None:
        return _fail(op, "操作未接线：%s" % op)
    result = handler(session, params)
    session.last = result
    return result


def make_session(solver=None, fv=None, fields=None, names=None, export_dir=None):
    """构造后处理会话（`PostProcessSession` 便捷工厂）。"""
    return PostProcessSession(solver=solver, fv=fv, fields=fields, names=names,
                              export_dir=export_dir)
