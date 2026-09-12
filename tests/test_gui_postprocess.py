# -*- coding: utf-8 -*-
"""V 波 GUI 接线（纯逻辑）：FVM 解析 / 会话状态 / 19 个后处理动作派发 / 诚实降级。"""
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fvm_core import FVM, cube_tet_mesh  # noqa: E402
from star_gui_postprocess import (ACTION_SPECS, POST_MENU_KEYS,  # noqa: E402
                                  PostProcessSession, action_op, make_session,
                                  match_location, official_colormap, resolve_fv,
                                  run_action, solver_fv)


def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _fields(fv):
    return {"x": fv.centroids[:, 0],
            "speed": fv.centroids[:, 0] + 1.0,
            "velocity": np.tile(np.array([[1.0, 0.0, 0.0]]), (fv.n_cells, 1))}


def _session(nx=2, export_dir=None):
    fv = _fv(nx)
    return PostProcessSession(fv=fv, fields=_fields(fv), export_dir=export_dir)


class _Stub(object):
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# FVM 解析 / 几何助手
# ---------------------------------------------------------------------------
def test_solver_fv_attribute_priority():
    fv = _fv(1)
    assert solver_fv(None) is None
    assert solver_fv(_Stub(fv=fv)) is fv
    assert solver_fv(_Stub(fvm=fv)) is fv
    assert solver_fv(_Stub(_fv=fv)) is fv
    assert solver_fv(_Stub()) is None
    assert solver_fv(_Stub(fv=None, fvm=fv)) is fv


def test_resolve_fv_three_paths_and_reject():
    V, C = cube_tet_mesh(2)
    fv = FVM(V, C)
    assert resolve_fv(fv=fv) is fv
    assert resolve_fv(solver=_Stub(fvm=fv)) is fv
    built = resolve_fv(vertices=V, cells=C)
    assert isinstance(built, FVM)
    assert built.n_cells == fv.n_cells
    assert resolve_fv(solver=_Stub(vertices=V, cells=C)) is not None
    assert resolve_fv(vertices=V, cells=np.zeros((4, 3), np.int64)) is None
    assert resolve_fv() is None
    assert resolve_fv(vertices=V, cells=np.zeros((0, 4), np.int64)) is None


def test_match_location_and_colormap():
    fv = _fv(2)
    assert match_location(fv, fv.n_vertices) is True
    assert match_location(fv, fv.n_cells) is False
    assert match_location(fv, fv.n_cells + fv.n_vertices + 7) is None
    cm = official_colormap()
    assert len(cm) % 4 == 0 and len(cm) >= 8


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------
def test_session_available_and_field_classification():
    s = _session(2)
    assert s.available() is True
    assert s.field_names() == ["speed", "velocity", "x"]
    assert s.scalar_names() == ["speed", "x"]
    assert s.vector_names() == ["velocity"]
    assert s.active_field in ("speed", "x")
    assert s.active_vector == "velocity"
    assert s.processor.fv is s.fv
    assert "后处理" in s.summary()


def test_session_unavailable_without_fv():
    s = PostProcessSession(fv=None)
    assert s.available() is False
    assert s.processor is None
    assert "不可用" in s.summary()
    assert run_action(s, "Post>ColorBy")["ok"] is False


def test_session_augment_from_solver_field_method():
    fv = _fv(2)

    class _Solverless(object):
        def field(self):
            return np.arange(fv.n_cells, dtype=float)

    s = PostProcessSession(fv=fv, solver=_Solverless())
    assert "field" in s.fields
    assert s.fields["field"].size == fv.n_cells


def test_session_augment_from_solver_vertex_field():
    fv = _fv(2)

    class _VertexSolver(object):
        def field(self):
            return np.arange(fv.n_vertices, dtype=float)

    s = PostProcessSession(fv=fv, solver=_VertexSolver())
    assert "field" in s.fields
    assert s.fields["field"].size == fv.n_vertices


def test_run_action_threshold_on_vertex_field():
    fv = _fv(2)
    s = PostProcessSession(fv=fv, solver=_Stub(field=lambda: np.arange(
        fv.n_vertices, dtype=float)))
    assert s.fields["field"].size == fv.n_vertices
    out = run_action(s, "Post>Threshold", field="field")
    assert out["ok"], out["message"]
    assert out["payload"]["count"] > 0


def test_session_update_fields_and_refresh():
    s = _session(2)
    s.set_field("extra", np.arange(s.fv.n_cells, dtype=float))
    assert "extra" in s.fields
    s.update_fields({"speed": np.ones(s.fv.n_cells)})
    assert "speed" in s.fields
    before = s.fv
    s.refresh()
    assert s.fv is not None and s.fv.n_cells == before.n_cells


def test_color_payload_on_points():
    s = _session(2)
    res = s.processor.color("x")
    payload = s.color_payload(res)
    assert payload["on_points"] is False
    assert payload["scalars"].size == s.fv.n_cells
    s.set_field("xn", s.fv.vertices[:, 0])
    payload_v = s.color_payload(s.processor.color("xn"))
    assert payload_v["on_points"] is True


# ---------------------------------------------------------------------------
# 动作注册表
# ---------------------------------------------------------------------------
def test_action_specs_and_menu_keys_consistent():
    assert len(ACTION_SPECS) == 19
    assert action_op("Post>ColorBy") == "color"
    assert action_op("Post>ExportCGNS") == "export_cgns"
    assert action_op("Post>Nope") is None
    menu_keys = [k for k in POST_MENU_KEYS if k is not None]
    assert set(menu_keys) == set(ACTION_SPECS.keys())
    assert len(menu_keys) == len(set(menu_keys))


def test_run_action_unknown_key():
    s = _session(1)
    out = run_action(s, "Post>Unknown")
    assert out["ok"] is False and out["op"] is None
    assert "未知" in out["message"]


# ---------------------------------------------------------------------------
# 19 个动作：常规路径
# ---------------------------------------------------------------------------
def test_run_action_color_and_isosurface():
    s = _session(2)
    out = run_action(s, "Post>ColorBy")
    assert out["ok"] and out["op"] == "color"
    assert out["payload"]["scalars"].size == s.fv.n_cells
    iso = run_action(s, "Post>IsoSurface", field="x", iso=0.5)
    assert iso["ok"] and iso["payload"]["triangles"].shape[1] == 3


def test_run_action_section_clip_threshold_mirror():
    s = _session(2)
    for key in ("Post>Section", "Post>Clip"):
        out = run_action(s, key)
        assert out["ok"], out["message"]
        assert out["payload"]["triangles"].shape[1] == 3
    th = run_action(s, "Post>Threshold", field="x")
    assert th["ok"] and th["payload"]["count"] > 0
    mir = run_action(s, "Post>Mirror")
    assert mir["ok"] and mir["payload"]["mirrored"] is True


def test_run_action_glyphs_probe_line_plane_iso_volume():
    s = _session(2)
    gl = run_action(s, "Post>Glyphs")
    assert gl["ok"] and gl["payload"]["count"] > 0
    pr = run_action(s, "Post>Probe")
    assert pr["ok"] and pr["payload"]["inside"].sum() >= 0
    ln = run_action(s, "Post>Line", n=5)
    assert ln["ok"] and len(ln["payload"]["t"]) == 5
    pl = run_action(s, "Post>PlaneSample", n=4)
    assert pl["ok"] and tuple(pl["payload"]["grid_shape"]) == (4, 4)
    iv = run_action(s, "Post>IsoVolume", field="x")
    assert iv["ok"] and 0.0 <= iv["payload"]["fraction"] <= 1.0


def test_run_action_xy_histogram_colorbar_legend():
    s = _session(2)
    xy = run_action(s, "Post>XY", x=[1.0, 2.0], y=[3.0, 4.0])
    assert xy["ok"] and xy["payload"]["n"] == 2
    hi = run_action(s, "Post>Histogram", field="speed", bins=4)
    assert hi["ok"] and hi["payload"]["counts"].sum() == s.fv.n_cells
    cb = run_action(s, "Post>Colorbar", field="speed", n=5)
    assert cb["ok"] and cb["payload"]["n"] == 5
    lg = run_action(s, "Post>Legend")
    assert lg["ok"] and len(lg["payload"]) == len(s.field_names())


def test_run_action_exports_and_animate_degradation():
    tmp = tempfile.mkdtemp(prefix="gui_post_")
    try:
        s = _session(2, export_dir=tmp)
        csv = run_action(s, "Post>ExportCSV")
        assert csv["ok"] and os.path.exists(csv["payload"])
        ens = run_action(s, "Post>ExportEnSight")
        assert ens["ok"] and os.path.exists(ens["payload"]["case"])
        cgns = run_action(s, "Post>ExportCGNS")
        assert cgns["ok"] and os.path.exists(cgns["payload"])
        ani = run_action(s, "Post>Animate")
        assert ani["ok"] is False and "帧序列" in ani["message"]
        frames = [np.zeros((3, 3, 3), float)]
        ani2 = run_action(s, "Post>Animate", frames=frames, fps=5)
        assert ani2["ok"] and ani2["payload"]["count"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_action_field_moves_active_and_missing_field():
    s = _session(2)
    out = run_action(s, "Post>ColorBy", field="speed")
    assert out["ok"] and s.active_field == "speed"
    bad = run_action(s, "Post>IsoSurface", field="missing")
    assert bad["ok"] is False and bad["op"] == "isosurface"


def test_run_action_xy_requires_series():
    s = _session(2)
    out = run_action(s, "Post>XY")
    assert out["ok"] is False and "x/y" in out["message"]


def test_make_session_factory():
    fv = _fv(2)
    s = make_session(fv=fv, fields=_fields(fv))
    assert isinstance(s, PostProcessSession) and s.available()
