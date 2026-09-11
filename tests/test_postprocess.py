# -*- coding: utf-8 -*-
"""V 波：后处理深化（Post-Processing）—— 纯 numpy 后处理原语全谱系验收。

覆盖（对应 parity_100pct_plan.md §8 V1–V6）：
  V1 标量/矢量 color-by：官方色表断点解析（parse_colormap）/重采样
     （sample_colormap）/标量范围（scalar_range）/标量→RGBA（map_scalars）/
     矢量幅值（vector_magnitude）/解场取色（color_by：scalar/magnitude/component）
  几何内核：四面体重心定位（locate_cell）/场插值（sample_scalar/sample_vector）/
     单元→顶点（cell_to_vertex）
  V2 显示器几何：流线/迹线/质点（streamline/streamlines/pathline/particle_trace）、
     等值面 marching tets（marching_tets）、切片/裁剪（section_plane/clip_plane：
     内部共享面剔除）、阈值（threshold_cells/threshold_surface）、镜像
     （mirror_geometry）、矢量符号（vector_glyphs）
  V3 派生零件：probe / line_sample / plane_sample / iso_volume / threshold_part /
     plane_box_polygon / DerivedCache 命中计数
  V4 绘图数据：xy_series / histogram / cumulative_distribution / decimate_series /
     MonitorBuffer 滚动缓冲
  V5 注记/图例/色标尺/动画：colorbar_ticks / colorbar_strip / legend_items /
     annotation / frame_times / frame_indices / frame_name / write_ppm /
     write_png / write_gif / export_animation
  V6 数据写出：write_csv / export_csv / write_ensight(.case/.geo/.dat) /
     write_cgns(CGNS-HDF5 子集)
  门面：fields_from_solver / PostProcessor（V1–V6 方法）/ make_postprocessor 别名

验收核心（V 波行）：对标 star.post / Scalar & Vector Scene / Derived Part /
Plot / Color Legend / Save Animation / Export —— 纯 numpy 可复现，域外诚实降级。
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from fvm_core import FVM, cube_tet_mesh
from pressure_solver import PressureSolver
from postprocess import (
    DEFAULT_COLORMAP_VALUES, DerivedCache, MonitorBuffer, PostProcessor,
    annotation, cell_to_vertex, clip_plane, color_by, colorbar_strip,
    colorbar_ticks, cumulative_distribution, decimate_series, export_animation,
    export_csv, extract_surface, fields_from_solver, frame_indices, frame_name,
    frame_times, histogram, iso_volume, legend_items, line_sample, locate_cell,
    make_postprocessor, map_scalars, marching_tets, mirror_geometry,
    particle_trace, parse_colormap, pathline, plane_box_polygon, plane_sample,
    probe, sample_colormap, sample_scalar, sample_vector, scalar_range,
    section_plane, streamline, streamlines, threshold_cells, threshold_part,
    threshold_surface, vector_glyphs, vector_magnitude, write_csv, write_ensight,
    write_png, write_ppm, xy_series,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _norm(v):
    return float(np.sqrt(np.sum(np.asarray(v, float) ** 2)))


def _surface_area(surf):
    V = np.asarray(surf["vertices"], float)
    T = np.asarray(surf["triangles"], np.int64).reshape(-1, 3)
    tot = 0.0
    for a, b, c in T:
        pa, pb, pc = V[a], V[b], V[c]
        tot += 0.5 * _norm(np.cross(pb - pa, pc - pa))
    return tot


def _uniform(fv, vec):
    return np.tile(np.asarray(vec, float)[None, :], (fv.n_cells, 1))


# ================================================================ V1 色彩
def test_parse_colormap_default_and_flip():
    pos, rgb, alpha = parse_colormap(DEFAULT_COLORMAP_VALUES)
    assert pos is not None and rgb.shape == (9, 3)
    assert np.all(np.diff(pos) > 0.0)
    assert np.allclose(alpha, 1.0)
    assert np.allclose(rgb[0], [0.0, 0.0, 1.0])   # 蓝
    assert np.allclose(rgb[-1], [1.0, 0.0, 0.0])  # 红
    # 位置降序 → 自动翻转
    rev = []
    for i in range(8, -1, -1):
        rev.extend(DEFAULT_COLORMAP_VALUES[4 * i:4 * i + 4])
    p2, rgb2, _ = parse_colormap(rev)
    assert np.all(np.diff(p2) > 0.0)
    assert np.allclose(rgb2[0], [0.0, 0.0, 1.0])


def test_parse_colormap_invalid():
    assert parse_colormap([0.0, 1.0, 2.0]) == (None, None, None)
    assert parse_colormap(None) == (None, None, None)


def test_sample_colormap_shape_and_endpoints():
    cmap = sample_colormap(n=64)
    assert cmap.shape == (64, 4)
    assert np.allclose(cmap[0, :3], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(cmap[-1, :3], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.all((cmap >= 0.0) & (cmap <= 1.0))


def test_scalar_range_ignores_nonfinite():
    assert scalar_range([1.0, np.nan, 3.0, np.inf]) == (1.0, 3.0)
    assert scalar_range([2.0, 2.0]) == (2.0, 3.0)  # 退化 → hi=lo+1
    assert scalar_range([]) == (0.0, 1.0)


def test_map_scalars_endpoints_and_clamp():
    rg = map_scalars(np.array([0.0, 1.0]), lo=0.0, hi=1.0)
    assert rg.shape == (2, 4)
    assert np.allclose(rg[0, :3], [0.0, 0.0, 1.0])
    assert np.allclose(rg[1, :3], [1.0, 0.0, 0.0])
    # 越界截断
    over = map_scalars(np.array([10.0]), lo=0.0, hi=1.0)
    assert np.allclose(over[0], rg[1])


def test_vector_magnitude():
    m = vector_magnitude(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(m, [5.0, 0.0])
    assert np.allclose(vector_magnitude(np.array([1.0, 2.0])), [1.0, 2.0])


def test_color_by_variants():
    vec = np.array([[3.0, 4.0, 0.0], [0.0, 6.0, 8.0]])
    mag = color_by(vec, kind="magnitude")
    assert set(mag) == {"rgba", "scalars", "range"}
    assert np.allclose(mag["scalars"], [5.0, 10.0])
    assert mag["rgba"].shape == (2, 4)
    comp = color_by(vec, kind="component", comp="y")
    assert np.allclose(comp["scalars"], [4.0, 6.0])
    with pytest.raises(ValueError):
        color_by(np.array([1.0, 2.0]), kind="component", comp="x")


# ================================================================ 几何内核
def test_locate_cell_inside_and_outside():
    fv = _fv(2)
    c, w = locate_cell(fv, (0.5, 0.5, 0.5))
    assert c >= 0 and w is not None
    assert np.isclose(w.sum(), 1.0, atol=1e-9)
    c2, w2 = locate_cell(fv, (2.0, 2.0, 2.0))
    assert c2 == -1 and w2 is None


def test_sample_scalar_linear_and_outside():
    fv = _fv(3)
    nodal = fv.vertices[:, 0]
    got = sample_scalar(fv, nodal, [[0.25, 0.5, 0.5], [2.0, 0.0, 0.0]])
    assert np.isclose(got[0], 0.25, atol=1e-9)
    assert np.isnan(got[1])


def test_sample_vector_outside_nan():
    fv = _fv(2)
    vec = np.column_stack([np.ones(fv.n_vertices), np.zeros(fv.n_vertices),
                           np.zeros(fv.n_vertices)])
    out = sample_vector(fv, vec, [[0.5, 0.5, 0.5], [9.0, 9.0, 9.0]])
    assert np.allclose(out[0], [1.0, 0.0, 0.0])
    assert np.all(np.isnan(out[1]))


def test_cell_to_vertex_constant_preserved():
    fv = _fv(2)
    nodal = cell_to_vertex(fv, np.full(fv.n_cells, 3.5))
    assert nodal.shape == (fv.n_vertices,)
    assert np.allclose(nodal, 3.5)


# ================================================================ V2 显示器
def test_streamline_uniform_field():
    fv = _fv(2)
    out = streamline(fv, _uniform(fv, (1.0, 0.0, 0.0)), (0.1, 0.5, 0.5),
                     max_steps=10)
    V = out["vertices"]
    assert V.shape[1] == 3 and len(V) > 1
    assert np.all(np.diff(V[:, 0]) > 0.0)
    assert np.allclose(V[:, 1], 0.5, atol=1e-9)
    assert np.allclose(V[:, 2], 0.5, atol=1e-9)


def test_streamlines_multi_and_bidirectional():
    fv = _fv(2)
    vel = _uniform(fv, (0.0, 1.0, 0.0))
    out = streamlines(fv, vel, [[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], max_steps=6)
    assert len(out["lines"]) == 2
    bi = streamlines(fv, vel, [[0.5, 0.5, 0.5]], max_steps=6,
                     bidirectional=True)
    assert len(bi["lines"][0]["vertices"]) > len(out["lines"][0]["vertices"])


def test_pathline_unsteady_snapshots():
    fv = _fv(2)
    snaps = [_uniform(fv, (1.0, 0.0, 0.0)), _uniform(fv, (1.0, 0.0, 0.0))]
    out = pathline(fv, snaps, (0.1, 0.5, 0.5), times=[0.0, 1.0], max_steps=8)
    V = out["vertices"]
    assert V[-1, 0] > V[0, 0]
    empty = pathline(fv, [], (0.1, 0.5, 0.5))
    assert len(empty["vertices"]) == 0


def test_particle_trace_ids_and_count():
    fv = _fv(2)
    out = particle_trace(fv, _uniform(fv, (1.0, 0.0, 0.0)),
                         [[0.1, 0.5, 0.5], [0.2, 0.3, 0.3]], n_steps=5)
    assert out["vertices"].shape == (10, 3)
    assert out["ids"].tolist() == [0, 1] * 5


def test_marching_tets_plane_area_and_orientation():
    fv = _fv(3)
    nodal = fv.vertices[:, 0]
    surf = marching_tets(fv, nodal, 0.5)
    assert np.isclose(_surface_area(surf), 1.0, atol=1e-6)
    assert np.allclose(surf["vertices"][:, 0], 0.5, atol=1e-9)
    assert np.isclose(np.mean(surf["scalars"]["scalar"]), 0.5, atol=1e-9)


def test_section_plane_area_and_meta():
    fv = _fv(3)
    sec = section_plane(fv, (0.5, 0.0, 0.0), (2.0, 0.0, 0.0),
                        fields={"x": fv.vertices[:, 0]})
    assert np.isclose(_surface_area(sec), 1.0, atol=1e-6)
    assert np.allclose(sec["normal"], [1.0, 0.0, 0.0])
    assert np.allclose(sec["vertices"][:, 0], 0.5, atol=1e-9)
    assert np.allclose(sec["scalars"]["x"], 0.5, atol=1e-9)
    with pytest.raises(ValueError):
        section_plane(fv, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_clip_plane_removes_internal_faces():
    fv = _fv(3)
    neg = clip_plane(fv, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), keep="negative")
    pos = clip_plane(fv, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), keep="positive")
    assert np.isclose(_surface_area(neg), 4.0, atol=1e-6)
    assert np.isclose(_surface_area(pos), 4.0, atol=1e-6)
    assert neg["vertices"][:, 0].max() <= 0.5 + 1e-9
    assert pos["vertices"][:, 0].min() >= 0.5 - 1e-9
    full = clip_plane(fv, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), keep="negative")
    assert np.isclose(_surface_area(full), 6.0, atol=1e-6)
    empty = clip_plane(fv, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       keep="negative")
    assert len(empty["vertices"]) == 0 and _surface_area(empty) == 0.0
    with pytest.raises(ValueError):
        clip_plane(fv, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_threshold_cells_and_surface():
    fv = _fv(2)
    fld = fv.centroids[:, 0]
    cells = threshold_cells(fv, fld, lo=0.5)
    assert len(cells) > 0
    assert np.all(fld[cells] >= 0.5)
    surf = threshold_surface(fv, fld, lo=0.5)
    assert np.array_equal(surf["cells"], cells)
    assert surf["count"] == len(cells)
    assert len(surf["vertices"]) > 0
    with pytest.raises(ValueError):
        threshold_cells(fv, np.zeros(fv.n_cells + 1), lo=0.0)


def test_mirror_geometry_reflect_and_merge():
    surf = {"vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]]),
            "triangles": np.array([[0, 1, 2]], np.int64),
            "scalars": {"s": np.array([1.0, 2.0, 3.0])}}
    m = mirror_geometry(surf, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    assert np.allclose(m["vertices"][:, 0], [0.0, -1.0, 0.0])
    assert np.array_equal(m["triangles"], [[2, 1, 0]])
    mm = mirror_geometry(surf, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
                         merge=True)
    assert len(mm["vertices"]) == 6 and len(mm["triangles"]) == 2
    assert np.allclose(mm["scalars"]["s"], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mirror_geometry(surf, normal=(0.0, 0.0, 0.0))


def test_vector_glyphs_geometry():
    fv = _fv(2)
    vel = np.zeros((fv.n_cells, 3))
    vel[:, 2] = 2.0
    g = vector_glyphs(fv, vel)
    assert g["count"] == fv.n_cells
    assert g["vectors"].shape == (fv.n_cells, 3)
    assert np.allclose(g["scalars"], 2.0)
    assert np.allclose(g["tips"], g["points"] + g["scale"] * vel)
    assert g["scale"] > 0.0
    assert g["rgba"].shape == (fv.n_cells, 4)
    strided = vector_glyphs(fv, vel, stride=2)
    assert strided["count"] == len(range(0, fv.n_cells, 2))


def test_extract_surface_full_cube_area():
    fv = _fv(3)
    surf = extract_surface(fv, np.arange(fv.n_cells))
    assert np.isclose(_surface_area(surf), 6.0, atol=1e-6)


# ================================================================ V3 派生零件
def test_probe_inside_outside_values():
    fv = _fv(2)
    out = probe(fv, {"x": fv.vertices[:, 0]},
                [[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])
    assert out["inside"].tolist() == [True, False]
    assert out["cells"][1] == -1
    assert np.isclose(out["values"]["x"][0], 0.5, atol=1e-9)
    assert np.isnan(out["values"]["x"][1])


def test_line_sample_values_equal_t():
    fv = _fv(3)
    out = line_sample(fv, {"x": fv.vertices[:, 0]},
                      (0.0, 0.5, 0.5), (1.0, 0.5, 0.5), n=5)
    assert np.allclose(out["t"], np.linspace(0.0, 1.0, 5))
    assert np.allclose(out["values"]["x"], out["t"], atol=1e-9)
    assert np.isclose(out["length"], 1.0)


def test_plane_sample_grid_shape():
    fv = _fv(2)
    out = plane_sample(fv, {"x": fv.vertices[:, 0]},
                       (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), n=4)
    assert out["grid_shape"] == (4, 4)
    assert len(out["points"]) == 16
    assert np.allclose(out["points"][:, 2], 0.5)
    assert out["inside"].all()
    assert np.allclose(out["values"]["x"], out["points"][:, 0], atol=1e-9)
    assert np.allclose(out["normal"], [0.0, 0.0, 1.0])


def test_iso_volume_partition_and_monotone():
    fv = _fv(3)
    nodal = fv.vertices[:, 0]
    above = iso_volume(fv, nodal, 0.6, above=True)
    below = iso_volume(fv, nodal, 0.6, above=False)
    assert np.isclose(above["fraction"] + below["fraction"], 1.0, atol=1e-9)
    assert np.isclose(above["volume"] / np.sum(fv.volumes), 1.0 / 3.0,
                      atol=1e-9)
    assert above["count"] + below["count"] == fv.n_cells
    assert iso_volume(fv, nodal, 0.3, True)["fraction"] > \
        iso_volume(fv, nodal, 0.6, True)["fraction"]


def test_threshold_part_has_volume():
    fv = _fv(2)
    part = threshold_part(fv, fv.centroids[:, 0], lo=0.5)
    assert {"cells", "count", "volume", "lo", "hi"} <= set(part)
    assert np.isclose(part["volume"],
                      np.sum(fv.volumes[part["cells"]]), atol=1e-12)


def test_plane_box_polygon_intersection():
    poly = plane_box_polygon([0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                             (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert poly.shape == (4, 3)
    assert np.allclose(poly[:, 0], 0.5)
    none = plane_box_polygon([0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                             (2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert none.shape == (0, 3)


def test_derived_cache_hit_miss():
    cache = DerivedCache()
    calls = []

    def factory():
        calls.append(1)
        return "value"

    a = cache.get_or_compute("iso", {"iso": 0.5}, factory)
    b = cache.get_or_compute("iso", {"iso": 0.5}, factory)
    assert a == b == "value" and len(calls) == 1
    assert cache.hits == 1 and cache.misses == 1 and len(cache) == 1
    cache.get_or_compute("iso", {"iso": 0.7}, factory)
    assert cache.misses == 2 and len(cache) == 2
    cache.clear()
    assert cache.hits == 0 and cache.misses == 0 and len(cache) == 0


# ================================================================ V4 绘图数据
def test_xy_series_stats():
    out = xy_series([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    assert out["n"] == 3 and out["valid"] == 3
    assert out["x_range"] == (0.0, 2.0) and out["y_range"] == (0.0, 4.0)
    bad = xy_series([0.0, 1.0, 2.0], [0.0, np.nan, 4.0])
    assert bad["valid"] == 2 and bad["y_range"] == (0.0, 4.0)


def test_histogram_and_density():
    h = histogram(np.arange(10.0), bins=5)
    assert h["counts"].sum() == 10 and h["n"] == 10
    assert len(h["edges"]) == 6 and len(h["centers"]) == 5
    hd = histogram(np.arange(10.0), bins=5, density=True)
    assert np.isclose(np.sum(hd["pdf"]) * hd["bin_width"], 1.0, atol=1e-9)
    empty = histogram([], bins=4)
    assert empty["n"] == 0 and empty["counts"].shape == (4,)


def test_cumulative_distribution_monotone():
    c = cumulative_distribution(np.linspace(0.0, 1.0, 50), bins=10)
    assert np.isclose(c["cdf"][-1], 1.0)
    assert np.all(np.diff(c["cdf"]) >= 0.0)
    assert len(c["x"]) == 10


def test_decimate_series_reduces_and_keeps_ends():
    x = np.arange(1000.0)
    y = np.sin(x * 0.01)
    out = decimate_series(x, y, max_points=64)
    assert out["original"] == 1000
    assert 2 <= out["n"] < 1000
    assert out["x"][0] == 0.0 and out["x"][-1] == 999.0
    small = decimate_series(np.arange(10.0), np.arange(10.0), max_points=64)
    assert small["n"] == 10


def test_monitor_buffer_rolling_window():
    buf = MonitorBuffer(["a", "b"], capacity=4)
    for i in range(6):
        buf.append(i, {"a": i * 10.0, "b": -float(i)})
    assert len(buf) == 4
    assert buf.iterations == [2.0, 3.0, 4.0, 5.0]
    s = buf.series("a")
    assert np.allclose(s["y"], [20.0, 30.0, 40.0, 50.0])
    _, arr = buf.arrays()
    assert arr["b"].shape == (4,)
    with pytest.raises(KeyError):
        buf.series("nope")
    buf.clear()
    assert len(buf) == 0


# ================================================================ V5 注记/动画
def test_colorbar_ticks_and_strip():
    cb = colorbar_ticks(0.0, 1.0, n=6)
    assert cb["n"] == len(cb["labels"]) == 6
    assert np.isclose(cb["values"][0], 0.0) and np.isclose(cb["values"][-1], 1.0)
    strip = colorbar_strip(samples=8)
    assert strip["rgba"].shape == (8, 4) and strip["height"] == 8


def test_legend_items_and_annotation():
    items = legend_items(["alpha", "beta", "gamma"])
    assert len(items) == 3
    assert all(set(it) == {"label", "rgba"} for it in items)
    assert all(np.asarray(it["rgba"]).shape == (4,) for it in items)
    ann = annotation("iter=10", position=(0.1, 0.9), size=12, align="right")
    assert ann["text"] == "iter=10" and ann["size"] == 12
    assert ann["position"] == (0.1, 0.9) and ann["align"] == "right"


def test_frame_helpers():
    assert np.allclose(frame_times(3, 0.5, start=1.0), [1.0, 1.5, 2.0])
    assert frame_indices(3, start=5, step=2).tolist() == [5, 7, 9]
    assert frame_name("run", 1) == "run_0001.png"
    assert frame_name("run", 12, digits=2, ext="ppm") == "run_12.ppm"


def test_write_png_and_ppm():
    tmp = tempfile.mkdtemp(prefix="pp_v5_")
    try:
        img = np.zeros((2, 3, 3), float)
        img[:, :, 0] = 1.0
        png = os.path.join(tmp, "a.png")
        write_png(png, img)
        assert open(png, "rb").read()[:8] == b"\x89PNG\r\n\x1a\n"
        ppm = os.path.join(tmp, "a.ppm")
        write_ppm(ppm, img)
        assert open(ppm, "rb").read()[:2] == b"P6"
        with pytest.raises(ValueError):
            write_png(os.path.join(tmp, "b.png"), np.zeros((2, 2)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_gif_requires_pillow():
    pytest.importorskip("PIL")
    from postprocess import write_gif
    tmp = tempfile.mkdtemp(prefix="pp_v5_")
    try:
        frames = [np.zeros((4, 4, 3), float), np.ones((4, 4, 3), float)]
        path = os.path.join(tmp, "anim.gif")
        write_gif(path, frames, duration=50)
        assert os.path.exists(path) and os.path.getsize(path) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_export_animation_sequence_and_mp4_honest():
    tmp = tempfile.mkdtemp(prefix="pp_v5_")
    try:
        frames = [np.zeros((4, 4, 3), float), np.ones((4, 4, 3), float)]
        out = export_animation(frames, os.path.join(tmp, "frames"), prefix="f",
                               fps=5, mp4="x.mp4")
        assert out["count"] == 2
        assert all(os.path.exists(p) for p in out["frames"])
        assert out["frames"][0].endswith("f_0000.png")
        assert out["mp4"] == "unavailable(ffmpeg)"
        assert out["gif"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================ V6 数据写出
def test_write_csv_dict_and_array():
    tmp = tempfile.mkdtemp(prefix="pp_v6_")
    try:
        p = os.path.join(tmp, "a.csv")
        write_csv(p, {"a": [1.0, 2.0], "b": [3.0, 4.0]})
        lines = open(p, encoding="utf-8").read().strip().splitlines()
        assert lines[0] == "a,b" and len(lines) == 3
        arr = np.arange(6.0).reshape(3, 2)
        p2 = os.path.join(tmp, "b.csv")
        write_csv(p2, arr, headers=["u", "v"])
        assert open(p2, encoding="utf-8").read().splitlines()[0] == "u,v"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_export_csv_contains_geometry():
    fv = _fv(2)
    tmp = tempfile.mkdtemp(prefix="pp_v6_")
    try:
        p = os.path.join(tmp, "cells.csv")
        export_csv(p, fv, {"speed": np.ones(fv.n_cells)})
        text = open(p, encoding="utf-8").read()
        header = text.splitlines()[0].split(",")
        assert {"cell", "x", "y", "z", "speed"} <= set(header)
        assert len(text.strip().splitlines()) == fv.n_cells + 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_ensight_files():
    fv = _fv(2)
    tmp = tempfile.mkdtemp(prefix="pp_v6_")
    try:
        out = write_ensight(os.path.join(tmp, "ens"), fv,
                            fields={"speed": np.ones(fv.n_cells)})
        assert os.path.exists(out["case"]) and os.path.exists(out["geo"])
        assert len(out["dat"]) == 1 and os.path.exists(out["dat"][0])
        assert "ensight gold" in open(out["case"], encoding="ascii").read()
        geo = open(out["geo"], encoding="ascii").read()
        assert "tetra4" in geo and str(fv.n_vertices) in geo
        assert out["variables"] == ["speed"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_cgns_roundtrip():
    h5py = pytest.importorskip("h5py")
    from postprocess import write_cgns
    fv = _fv(2)
    tmp = tempfile.mkdtemp(prefix="pp_v6_")
    try:
        path = os.path.join(tmp, "mesh.cgns")
        write_cgns(path, fv, fields={"speed": np.ones(fv.n_cells)})
        with h5py.File(path, "r") as f:
            zone = f["Base"]["Zone"]
            assert bytes(np.asarray(zone["ZoneType"][()])) == b"Unstructured"
            conn = zone["Elements"]["ElementConnectivity"]
            assert conn.shape == (fv.n_cells, 4)
            assert conn[()].min() == 1
            assert "speed" in zone["FlowSolution"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================ 门面 / 工厂
def _pp(nx=2):
    fv = _fv(nx)
    fields = {"x": fv.centroids[:, 0], "speed": fv.centroids[:, 0],
              "velocity": _uniform(fv, (1.0, 0.0, 0.0))}
    return PostProcessor(fv, fields), fv


def test_postprocessor_v1_v3():
    pp, fv = _pp(3)
    assert pp.color("speed")["rgba"].shape == (fv.n_cells, 4)
    assert pp.color("velocity", kind="magnitude")["scalars"].shape == \
        (fv.n_cells,)
    assert len(pp.streamline((0.1, 0.5, 0.5), max_steps=6)["vertices"]) > 1
    assert len(pp.streamlines([[0.5, 0.5, 0.5]], max_steps=4)["lines"]) == 1
    iso = pp.isosurface("x", 0.5)
    assert iso["triangles"].shape[1] == 3 and len(iso["vertices"]) > 0
    pp.set_field("xn", fv.vertices[:, 0])
    iso_n = pp.isosurface("xn", 0.5)
    assert iso_n["triangles"].shape[1] == 3 and len(iso_n["vertices"]) > 0
    assert np.isclose(_surface_area(pp.section((0.5, 0.0, 0.0),
                                               (1.0, 0.0, 0.0))), 1.0,
                      atol=1e-6)
    assert np.isclose(_surface_area(pp.clip((0.5, 0.0, 0.0),
                                            (1.0, 0.0, 0.0))), 4.0,
                      atol=1e-6)
    assert pp.threshold("x", lo=0.5)["count"] > 0
    assert pp.glyphs("velocity")["count"] > 0
    assert pp.probe([[0.5, 0.5, 0.5]])["inside"][0]
    assert len(pp.line((0.0, 0.5, 0.5), (1.0, 0.5, 0.5), n=4)["t"]) == 4
    assert pp.plane((0.0, 0.0, 0.5), (0.0, 0.0, 1.0), n=4)["grid_shape"] == \
        (4, 4)
    assert 0.0 <= pp.iso_volume("x", 0.6, above=True)["fraction"] <= 1.0
    assert pp.mirror({"vertices": np.zeros((3, 3)), "triangles":
                      np.array([[0, 1, 2]])})["mirrored"] is True
    calls = []
    assert pp.derived("d", {"k": 1}, lambda: calls.append(1) or 5) == 5
    pp.derived("d", {"k": 1}, lambda: 9)
    assert len(calls) == 1


def test_postprocessor_v4_v6():
    pp, fv = _pp(2)
    tmp = tempfile.mkdtemp(prefix="pp_v46_")
    try:
        assert pp.xy([1.0, 2.0], [3.0, 4.0])["n"] == 2
        assert pp.histogram("speed", bins=4)["counts"].sum() == fv.n_cells
        assert np.isclose(pp.cdf("speed", bins=4)["cdf"][-1], 1.0)
        assert pp.colorbar("speed", n=5)["n"] == 5
        assert len(pp.legend(["a", "b"])) == 2
        assert pp.animate([np.zeros((3, 3, 3), float)],
                          os.path.join(tmp, "anim"), prefix="z")["count"] == 1
        csv_path = os.path.join(tmp, "f.csv")
        pp.export_csv(csv_path, names=["speed"])
        assert "speed" in open(csv_path, encoding="utf-8").read().splitlines()[0]
        ens = pp.export_ensight(os.path.join(tmp, "ens"), names=["speed"])
        assert os.path.exists(ens["case"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_postprocessor_set_and_update_fields():
    pp, fv = _pp(2)
    pp.set_field("extra", np.arange(fv.n_cells, dtype=float))
    assert "extra" in pp.fields
    pp.update_fields({"extra2": np.ones(fv.n_cells)})
    assert "extra2" in pp.fields
    with pytest.raises(KeyError):
        pp.color("missing")


def test_fields_from_solver_and_facade_from_solver():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    for _ in range(2):
        s.step()
    fld = fields_from_solver(s)
    assert {"velocity", "speed", "pressure", "rho"} <= set(fld)
    assert fld["velocity"].shape == (s.fvm.n_cells, 3)
    assert fld["speed"].shape == (s.fvm.n_cells,)
    only = fields_from_solver(s, names=["velocity", "speed"])
    assert set(only) == {"velocity", "speed"}
    pp = PostProcessor.from_solver(s)
    assert pp.fv is s.fvm
    assert pp.color("speed")["rgba"].shape == (s.fvm.n_cells, 4)
    with pytest.raises(ValueError):
        fields_from_solver(object())


def test_make_postprocessor_aliases_and_solver():
    fv = _fv(2)
    for name in ("postprocess", "Post-Processing", "DISPLAY", "Visualisation",
                 "post", "post_processor"):
        pp = make_postprocessor(fv, fields={"x": fv.vertices[:, 0]},
                                model=name)
        assert isinstance(pp, PostProcessor)
    with pytest.raises(ValueError):
        make_postprocessor(fv, model="nope")
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    for _ in range(2):
        s.step()
    pp = make_postprocessor(solver=s)
    assert isinstance(pp, PostProcessor) and pp.fv is s.fvm
    assert "velocity" in pp.fields
    with pytest.raises(ValueError):
        make_postprocessor()
