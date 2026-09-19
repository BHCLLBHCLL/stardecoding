# -*- coding: utf-8 -*-
"""S2 第 3 步 ③(a)：方形 O 型环带 + 张量积外围混合网格（mesh_hybrid）。

背景（本会话实测，非推测）：
  - 经典 O 型网格瞬态发散的阻断点是网格（非正交 median 56.6°/偏斜 p95 0.97/
    max 1.25、d_n/|d|<0.15 的面占 6%）；极坐标环带本身求解器友好。
  - 混合网格：环带外边界 = 与圆柱同心的正方形（边落在张量网格线上），外围 9 块
    张量积块与环带**逐点共线**（细分偏移 off_j = a·tan(−45°+jΔ) 与环带射线落点
    满足解析恒等式 a·cot(45°+iΔ) = a·tan(−45°+(m−i)Δ)，且直接共用同一数组）。
  - 质量校准（同一 orthogonality_report 口径）：阶梯笛卡尔 median 0°/p95 35.3°/
    skew 0.289（已知稳）；O 型 96×14 median 40.6°/p95 76.2°/skew 0.53（已知
    瞬态发散）；混合默认参数 median 32.6°/p95 69.0°/skew 0.35 且 d_n/|d|<0.15
    占比 **0%**（O 型杀手指标清零）。
  - 体积相对误差的来源**全部**是圆柱面用内接 4m 边形近似（可解析预测），拼接界面
    本身精确 —— 本文件用解析预测值做尖锐核对。
  - tet 分解的 z 长细比教训：thickness_D=0.5 时近壁棱柱内剖面对角面非正交达 ~84°
    （p95 82.8° poor）；thickness_D=0.25（z ≈ 外围 h）后 p95 69°/d_n 杀手面 0%。
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mesh_hybrid import hybrid_channel_mesh, hybrid_mesh_report  # noqa: E402
from mesh_amr import mesh_conformity  # noqa: E402
from mesh_quality import orthogonality_report  # noqa: E402


def _small(**kw):
    base = dict(D=0.04, length_D=10.0, height_D=6.0, thickness_D=0.25,
                center_x_D=3.0, h_factor=2.0, m=8, n_r=6, a_D=2.0,
                stretch=1.3, n_layers=1)
    base.update(kw)
    return hybrid_channel_mesh(**base)


def _polygon_volume(mesh):
    """解析预测体积：圆柱用内接 n_theta 边形（拼接界面精确，误差全来自多边形）。"""
    n = int(mesh["n_theta"])
    r0 = float(mesh["hole_r"])
    poly = 0.5 * n * r0 ** 2 * math.sin(2.0 * math.pi / n)
    return (float(mesh["length"]) * float(mesh["height"]) - poly) * float(mesh["thickness"])


def test_hybrid_conforming_and_exact_piecing():
    """协调性硬判据全过 + 体积与解析预测一致到 1e-9（拼接界面零失配）。"""
    m = _small()
    assert m["ok"]
    conf = mesh_conformity(m["vertices"], m["cells"])
    assert conf["conforming"], conf
    assert conf["n_dup_faces"] == 0
    assert conf["n_open_edges"] == 0
    assert conf["n_hanging_edges"] == 0
    assert m["n_negative"] == 0
    assert m["n_cells"] == 3 * m["n_prisms"]      # 每三棱柱 3 tet（n_prisms 已含 2 三角形/quad）
    pred = _polygon_volume(m)
    assert abs(m["volume"] - pred) / pred < 1e-9   # 拼接失配会被这里抓住
    # 对照：把多边形近似也算进去之前的"名义"解析值（含 π 圆）误差应 > 1e-5，
    # 证明上面的尖锐核对不是平凡通过
    assert abs(m["volume"] - m["volume_exact"]) / m["volume_exact"] > 1e-5


def test_hybrid_multiple_layers_and_validation():
    m = _small(n_layers=2)
    conf = mesh_conformity(m["vertices"], m["cells"])
    assert conf["conforming"], conf
    assert m["n_cells"] == 2 * _small()["n_cells"]
    with pytest.raises(ValueError):
        _small(a_D=6.0)                              # 正方形超出通道
    with pytest.raises(ValueError):
        _small(m=2)                                  # 每边至少 4 段


def test_hybrid_default_quality_not_poor():
    """默认参数质量：非 poor 且 d_n/|d|<0.15（O 型瞬态杀手指标）占比为 0。"""
    from fvm_core import FVM
    m = hybrid_channel_mesh(D=0.04)                  # 默认 m=24/n_r=16/s=1.3
    o = orthogonality_report(m["vertices"], m["cells"])
    assert o["ok"]
    assert o["verdict"] in ("good", "marginal"), o["reason"]
    assert o["ortho_deg"]["p95"] < 75.0             # O 型 96×14 是 76.2（已知发散）
    fv = FVM(np.asarray(m["vertices"], float), np.asarray(m["cells"], np.int64))
    mm = ~fv.is_boundary
    d = fv.centroids[fv.neighbor[mm]] - fv.centroids[fv.owner[mm]]
    L = np.linalg.norm(d, axis=1)
    frac = float((fv._d_n[mm] / np.maximum(L, 1e-300) < 0.15).mean())
    assert frac == 0.0                               # 实测：默认参数下严格为 0


def test_hybrid_thickness_aspect_quality_lever():
    """z 长细比杠杆（负结果记录）：thickness 0.5D 使近壁棱柱内剖面非正交 p95>80°，
    0.25D（z≈外围 h）降到 ~69° —— tet 分解对 z 长细比敏感的直接证据。"""
    thick = hybrid_channel_mesh(D=0.04, thickness_D=0.5)
    thin = hybrid_channel_mesh(D=0.04, thickness_D=0.25)
    ot = orthogonality_report(thick["vertices"], thick["cells"])
    on = orthogonality_report(thin["vertices"], thin["cells"])
    assert ot["ortho_deg"]["p95"] > 75.0             # 实测默认参数下 78.6°
    assert on["ortho_deg"]["p95"] < ot["ortho_deg"]["p95"] - 5.0


def test_hybrid_cylinder_and_inlet_faces():
    """圆柱边界面恰为 2·n_theta·n_layers 个三角面（run_case 的识别口径）。"""
    from fvm_core import FVM
    m = _small()
    fv = FVM(np.asarray(m["vertices"], float), np.asarray(m["cells"], np.int64))
    b = np.where(fv.is_boundary)[0]
    c2 = fv.face_centroid[b][:, :2] - np.asarray(m["hole_center"], float)[:2]
    r = np.linalg.norm(c2, axis=1)
    n_cyl = int((r <= m["hole_r"] + 1e-9).sum())
    assert n_cyl == 2 * m["n_theta"] * m["n_layers"]
    x = fv.face_centroid[b][:, 0]
    assert int((x <= x.min() + 1e-10).sum()) > 0     # 入口面存在
    assert int((x >= x.max() - 1e-10).sum()) > 0     # 出口面存在
    # 圆柱面顶点都恰在 r0 圆（内接多边形）上：半径 ∈ [r0·cos(π/n), r0]
    vr = np.linalg.norm(m["vertices"][:, :2] - np.asarray(m["hole_center"])[:2], axis=1)
    on_cyl = vr[(vr > 0.5 * m["hole_r"]) & (vr < m["hole_r"] * 1.001)]
    assert on_cyl.size >= m["n_theta"]
    assert on_cyl.max() <= m["hole_r"] + 1e-12
    assert on_cyl.min() >= m["hole_r"] * math.cos(math.pi / m["n_theta"]) - 1e-12


def test_hybrid_solver_smoke():
    """端到端冒烟：混合网格 + 非正交修正 0.33 + limited 对流，12 步有限且有界。"""
    from pressure_solver import PressureSolver
    m = _small()
    s = PressureSolver(np.asarray(m["vertices"], float),
                       np.asarray(m["cells"], np.int64),
                       rho=1.0, mu=1e-5, inlet_axis=0, inlet_side="min",
                       inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
                       convection="limited", wall_slip_axes=(1, 2),
                       nonorth_corrected=True, corr_limit=0.33)
    for _ in range(12):
        r = s.step()
    vel = s.velocity()
    assert np.isfinite(vel).all()
    assert float(np.linalg.norm(vel, axis=1).max()) < 1.0   # 有界（U=0.05）
    assert r["residual"] < 1.0


def test_run_case_hybrid_wiring(monkeypatch):
    """run_case 的 hybrid 分支与修正旋钮接线：短跑端到端可跑通。"""
    from official_diff import run_case
    monkeypatch.setenv("STARDECODING_LONG", "1")
    out = run_case(D=0.04, u_inf=0.05, steps=3, n_inner=1, dt=0.01,
                   mesher="hybrid", thickness_D=0.25, convection="limited",
                   wall_slip_axes=(1, 2), piso_correctors=1, perturb=0.05,
                   nonorth_corrected=True, corr_limit=0.33,
                   hybrid_m=8, hybrid_nr=6, hybrid_a_D=2.0,
                   length_D=10.0, height_D=6.0, center_x_D=3.0)
    assert out["ok"], out.get("reason")
    assert out["mesher"] == "hybrid"
    assert out["n_cyl_faces"] == 2 * 4 * 8 * 1      # m=8 → n_theta=32
    assert out["corrections"]["nonorth"] is True
    assert out["mesh_quality"]["verdict"] in ("good", "marginal")
    assert len(out["series"][0]) == 3
    assert out["w_absmax_final"] is not None
    # 未设长耗时门槛时必须拒绝（诚实边界）
    monkeypatch.delenv("STARDECODING_LONG", raising=False)
    out2 = run_case(mesher="hybrid", steps=1)
    assert not out2["ok"]
    assert "STARDECODING_LONG" in out2["reason"]


def test_hybrid_graded_wake_refinement():
    """外围几何渐变（stretch_far）：近尾迹分辨率翻倍而单元数不增，协调性与质量保持。"""
    from fvm_core import FVM
    base = hybrid_channel_mesh(D=0.04)
    graded = hybrid_channel_mesh(D=0.04, h_factor=8.0, stretch_far=4.0)
    assert graded["n_cells"] <= base["n_cells"], (graded["n_cells"], base["n_cells"])
    conf = mesh_conformity(graded["vertices"], graded["cells"])
    assert conf["conforming"], conf
    pred = _polygon_volume(graded)
    assert abs(graded["volume"] - pred) / pred < 1e-9
    o = orthogonality_report(graded["vertices"], graded["cells"])
    assert o["verdict"] in ("good", "marginal"), o["reason"]
    assert o["ortho_deg"]["p95"] < 75.0
    # 近尾迹（x=cx+0.5D..1.5D， wake 中心带）最细 x 间距 ≈ h=D/8
    V = np.asarray(graded["vertices"], float)
    xs = np.unique(np.round(V[(V[:, 0] > 0.16 + 0.02) & (V[:, 0] < 0.16 + 0.06)
                              & (np.abs(V[:, 1] - 0.16) < 0.02), 0], 7))
    assert float(np.diff(xs).min()) <= 0.04 / 8.0 + 1e-9
    fv = FVM(V, np.asarray(graded["cells"], np.int64))
    mm = ~fv.is_boundary
    d = fv.centroids[fv.neighbor[mm]] - fv.centroids[fv.owner[mm]]
    L = np.linalg.norm(d, axis=1)
    assert float((fv._d_n[mm] / np.maximum(L, 1e-300) < 0.15).mean()) == 0.0


def test_pressure_solver_planar_2d_flag():
    """planar_2d：w 恒 0 且解有限（默认 False 时行为不变 —— 零回归）。"""
    from pressure_solver import PressureSolver
    m = _small()
    V = np.asarray(m["vertices"], float)
    C = np.asarray(m["cells"], np.int64)
    kw = dict(rho=1.0, mu=1e-5, inlet_axis=0, inlet_side="min",
              inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
              convection="limited", wall_slip_axes=(1, 2),
              nonorth_corrected=True, corr_limit=0.33)
    s = PressureSolver(V, C, planar_2d=True, **kw)
    for _ in range(8):
        r = s.step()
    assert np.isfinite(s.velocity()).all()
    assert float(np.abs(s.velocity()[:, 2]).max()) == 0.0
    assert r["residual"] < 1.0
    s2 = PressureSolver(V, C, **kw)                 # 默认关：不回归
    assert s2.planar_2d is False
    for _ in range(4):
        s2.step()
    assert np.isfinite(s2.velocity()).all()



def test_hybrid_mesh_report_helper():
    rep = hybrid_mesh_report(_small())
    assert rep["conformity"]["conforming"]
    assert rep["orthogonality"]["ok"]
    assert rep["volume_rel_err"] is not None
