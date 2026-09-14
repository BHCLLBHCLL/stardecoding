# -*- coding: utf-8 -*-
"""R 波 R4：官方解差分（官方参考量 / 同工况几何 / 差分指标 / 长耗时门控）。

覆盖：
  Strouhal：合成 0.22 Hz 正弦 → St≈0.176（FFT 与过零双路）；短序列/非法步长诚实拒绝
  几何：环形域表面水密 + 面积/体积对解析值误差 < 2%%（三角化离散）
  指标：St 比/振幅比/平均升力带内带外判定；缺参考量时 ok=None 不假装
  参考：力系数报告参考量解析（合成对象图）
  语料：vortexShed_tutor_v3_0.05_2502（U=0.05/ρ=1/A=0.04 → D=0.04、Re=200、
        St≈0.176、Continuity 末值 3.2e-10）—— 内存不足时诚实跳过
  门控：run_case 未设 STARDECODING_LONG=1 时拒绝并说明
  O 型网格（S4 修复）：棱柱→tet 对角线按全局编号规范化 → 网格协调（开口边/重复面/T 型
        全 0、边界面数符合几何），细网格（96×14 / 192×28）求解器可稳定推进
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import official_diff as od  # noqa: E402
from mesh_amr import mesh_conformity  # noqa: E402

BENCH = r"D:/training/openfoam/benchmark"
VORTEX = os.path.join(BENCH, "vortexShed_tutor_v3_0.05_2502.sim")
needs_vortex = pytest.mark.skipif(not os.path.isfile(VORTEX), reason="语料缺失")


# ---------------------------------------------------------------- Strouhal
def test_strouhal_synthetic_matches_official():
    t = np.linspace(0.0, 200.0, 20000)
    y = 0.28 * np.sin(2 * np.pi * 0.22 * t) + 0.01
    st = od.strouhal_from_series(t, y, 0.04, 0.05)
    assert st["ok"] and abs(st["st"] - 0.176) < 3e-3
    assert abs(st["st_fft"] - 0.176) < 3e-3 and abs(st["st_cross"] - 0.176) < 3e-3
    assert abs(st["amplitude"] - 0.28) < 1e-3
    assert abs(st["mean"] - 0.01) < 1e-3 and st["n"] == 20000


def test_strouhal_short_series_and_bad_dt_are_honest():
    assert od.strouhal_from_series(np.arange(5.0), np.arange(5.0), 0.04, 0.05)["ok"] is False
    t = np.zeros(32)
    assert od.strouhal_from_series(t, np.random.default_rng(0).random(32), 0.04,
                                   0.05)["ok"] is False


def test_strouhal_handles_unsorted_time():
    t = np.linspace(0.0, 100.0, 10000)
    y = np.sin(2 * np.pi * 0.2 * t)
    idx = np.random.default_rng(1).permutation(t.size)
    st = od.strouhal_from_series(t[idx], y[idx], 0.04, 0.05)
    assert st["ok"] and abs(st["st"] - 0.2 * 0.04 / 0.05) < 5e-3


# ---------------------------------------------------------------- 几何
def test_annulus_surface_watertight_and_analytic():
    s = od.annulus_surface(0.04, r_far_factor=10.0, n_theta=96, n_r=14)
    assert s["ok"] and s["watertight"]
    assert s["triangles"].shape[0] == 4 * 96 * (14 + 1)
    assert abs(s["area"] / s["area_exact"] - 1.0) < 0.005
    assert abs(s["volume"] / s["volume_exact"] - 1.0) < 0.02
    assert s["r_far"] == pytest.approx(0.4) and s["thickness"] == pytest.approx(0.04)


def test_annulus_surface_scales_with_parameters():
    a = od.annulus_surface(0.1, r_far_factor=6.0, thickness=0.5, n_theta=48, n_r=8)
    assert a["watertight"] and abs(a["volume"] / a["volume_exact"] - 1.0) < 0.02
    big = od.annulus_surface(0.1, r_far_factor=6.0, thickness=0.5, n_theta=96, n_r=16)
    assert big["triangles"].shape[0] > a["triangles"].shape[0]
    assert abs(big["volume"] / big["volume_exact"] - 1.0) < 0.01


# ---------------------------------------------------------------- 指标
def test_diff_metrics_within_and_outside_bands():
    ref = {"strouhal": {"st": 0.176, "amplitude": 0.28}}
    good = od.diff_metrics({"st": 0.18, "amplitude": 0.30, "mean": 0.001}, ref)
    assert good["items"]["strouhal"]["ok"] and good["items"]["amplitude"]["ok"]
    assert good["items"]["mean_lift"]["ok"]
    bad = od.diff_metrics({"st": 0.06, "amplitude": 0.01, "mean": 0.5}, ref)
    assert bad["items"]["strouhal"]["ok"] is False
    assert bad["items"]["amplitude"]["ok"] is False
    assert bad["items"]["mean_lift"]["ok"] is False


def test_diff_metrics_without_reference_is_none_not_fake_pass():
    out = od.diff_metrics({"st": 0.18}, {"strouhal": None})
    assert out["items"]["strouhal"]["ok"] is None
    assert "reason" in out["items"]["strouhal"]


# ---------------------------------------------------------------- 参考量解析
def test_report_reference_synthetic_object_graph():
    class _Q:
        def __init__(self, val=None, vec=None):
            self.dict = {"Value": val} if val is not None else {"Vector": vec}

    class _O:
        def __init__(self, cn, d):
            self.class_name, self.dict, self.name = cn, d, "lift"

    class _Sim:
        objmap = {1: _Q(0.05), 2: _Q(1.0), 3: _Q(0.04), 4: _Q(vec=[0.0, 1.0, 0.0])}
        objects = [_O("star.base.report.ForceCoefficientReport",
                      {"ReferenceVelocity": 1, "ReferenceDensity": 2,
                       "ReferenceArea": 3, "Direction": 4})]

    rep = od.report_reference(_Sim())
    assert rep["ok"] and rep["u_ref"] == 0.05 and rep["rho_ref"] == 1.0
    assert rep["area_ref"] == 0.04 and rep["direction"] == [0.0, 1.0, 0.0]
    assert od.report_reference(type("S", (), {"objects": []})())["ok"] is False


# ---------------------------------------------------------------- 语料/门控
@needs_vortex
def test_corpus_official_reference():
    from sim_parser import SimFile
    try:
        sim = SimFile(VORTEX)
        ref = od.reference_case(sim)
    except MemoryError:
        pytest.skip("内存不足（本机另有大型 CAE 进程占用）")
    assert ref["ok"], ref.get("reason")
    assert ref["u_ref"] == 0.05 and ref["rho_ref"] == 1.0 and ref["area_ref"] == 0.04
    assert ref["diameter"] == pytest.approx(0.04, abs=1e-9)
    assert ref["reynolds"] == pytest.approx(200.0, rel=1e-6)
    st = ref["strouhal"]
    assert st["ok"] and 0.16 < st["st"] < 0.19, st
    assert abs(st["st"] - 0.176) < 0.01
    assert 0.2 < st["amplitude"] < 0.35
    assert ref["residual"]["final"] < 1e-8


# ---------------------------------------------------------------- 同工况结构化网格（自研瞬态用）
def test_channel_tet_mesh_ogrid_volume_and_orientation():
    m = od.channel_tet_mesh(0.04, n_theta=24, n_r=4)
    assert m["ok"] and m["n_cells"] == 6 * 24 * 4
    assert m["n_negative"] == 0 and m["quality_proxy"]["min_vol"] > 0
    assert abs(m["volume"] / m["volume_exact"] - 1.0) < 0.05


def test_channel_ogrid_is_conforming_and_seam_faces_are_shared():
    """S4 修复回归守卫：棱柱→tet 的三条侧面对角线必须按**全局顶点编号**规范化。

    修复前按局部次序取对角线 → θ 周期缝合面两侧取到不同对角线（
    bottom(i,0)–top(i+1,0) vs bottom(i+1,0)–top(i,0)）→ 内部面配不上
    （n_theta=48 实测 1348 个内部单侧面）→ 细网格压力矩阵奇异（dgstrf info/NaN）。
    """
    from mesh_amr import _cell_faces
    for nth, nr in ((24, 4), (48, 7), (96, 14)):
        m = od.channel_tet_mesh(0.04, n_theta=nth, n_r=nr)
        conf = mesh_conformity(m["vertices"], m["cells"])
        assert conf["conforming"] is True, (nth, nr, conf)
        assert conf["n_open_edges"] == 0 and conf["n_dup_faces"] == 0
        assert conf["n_hanging_edges"] == 0
        # 边界面 = 圆柱面 2nθ + 外边界 2nθ + 上下端面 2·(2·n_r·n_θ)
        assert conf["n_boundary"] == 2 * nth * (2 + 2 * nr)
    # θ=0 缝合面上的内部侧面必须被两个单元共享（修复前为 1）
    m = od.channel_tet_mesh(0.04, n_theta=24, n_r=4)
    V, C = m["vertices"], m["cells"]
    cx, cy = m["hole_center"]
    T, r0 = m["thickness"], m["hole_r"]
    faces = _cell_faces(C)
    # 缝合面上的棱柱侧面：三点都在 θ=0 射线平面（y=cy 且 x>cx），且跨越 z 层
    seam = [k for k, cs in faces.items() if len(cs) == 2
            and all(abs(V[v][1] - cy) < 1e-12 and V[v][0] > cx for v in k)
            and len({round(float(V[v][2]), 12) for v in k}) > 1]
    assert seam, "未找到 θ=0 缝合面上的内部侧面（用例失效）"
    assert max(np.hypot(V[k[1]][0] - cx, V[k[1]][1] - cy) for k in seam) > r0


def test_channel_ogrid_is_poorly_orthogonal_contrast_with_staircase():
    """S2/S4 贴体路线的**当前阻断点**（定量，本轮实测）：

    O 型网格虽已协调（拓扑合法、细网格不再奇异），但单元高度非正交/畸变 →
    压力-速度耦合的非正交修正不稳：稳态 60 步残差 0.009→0.386、|u|max 8.4e8；
    瞬态 40 步四种格式（upwind/limited/central，含无扰动）全部发散。
    对照阶梯网格：非正交角中位 0°、稳态残差 1.9e-6、|u|max 0.07 稳定。
    """
    from mesh_quality import orthogonality_report
    og = od.channel_tet_mesh(0.04, n_theta=96, n_r=14)
    r_og = orthogonality_report(og["vertices"], og["cells"])
    assert r_og["ok"] and r_og["verdict"] == "poor", r_og
    assert r_og["ortho_deg"]["median"] > 45.0
    assert r_og["ortho_deg"]["p95"] > 70.0
    assert r_og["skew"]["p95"] > 0.8
    st = od.channel_tet_mesh_cartesian(0.04, length_D=16.0, height_D=8.0,
                                       thickness_D=0.25, h_factor=4.0)
    r_st = orthogonality_report(st["vertices"], st["cells"])
    assert r_st["verdict"] == "good"
    assert r_st["ortho_deg"]["median"] == pytest.approx(0.0, abs=1e-4)


def test_channel_ogrid_solver_runs_on_fine_mesh():
    """修复前：细 O 型网格压力矩阵奇异（dgstrf info / NaN）无法求解。"""
    from pressure_solver import PressureSolver
    for nth, nr, steps in ((96, 14, 3), (192, 28, 2)):
        m = od.channel_tet_mesh(0.04, n_theta=nth, n_r=nr)
        s = PressureSolver(m["vertices"], m["cells"], rho=1.0, mu=1e-5,
                           inlet_axis=0, inlet_side="min",
                           inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
                           convection="central", wall_slip_axes=(1, 2),
                           piso_correctors=1)
        s.enable_transient(0.04, snapshot=True)
        for _ in range(steps):
            out = s.advance(dt=0.04, n_inner=1)
            assert np.isfinite(float(out["residual"])), (nth, nr, out)
        u = s.velocity()
        assert np.all(np.isfinite(u)) and float(np.abs(u).max()) > 0.0


def test_channel_tet_mesh_cartesian_staircase():
    m = od.channel_tet_mesh_cartesian(0.04, length_D=16.0, height_D=8.0,
                                      thickness_D=0.5, h_factor=4.0)
    assert m["ok"] and m["n_cells"] == 6 * m["n_hex"] and m["n_negative"] == 0
    assert abs(m["volume"] / m["volume_exact"] - 1.0) < 0.005
    assert m["h"] == pytest.approx(0.01) and m["blockage"] == pytest.approx(0.125)
    assert m["stair_deviation"] == pytest.approx(0.005)
    assert m["n_blocked"] > 0


# ---------------------------------------------------------------- 实跑结论（离线段）
def test_diff_metrics_flags_unresolved_shedding():
    # 本轮实跑结论：粗网格 + 一阶上风在 Re=200 数值耗散抑制涡脱 → Cl 振幅 0.0043（官方 0.2805）
    ours = {"st": None, "amplitude": 0.004257940242640512, "mean": 0.0010791142265866736}
    ref = {"strouhal": {"st": 0.1752, "amplitude": 0.2805}}
    d = od.diff_metrics(ours, ref)
    assert d["items"]["strouhal"]["ok"] is None          # 无自研 St → 不假装比对
    assert d["items"]["amplitude"]["ok"] is False         # 振幅比 0.015 → 带外
    assert d["items"]["amplitude"]["ratio"] < 0.05
    assert d["items"]["mean_lift"]["ok"] is True          # 无升力体平均升力≈0


def test_run_case_requires_long_flag():
    os.environ.pop("STARDECODING_LONG", None)
    out = od.run_case(D=0.04, u_inf=0.05, steps=4)
    assert out["ok"] is False and "STARDECODING_LONG" in out["reason"]
