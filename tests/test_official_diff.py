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
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import official_diff as od  # noqa: E402

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


def test_run_case_requires_long_flag():
    os.environ.pop("STARDECODING_LONG", None)
    out = od.run_case(D=0.04, u_inf=0.05, steps=4)
    assert out["ok"] is False and "STARDECODING_LONG" in out["reason"]
