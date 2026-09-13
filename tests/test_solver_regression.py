# -*- coding: utf-8 -*-
"""R1-4 教程工况数值验收（parity_100pct_plan.md R 波）。

承接 R1-1~R1-3 补齐的瞬态内核 / 非凸网格 / 力系数三项能力，做**数值验证**与
**教程工况对标**：

  瞬态（r1-4a）
    1) 逐步全局质量守恒：每步 sum(mdot[inlet]) + sum(mdot[outlet]) == 0（机器零）
    2) 内迭代收敛性：单步内迭代越多 → 残差越小
    3) 从静止起动：u_mean 单调抬升并趋近稳态解（流场发展）
    4) 物理时间推进历史：solve_transient 残差随步衰减、全程有限

  教程工况对标（r1-4b）
    5) 直管压降 ∝ 黏度 μ（Stokes/层流线性律），比值 2.0
    6) 直管压降 ∝ 入口速度 U，比值 2.0
    7) 直管壁面黏性阻力 ∝ μ，且横流向合力 ≪ 流向（对称性）
    8) 直管壁面黏性阻力 ∝ U，且 Cl ≪ Cd
    9) 突缩台阶压力阻力 Fx>0、Cd>0、压力项主导黏性项（钝体阻力对标）
   10) 力系数参考量归一化：Cd ∝ 1/A_ref（动压除子正确）

  长耗时（r1-4c，默认 skip）
   11) 涡脱 Strouhal 提取：需 STARDECODING_LONG=1 开启。当前网格分辨率下尾迹
       稳态（无涡脱），用例断言"稳态"或"检出涡脱且 St∈(0,2)"二者之一，两者
       皆为真实结论，不伪造数值。

几何说明：本仓库无专用翼型/圆柱几何生成器，教程工况由现有原语构造——
  直管 = `fvm_core.cube_tet_mesh` 单元沿流向仿射拉伸（结构化，网格质量高）；
  突缩台阶 = 星形多边形 z 向拉伸水密面 + `mesh_tet.tet_mesh`（非凸域裁剪）。
后两者依赖 scipy Delaunay，故单独以 `require_scipy` 门控（与 test_mesh_tet.py 同约定）。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from fvm_core import cube_tet_mesh
from mesh_tet import tet_mesh
from pressure_solver import PressureSolver


def _has(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


_LONG = os.environ.get("STARDECODING_LONG", "") not in ("", "0", "false", "False")
require_scipy = pytest.mark.skipif(not _has("scipy"),
                                   reason="需要 scipy（Delaunay 非凸体网格）")
require_long = pytest.mark.skipif(not _LONG,
                                  reason="长耗时用例：设 STARDECODING_LONG=1 开启")


# ---------------------------------------------------------------- 构造工具
def _duct(length=3.0, nx=3, **kw):
    """直管：单位立方体结构化四面体网格沿 x 拉伸到 length（保形，质量不降）。"""
    V, C = cube_tet_mesh(nx)
    V = V.copy()
    V[:, 0] *= float(length)
    kw.setdefault("rho", 1.0)
    kw.setdefault("mu", 1e-3)
    kw.setdefault("inlet_axis", 0)
    kw.setdefault("inlet_side", "min")
    kw.setdefault("inlet_velocity", (1.0, 0.0, 0.0))
    kw.setdefault("outlet_side", "max")
    kw.setdefault("alpha_momentum", 0.7)
    kw.setdefault("alpha_pressure", 0.3)
    return PressureSolver(V, C, **kw)


def _restart(nx=2, **kw):
    """小方腔瞬态算例（层流，低 Re，收敛稳健）。"""
    kw.setdefault("rho", 1.0)
    kw.setdefault("mu", 0.05)
    kw.setdefault("inlet_velocity", (1.0, 0.0, 0.0))
    kw.setdefault("alpha_momentum", 0.7)
    kw.setdefault("alpha_pressure", 0.3)
    return PressureSolver(*cube_tet_mesh(nx), **kw)


def _run(s, n):
    last = None
    for _ in range(n):
        last = s.step()
    return last


def _pressure_drop(s, frac=0.1):
    """进出口近端单元平均压差 p_in - p_out（长度方向 10% 端带）。"""
    p = s.pressure()
    xc = s._fv.centroids[:, 0]
    lo, hi = float(xc.min()), float(xc.max())
    span = max(hi - lo, 1e-30)
    return float(p[xc < lo + frac * span].mean()
                 - p[xc > hi - frac * span].mean())


def _prism(poly, z0=0.0, z1=1.0):
    """星形多边形 z 向拉伸 → 水密棱柱表面（顶点 0 对多边形完全可见）。

    poly 须相对 vertex 0 星形（本文件用的方管/突缩 L 形均满足），
    顶/底盖用扇形三角化即可无缝无缝覆盖。
    """
    poly = np.asarray(poly, float)
    n = len(poly)
    V = np.vstack([np.column_stack([poly, np.full(n, z0)]),
                   np.column_stack([poly, np.full(n, z1)])])
    F = []
    for i in range(1, n - 1):
        F.append([0, i, i + 1])
    for i in range(1, n - 1):
        F.append([n, n + i + 1, n + i])
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, n + j])
        F.append([i, n + j, n + i])
    return V, np.array(F, np.int64)


_L_POLY = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0),
           (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]


def _contraction_solver(spacing=0.3, mu=5e-2, **kw):
    """突缩台阶（2:1 收缩，台阶面法向 +x）流体域 + 高黏度稳健求解器。

    欠松弛取标准 α_m=0.7 / α_p=0.3：实测该组合对稳态与瞬态推进均稳定（残差 1e-14）；
    更激进的 α_m=0.3 / α_p=0.2 虽能收敛稳态，但在瞬态推进（小 Δt）下欠阻尼会缓慢
    发散（u_max 从 2.4 涨到 10+），故此处不使用。
    """
    V, F = _prism(_L_POLY, 0.0, 1.0)
    out = tet_mesh(V, F, spacing=spacing, method="scipy")
    assert out["ok"] and out["n_cells"] > 0
    kw.setdefault("rho", 1.0)
    kw.setdefault("mu", mu)
    kw.setdefault("inlet_axis", 0)
    kw.setdefault("inlet_side", "min")
    kw.setdefault("inlet_velocity", (1.0, 0.0, 0.0))
    kw.setdefault("outlet_side", "max")
    kw.setdefault("alpha_momentum", 0.7)
    kw.setdefault("alpha_pressure", 0.3)
    return PressureSolver(out["vertices"], out["cells"], **kw)


def _step_faces(s):
    """台阶迎流面：壁面中法向近 ±x、位于 x≈1 且 y>0.5 的面。"""
    fc = s._fv.face_centroid
    fn = s._fv.face_normal
    wf = s._wall_faces
    return wf[(np.abs(fn[wf, 0]) > 0.5)
              & (fc[wf, 0] > 0.5) & (fc[wf, 0] < 1.5)
              & (fc[wf, 1] > 0.5)]


# ================================================================ r1-4a 瞬态
def test_regression_transient_inlet_outlet_mass_balance():
    """逐步全局质量守恒：每步进口净通量 + 出口净通量 == 0（机器零）。"""
    s = _restart(nx=2)
    s.enable_transient(0.1)
    worst = 0.0
    for _ in range(40):
        s.advance(n_inner=1)
        md = s.mass_flux()
        worst = max(worst, abs(float(md[s._inlet_faces].sum()
                                    + md[s._outlet_faces].sum())))
    assert worst < 1e-10, "逐步质量不守恒：%.3e" % worst
    assert s.continuity_ratio() < 1e-4


def test_regression_transient_inner_iterations_reduce_residual():
    """单步内迭代收敛性：内迭代越多，步末残差越小（SIMPLE 内层收敛）。"""
    res = {}
    for ni in (1, 8):
        t = _restart(nx=2)
        t.enable_transient(0.1)
        for _ in range(20):
            t.advance(n_inner=1)
        res[ni] = float(t.advance(n_inner=ni)["residual"])
    assert np.isfinite(res[1]) and np.isfinite(res[8])
    assert res[8] < res[1], "内迭代未降残差：n=1 %.3e vs n=8 %.3e" % (res[1], res[8])


def test_regression_transient_flow_develops_toward_steady():
    """从静止起动：u_mean 抬升并趋近稳态解（瞬态流场发展）。"""
    steady = _restart(nx=2)
    _run(steady, 200)
    u_steady = float(steady.velocity()[:, 0].mean())
    assert u_steady > 0.5

    s = _restart(nx=2)
    s._u[:] = 0.0
    s._v[:] = 0.0
    s._w[:] = 0.0
    s._rebuild_mdot()
    s.enable_transient(0.05)
    hist = np.array([s.advance(n_inner=1)["u_mean"] for _ in range(200)])
    assert np.isfinite(hist).all()
    assert hist[0] < hist[9] < hist[-1], "静止起动应单调抬升"
    assert abs(hist[-1] - u_steady) / u_steady < 0.03, \
        "瞬态终值应趋近稳态：%.5f vs %.5f" % (hist[-1], u_steady)


def test_regression_transient_time_march_residual_decays():
    """物理时间推进：solve_transient 全程有限且残差随步衰减。"""
    s = _restart(nx=2)
    hist = s.solve_transient(t_end=1.0, dt=0.1, n_inner=2)
    assert len(hist) == 10
    assert s.time == pytest.approx(1.0)
    res = np.array([p[1] for p in hist])
    assert np.isfinite(res).all(), "瞬态残差历史出现非有限值"
    assert res[-1] < res[0], "残差未随步衰减：%.3e -> %.3e" % (res[0], res[-1])


# ================================================================ r1-4b 工况对标
def test_regression_duct_pressure_drop_scales_with_viscosity():
    """直管压降 ∝ 黏度（层流线性律）：μ 加倍 → Δp 加倍。"""
    a = _duct(mu=1e-3)
    _run(a, 300)
    b = _duct(mu=2e-3)
    _run(b, 300)
    dp1, dp2 = _pressure_drop(a), _pressure_drop(b)
    assert dp1 > 0.0 and dp2 > 0.0
    ratio = dp2 / dp1
    assert 1.9 < ratio < 2.1, "Δp 未随 μ 线性：ratio=%.4f (dp1=%.6f dp2=%.6f)" \
        % (ratio, dp1, dp2)


def test_regression_duct_pressure_drop_scales_with_velocity():
    """直管压降 ∝ 入口速度：U 加倍 → Δp 加倍。"""
    a = _duct(inlet_velocity=(1.0, 0.0, 0.0))
    _run(a, 300)
    b = _duct(inlet_velocity=(2.0, 0.0, 0.0))
    _run(b, 300)
    dp1, dp2 = _pressure_drop(a), _pressure_drop(b)
    assert dp1 > 0.0 and dp2 > 0.0
    ratio = dp2 / dp1
    assert 1.9 < ratio < 2.1, "Δp 未随 U 线性：ratio=%.4f (dp1=%.6f dp2=%.6f)" \
        % (ratio, dp1, dp2)


def test_regression_duct_wall_drag_scales_with_viscosity():
    """直管壁面黏性阻力 ∝ μ；横流向合力 ≪ 流向（对称性）。"""
    a = _duct(mu=5e-3)
    _run(a, 300)
    b = _duct(mu=1e-2)
    _run(b, 300)
    fa = a.forces(a_ref=1.0, u_ref=1.0)["force"]
    fb = b.forces(a_ref=1.0, u_ref=1.0)["force"]
    assert fa[0] > 0.0, "流向阻力应为正（流体拖曳壁面）"
    ratio = float(fb[0] / fa[0])
    assert 1.9 < ratio < 2.1, "壁面阻力未随 μ 线性：ratio=%.4f" % ratio
    side = max(abs(float(fa[1])), abs(float(fa[2])))
    assert side < 0.02 * float(fa[0]), "横流向合力应远小于流向：%.2e vs %.2e" \
        % (side, fa[0])


def test_regression_duct_wall_drag_scales_with_velocity():
    """直管壁面黏性阻力 ∝ U；Cl ≪ Cd（对称管无升力）。"""
    a = _duct(inlet_velocity=(1.0, 0.0, 0.0), mu=5e-3)
    _run(a, 300)
    b = _duct(inlet_velocity=(2.0, 0.0, 0.0), mu=5e-3)
    _run(b, 300)
    ra = a.forces(a_ref=1.0, u_ref=1.0)
    rb = b.forces(a_ref=1.0, u_ref=1.0)
    ratio = float(rb["force"][0] / ra["force"][0])
    assert 1.9 < ratio < 2.15, "壁面阻力未随 U 线性：ratio=%.4f" % ratio
    assert abs(ra["cl"]) < 0.05 * abs(ra["cd"]), "对称管应近零升力"


def test_regression_force_coefficient_reference_normalization():
    """力系数归一化：Cd = F·d̂ / (½ρU²A_ref)，A_ref 加倍 → Cd 减半。"""
    s = _duct(mu=5e-3)
    _run(s, 300)
    r1 = s.forces(a_ref=1.0, u_ref=1.0)
    r2 = s.forces(a_ref=2.0, u_ref=1.0)
    assert r1["cd"] > 0.0
    assert r2["cd"] == pytest.approx(0.5 * r1["cd"], rel=1e-9), \
        "Cd∝1/A_ref：%.6f vs %.6f" % (r1["cd"], r2["cd"])
    assert r1["q"] == pytest.approx(0.5 * 1.0 * 1.0 * 1.0)
    assert np.array_equal(r1["force"], r2["force"])


@require_scipy
def test_regression_contraction_step_pressure_drag_positive():
    """突缩台阶（钝体）阻力：迎流面 Fx>0、Cd>0、压力项主导（对流 Re 下）。"""
    s = _contraction_solver(spacing=0.3, mu=5e-2)
    last = _run(s, 600)
    assert np.isfinite(last["residual"]) and last["residual"] < 1e-6

    faces = _step_faces(s)
    assert faces.size > 0, "未识别到台阶迎流面"
    res = s.forces(a_ref=1.0, u_ref=1.0, faces=faces)
    assert np.isfinite(res["force"]).all()
    assert res["force"][0] > 0.0, "台阶流向阻力应为正：%.6f" % res["force"][0]
    assert res["cd"] > 0.0
    # 压力阻力主导：迎流法向面上 |F_p| 远大于黏性项
    assert abs(res["pressure"][0]) > 5.0 * abs(res["viscous"][0]), \
        "压力阻力应主导：p=%.4f v=%.4f" % (res["pressure"][0], res["viscous"][0])
    # 阻力方向取入口速度方向 → 合力在阻力方向投影即 Cd 分子
    assert res["drag_dir"][0] == pytest.approx(1.0)
    assert res["force"][0] / res["q"] == pytest.approx(res["cd"])


@require_scipy
def test_regression_contraction_total_wall_force_physical():
    """突缩台阶全壁面合力有限，且台阶面贡献流向阻力（不含入口/出口面）。"""
    s = _contraction_solver(spacing=0.3, mu=5e-2)
    _run(s, 600)
    res = s.forces(a_ref=1.0, u_ref=1.0)
    assert np.isfinite(res["force"]).all()
    assert res["n_faces"] == int(s._wall_faces.size)
    assert res["force"][0] > 0.0


# ================================================================ r1-4c 长耗时
@require_scipy
@require_long
def test_regression_long_vortex_shedding_strouhal():
    """涡脱 Strouhal 提取（长耗时，默认 skip，STARDECODING_LONG=1 开启）。

    以突缩台阶尾迹升力时程估计 St = f·D/U（D 取台阶高度）。当前网格分辨率与
    黏度量级下尾迹为稳态（无涡脱）——此时断言"稳态"（脉动远小于均值）；若检出
    显著脉动则断言 St 有限且落在物理带 (0, 2)。两条分支皆为真实结论。

    稳健性：该粗网格 + SIMPLE 对低黏度（Re 高）发散（实测 μ≤2e-2 稳态即出 nan）；高黏度
    μ=5e-2 配上标准 α_m=0.7/α_p=0.3 后稳态残差 1e-14、瞬态推进全程有限（低 Re 稳态尾迹）。
    """
    s = _contraction_solver(spacing=0.3, mu=5e-2)
    _run(s, 800)
    assert s.residual() < 1e-6
    faces = _step_faces(s)
    assert faces.size > 0

    dt = 0.05
    n_steps = 600
    s.enable_transient(dt=dt)
    cl_hist = np.empty(n_steps, float)
    for k in range(n_steps):
        s.advance(n_inner=3)
        cl_hist[k] = s.forces(a_ref=1.0, u_ref=1.0, faces=faces)["cl"]
    assert np.isfinite(cl_hist).all(), "升力时程出现非有限值"

    mean = float(cl_hist.mean())
    amp_rel = float(cl_hist.std()) / max(abs(mean), 1e-12)
    if amp_rel < 0.05:
        assert np.isfinite(mean)
    else:
        # 仅在下述"显著脉动"分支检索谱峰，并排除奈奎斯特混叠带（网格尺度数值噪声），
        # 以免把噪声误判为物理涡脱。
        spec = np.abs(np.fft.rfft(cl_hist - mean))
        freq = np.fft.rfftfreq(n_steps, d=dt)
        band = (freq > 0.0) & (freq < 0.9 * freq[-1])
        f_peak = float(freq[int(np.argmax(np.where(band, spec, 0.0)))])
        st = f_peak * 1.0 / 1.0
        assert np.isfinite(st) and 0.0 < st < 2.0, "Strouhal 越界：%.4f" % st
