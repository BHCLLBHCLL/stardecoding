# -*- coding: utf-8 -*-
"""S4 第四轮：线性求解器**按系统类型分流**的正确性与路线验证。

背景（本机实测，97,680 未知量 / 454k nnz）：
  · 动量（对流扩散，非对称）：直接 LU 5.56s / AMG-SA 3.47s / **ILU(1e-4)+BiCGSTAB 0.93s**；
  · 压力（泊松，对称正定）：直接 LU 6.54s / **AMG-SA 0.90s** / ILU+BiCGSTAB 失效（info=-10）；
  · 阈值也一并修正：旧默认 DIRECT_MAX=150000 是只在 24k 上标定的，10 万级仍走直接 LU（5s/次）；
    实测交叉点在 5 万附近，故默认降到 50000。

本文件验证：① 两条迭代路线各自能解对应类型的系统（残差达标）；② solve_linear 的
system 分流在强制迭代时与直接 LU 解一致；③ 兜底：迭代失败时必须回退而不是返回未收敛解。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import pressure_solver as ps  # noqa: E402


def _poisson_system(nx=40, ny=40):
    """二维五点 Laplace（对称正定）—— 压力修正方程的原型。"""
    n = nx * ny
    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            rows.append(k); cols.append(k); vals.append(4.0)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    rows.append(k); cols.append(jj * nx + ii); vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    rng = np.random.default_rng(7)
    x_true = rng.standard_normal(n)
    return A, A @ x_true, x_true


def _convection_system(nx=40, ny=40, u=0.6, v=0.2, gamma=0.05):
    """一阶上风对流 + 中心扩散（非对称、对角占优）—— 动量方程的原型。"""
    n = nx * ny
    h = 1.0 / nx
    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            diag = 4.0 * gamma / h
            off = []
            for di, dj, flux in ((1, 0, u), (-1, 0, -u), (0, 1, v), (0, -1, -v)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < nx and 0 <= jj < ny):
                    diag += max(flux, 0.0)          # 出流边界：上风项计入对角
                    continue
                kk = jj * nx + ii
                off.append(kk)
                rows.append(k); cols.append(kk); vals.append(-gamma / h - min(flux, 0.0))
                diag += max(flux, 0.0)
            rows.append(k); cols.append(k); vals.append(diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    rng = np.random.default_rng(11)
    x_true = rng.standard_normal(n)
    return A, A @ x_true, x_true


def _coo(A):
    A = A.tocoo()
    return A.row, A.col, A.data


def _rel_res(A, x, b):
    return float(np.linalg.norm(b - A @ np.asarray(x, float).ravel())\
                 / max(float(np.linalg.norm(b)), 1e-300))


def test_ilu_bicgstab_solves_convection_system():
    A, b, _x = _convection_system()
    r, c, d = _coo(A)
    x = ps._ilu_bicgstab(A, r, c, d, b, 1e-8, 4000)
    assert x is not None, "ILU+BiCGSTAB 未在对流扩散系统上收敛"
    assert _rel_res(A, x, b) < 1e-5


def test_amg_solves_poisson_system():
    A, b, _x = _poisson_system()
    r, c, d = _coo(A)
    x = ps._amg_solve(A, r, c, d, b, None, 1e-8, 4000)
    assert x is not None, "AMG 未在泊松系统上收敛"
    assert _rel_res(A, x, b) < 1e-5


def test_solve_linear_system_routing_matches_direct(monkeypatch):
    """强制走迭代路线（阈值=0）时，解必须与直接 LU 一致（同一物理答案）。"""
    monkeypatch.setenv("STARDECODING_DIRECT_MAX", "1000000")
    for maker, system in ((_poisson_system, "poisson"),
                          (_convection_system, "convection")):
        A, b, _x = maker()
        r, c, d = _coo(A)
        n = A.shape[0]
        x_direct = ps.solve_linear(r, c, d, b, n, tol=1e-10, system=system)
        monkeypatch.setenv("STARDECODING_DIRECT_MAX", "0")
        x_iter = ps.solve_linear(r, c, d, b, n, tol=1e-8, system=system)
        assert _rel_res(A, x_iter, b) < 1e-5, (system, _rel_res(A, x_iter, b))
        assert _rel_res(A, x_direct, b) < 1e-6, system
        # 迭代解按 1e-6 相对残差收敛 → 点位差应远小于解本身的量级
        scale = max(float(np.abs(x_direct).max()), 1.0)
        assert np.allclose(x_direct, x_iter, rtol=1e-3, atol=1e-4 * scale), \
            (system, float(np.abs(x_direct - x_iter).max()) / scale)


def test_solve_linear_rejects_unconverged(monkeypatch):
    """迭代几乎不可能收敛（maxit=1）时必须逐级回退，仍返回可用解（不得返回未收敛解）。"""
    monkeypatch.setenv("STARDECODING_DIRECT_MAX", "0")
    A, b, _x = _convection_system(nx=24, ny=24)
    r, c, d = _coo(A)
    n = A.shape[0]
    x = ps.solve_linear(r, c, d, b, n, tol=1e-9, maxit=1, system="convection")
    assert np.isfinite(x).all()
    assert _rel_res(A, x, b) < 1e-6, _rel_res(A, x, b)   # 回退到直接 LU 后必须精确


def test_ilu_defaults_are_zero_fill():
    """回归护栏：ILU 默认必须是零填充 ILU(0)（高填充把成本全花在因式分解上，实测 2.3× 慢）。"""
    import inspect
    sig = inspect.signature(ps._ilu_bicgstab)
    assert sig.parameters["drop_tol"].default == 0.0
    assert sig.parameters["fill_factor"].default == 1.0


def test_amg_uses_cg_with_amg_preconditioner():
    """回归护栏：压力泊松走 AMG 预条件 CG（比 ml.solve 迭代精化快近 2×）。"""
    src = open(os.path.join(ROOT, "pressure_solver.py"), encoding="utf-8").read()
    assert "aspreconditioner()" in src
    assert "max_coarse=200" in src


def test_direct_threshold_default_now_below_100k():
    """回归护栏：默认直接 LU 阈值必须低于 10 万（否则 10 万级又回到 5s/次的直接 LU）。"""
    src = open(os.path.join(ROOT, "pressure_solver.py"), encoding="utf-8").read()
    assert 'STARDECODING_DIRECT_MAX", "50000"' in src