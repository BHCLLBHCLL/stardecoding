# -*- coding: utf-8 -*-
"""S 波 S4：线性求解路径（直接解路由 / ILU 修复 / 残差检查向量化）。

背景：旧 `_solve_scipy` 把 spilu 返回对象直接当 M= 传给 bicgstab，新版 SciPy 抛 TypeError
被静默吞掉 → ILU 快路径从未生效，且每次白做一次因式分解（24k 单元实测白花 1.2 s/步）。
本轮修好并改为**按规模选路**（默认 ≤15 万未知量直接稀疏 LU，更大规模先 AMG）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from pressure_solver import _check_residual, _solve_scipy, solve_linear  # noqa: E402


def _spd_system(n=400, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.01, random_state=rng, format="csr")
    A = (A + A.T) + sp.diags(np.full(n, 10.0))       # 对角占优 → SPD
    b = rng.random(n)
    return A, b


def _coo(A):
    A = A.tocoo()
    return A.row.astype(np.int64), A.col.astype(np.int64), A.data.astype(float)


def test_direct_route_matches_spsolve():
    A, b = _spd_system()
    r, c, d = _coo(A)
    x = _solve_scipy(r, c, d, b, A.shape[0], 1e-9, 8000, None)
    ref = spla.spsolve(A.tocsc(), b)
    assert np.allclose(x, ref, rtol=0, atol=1e-10)


def test_route_switch_env_forces_iterative_path(monkeypatch):
    """STARDECODING_DIRECT_MAX=0 → 走迭代路径，仍须给出准确解。"""
    monkeypatch.setenv("STARDECODING_DIRECT_MAX", "0")
    A, b = _spd_system(seed=1)
    r, c, d = _coo(A)
    x = _solve_scipy(r, c, d, b, A.shape[0], 1e-9, 8000, None)
    ref = spla.spsolve(A.tocsc(), b)
    rel = float(np.linalg.norm(x - ref) / np.linalg.norm(ref))
    assert rel < 1e-5, "迭代路径相对误差应小（实测 %.2e）" % rel


def test_solve_linear_modes_agree():
    A, b = _spd_system(seed=2)
    r, c, d = _coo(A)
    x_auto = solve_linear(r, c, d, b, A.shape[0])
    x_np = solve_linear(r, c, d, b, A.shape[0], kind="numpy", maxit=20000)
    ref = spla.spsolve(A.tocsc(), b)
    assert np.allclose(x_auto, ref, atol=1e-9)
    assert float(np.linalg.norm(x_np - ref) / np.linalg.norm(ref)) < 1e-3


def test_residual_check_vectorized_is_correct():
    A, b = _spd_system(seed=3)
    r, c, d = _coo(A)
    x = spla.spsolve(A.tocsc(), b)
    assert _check_residual(r, c, d, x, b, 1e-8) is True
    assert _check_residual(r, c, d, x * 0.0, b, 1e-8) is False


def test_solve_linear_rejects_bad_rhs_shape():
    A, b = _spd_system(seed=4)
    r, c, d = _coo(A)
    with pytest.raises(ValueError):
        solve_linear(r, c, d, b[:-1], A.shape[0])
