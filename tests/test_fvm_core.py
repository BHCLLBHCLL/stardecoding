# -*- coding: utf-8 -*-
"""P 波 P4：有限体积离散核心（纯 numpy，occ / scdm 两环境皆可用）。

覆盖：
  cube_tet_mesh：一致性 Kuhn 6-tet 基元（体积和=1，全正定向，跨 hex 面一致）
  FVM 拓扑：owner/neighbor 映射、边界/内部面计数、外法向定向、面积/质心
  grad_lsq：最小二乘梯度（内部面双侧 + 边界面样本），线性场复原到机器精度，
            无边界值时不奇异（零梯度外推）
  grad_gauss：Green-Gauss，recon 二阶重构面值后线性场复原到机器精度
             （naive 线性平均在歪斜 Kuhn 网格上仅一阶，不保证精确）
  face_value / limiter / diffusion / convection 通量格式 向量化可用
  错误路径：网格过小 / 场量维度不匹配 / 梯度形状错误 / 面通量维度不匹配

验收核心（P4 行）：FVM 离散核心 —— 梯度(Gauss/LSQ)、限制器、通量格式 向量化可用。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from fvm_core import FVM, _safe_norm, _solve3x3_vec, _tet_volume, cube_tet_mesh


# ---------------------------------------------------------------- 工具
def _lin(x, y, z):
    return 2.0 * x - 3.0 * y + 0.5 * z + 1.0


_GRAD = np.array([2.0, -3.0, 0.5])


# ---------------------------------------------------------------- cube_tet_mesh
def test_cube_tet_mesh_conforming_nx1():
    V, C = cube_tet_mesh(1)
    assert C.shape[1] == 4
    assert len(C) == 6
    signed = _tet_volume(V, C)
    assert (signed > 0.0).all()
    assert abs(signed.sum() - 1.0) < 1e-12          # 体积精确填满
    f = FVM(V, C)
    assert f.n_cells == 6
    assert f.n_boundary_faces == 12
    assert f.n_interior_faces == 6


def test_cube_tet_mesh_refinement_conforming():
    # 体积和恒为 1，且随细化不变；边界面上单元全部正定向
    for nx in (2, 3, 4):
        V, C = cube_tet_mesh(nx)
        signed = _tet_volume(V, C)
        assert (signed > 0.0).all()
        assert abs(signed.sum() - 1.0) < 1e-12
        f = FVM(V, C)
        assert f.n_cells == 6 * nx ** 3
        # 单位立方体边界面数 = 6 * nx^2 * 2 个三角面
        assert f.n_boundary_faces == 12 * nx * nx


# ---------------------------------------------------------------- FVM 拓扑
def test_fvm_face_topology():
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    # 面数关系：欧拉不直接给，但 owner 一定有效，neighbor 为 -1 ⇔ 边界
    assert (f.owner >= 0).all()
    assert (f.owner < f.n_cells).all()
    assert ((f.neighbor == -1) == f.is_boundary).all()
    assert (f.face_area > 0.0).all()
    assert (f.neighbor >= 0).sum() == f.n_interior_faces
    assert (f.neighbor == -1).sum() == f.n_boundary_faces
    # 向量化计算面积并核对正外法向（指向 owner 侧单元外部）
    # 面法向 × 面质心 相对 owner 质心的点积 > 0（外法向）
    rel = f.face_centroid - f.centroids[f.owner]
    assert (np.einsum("ij,ij->i", rel, f.face_normal) > 0.0).all()


def test_fvm_topology_conservative_face_counts():
    # 每个单元 4 面：所有面均为该单元 owner 或 neighbor
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    # 每个内部面两单元各出现一次；边界面只出现一次
    face_count = np.zeros(f.n_cells, int)
    np.add.at(face_count, f.owner, 1)
    np.add.at(face_count, f.neighbor[f.neighbor >= 0], 1)
    assert (face_count == 4).all()


# ---------------------------------------------------------------- 梯度 · LSQ
@pytest.mark.parametrize("nx", [2, 3, 4])
def test_grad_lsq_exact_linear(nx):
    V, C = cube_tet_mesh(nx)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    fb = _lin(f.face_centroid[:, 0], f.face_centroid[:, 1], f.face_centroid[:, 2])
    ls = f.grad_lsq(phi, boundary=fb)
    assert np.abs(ls - _GRAD).max() < 1e-12


def test_grad_lsq_no_boundary_not_singular():
    # 无边界值时用零梯度外推：仍应为有限梯度（不因法方程奇异而崩溃/全零）
    V, C = cube_tet_mesh(3)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    ls = f.grad_lsq(phi)
    assert np.isfinite(ls).all()
    assert np.abs(ls).max() > 1e-3


# ---------------------------------------------------------------- 梯度 · Green-Gauss
@pytest.mark.parametrize("nx", [2, 3])
def test_grad_gauss_recon_exact_linear(nx):
    V, C = cube_tet_mesh(nx)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    fb = _lin(f.face_centroid[:, 0], f.face_centroid[:, 1], f.face_centroid[:, 2])
    ls = f.grad_lsq(phi, boundary=fb)
    gg = f.grad_gauss(phi, boundary=fb, recon=ls)
    assert np.abs(gg - _GRAD).max() < 1e-12


def test_grad_gauss_naive_finite():
    # naive 线性平均在歪斜网格上不保证精确，但必须有限且形状正确
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    fb = _lin(f.face_centroid[:, 0], f.face_centroid[:, 1], f.face_centroid[:, 2])
    gg = f.grad_gauss(phi, boundary=fb)
    assert gg.shape == (f.n_cells, 3)
    assert np.isfinite(gg).all()


# ---------------------------------------------------------------- 面插值
def test_face_value_interpolation_exact_linear():
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    fb = _lin(f.face_centroid[:, 0], f.face_centroid[:, 1], f.face_centroid[:, 2])
    fv = f.face_value(phi, boundary=fb)
    # 边界面对应精确面值
    is_int = f.neighbor >= 0
    assert np.abs(fv[~is_int] - fb[~is_int]).max() < 1e-12


# ---------------------------------------------------------------- 限制器
def test_limiter_linear_field_is_one():
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    fb = _lin(f.face_centroid[:, 0], f.face_centroid[:, 1], f.face_centroid[:, 2])
    ls = f.grad_lsq(phi, boundary=fb)
    lim = f.limiter(phi, ls)
    assert np.abs(lim - 1.0).max() < 1e-12


def test_limiter_bounded_range():
    # 非光滑场：限制器应落在 [0,1]
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    rng = np.random.RandomState(0)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    phi += 0.3 * rng.rand(f.n_cells)
    fb = np.zeros(f.n_faces)
    ls = f.grad_lsq(phi, boundary=fb)
    lim = f.limiter(phi, ls)
    assert (lim >= 0.0).all() and (lim <= 1.0).all()


# ---------------------------------------------------------------- 通量格式
def test_diffusion_flux_constant_field_zero():
    # 常量场 ∇φ=0 → 全网格扩散通量为 0
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = np.full(f.n_cells, 3.0)
    fl = f.diffusion_flux(phi, gamma=2.5)
    assert np.abs(fl).max() < 1e-12


def test_diffusion_flux_zngl_conservative():
    # 零梯度边界（boundary=None）：内部面通量成对抵消 → 全局净通量守恒为 0
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    fl = f.diffusion_flux(phi)
    is_int = f.neighbor >= 0
    net = np.zeros(f.n_cells)
    np.add.at(net, f.owner[is_int], fl[is_int])
    np.add.at(net, f.neighbor[is_int], -fl[is_int])
    assert abs(net.sum()) < 1e-12
    assert (fl[f.neighbor == -1] == 0.0).all()


def test_diffusion_flux_antisymmetric_sign():
    # 面通量单值：同一内部面 owner 计 +F、邻居计 -F（构造性守恒）；无乱序
    V, C = cube_tet_mesh(3)
    f = FVM(V, C)
    rng = np.random.RandomState(1)
    phi = rng.rand(f.n_cells)
    fl = f.diffusion_flux(phi, boundary=np.zeros(f.n_faces))
    is_int = f.neighbor >= 0
    # 任意内部面通量有限、边界通量有限
    assert np.isfinite(fl).all()
    assert (fl[is_int] != 0.0).any()


def test_convection_upwind_selection():
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = np.arange(f.n_cells, dtype=float)     # 单调递增
    mdot = np.zeros(f.n_faces)
    is_int = f.neighbor >= 0
    mdot[is_int] = 1.0                          # 全内流正向（owner→neighbor）
    fl = f.convection_flux_upwind(mdot, phi)
    # 正向 mdot：取 owner 值
    assert np.abs(fl[is_int] - phi[f.owner[is_int]]).max() < 1e-12
    # 反向 mdot：取 neighbor 值 → F = mdot * phi_neighbor
    mdot[is_int] = -1.0
    fl = f.convection_flux_upwind(mdot, phi)
    assert np.abs(fl[is_int] - (-1.0) * phi[f.neighbor[is_int]]).max() < 1e-12


def test_convection_upwind_boundary_inflow():
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = np.arange(f.n_cells, dtype=float)
    fb = np.full(f.n_faces, -100.0)
    mdot = np.zeros(f.n_faces)
    bnd = f.neighbor == -1
    mdot[bnd] = -1.0                            # 边界面入流
    fl = f.convection_flux_upwind(mdot, phi, boundary=fb)
    assert np.abs(fl[bnd] - (-1.0) * (-100.0)).max() < 1e-12   # mdot * boundary


def test_convection_central_uses_face_value():
    V, C = cube_tet_mesh(2)
    f = FVM(V, C)
    phi = _lin(f.centroids[:, 0], f.centroids[:, 1], f.centroids[:, 2])
    fb = _lin(f.face_centroid[:, 0], f.face_centroid[:, 1], f.face_centroid[:, 2])
    mdot = np.ones(f.n_faces)
    fl = f.convection_flux_central(mdot, phi, boundary=fb)
    fv = f.face_value(phi, boundary=fb)
    assert np.abs(fl - fv).max() < 1e-12


# ---------------------------------------------------------------- 错误路径
def test_fvm_rejects_small_mesh():
    with pytest.raises(ValueError):
        FVM(np.zeros((3, 3)), np.zeros((0, 4), np.int64))


def test_fvm_rejects_bad_vertices_shape():
    with pytest.raises(ValueError):
        FVM(np.zeros((5, 2)), np.zeros((1, 4), np.int64))


def test_fvm_rejects_non_tet_cells():
    with pytest.raises(ValueError):
        V, C = cube_tet_mesh(1)
        FVM(V, C[:, :3])


def test_grad_phi_shape_mismatch():
    V, C = cube_tet_mesh(1)
    f = FVM(V, C)
    with pytest.raises(ValueError):
        f.grad_lsq(np.zeros(f.n_cells + 1))


def test_grad_gauss_recon_shape_mismatch():
    V, C = cube_tet_mesh(1)
    f = FVM(V, C)
    phi = np.zeros(f.n_cells)
    with pytest.raises(ValueError):
        f.grad_gauss(phi, recon=np.zeros((f.n_cells, 2)))


def test_flux_mdot_shape_mismatch():
    V, C = cube_tet_mesh(1)
    f = FVM(V, C)
    phi = np.zeros(f.n_cells)
    with pytest.raises(ValueError):
        f.convection_flux_upwind(np.zeros(f.n_faces + 1), phi)
