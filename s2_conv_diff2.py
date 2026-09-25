# -*- coding: utf-8 -*-
"""S2 对流数值扩散的定量测量（**仅内部面的二次型**，无边界污染）。

对流算子的模态衰减率（Rayleigh 商）可直接写成面求和：
  φᵀCφ = Σ_内部面 m_f · φ_U · (φ_U − φ_D)        （上风）
  φᵀCφ = Σ_内部面 0.5·m_f · (φ_o² − φ_nb²)      （中心，= 0.5m(φ_o−φ_nb)(φ_o+φ_nb)）
分母 Σ V φ²。模态 φ=sin(kx x)·sin(ky y)，衰减率即数值扩散 ν_num·(kx²+ky²)。
只用内部面 ⇒ 完全不含入口/壁面/出口边界行，结果可解释。
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver

U, RHO, NU = 0.05, 1.0, 1e-5
for NX, NY in ((16, 16), (32, 32)):
    V, C = cube_tet_mesh(nx=NX, ny=NY, nz=1)
    s = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                       inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                       convection='upwind', wall_slip_axes=(1, 2))
    fv = s.fvm
    nn = np.asarray(fv.face_normal, float)
    A = np.asarray(fv.face_area, float)
    m_all = RHO * U * nn[:, 0] * A            # 均匀 x 向流
    cen = np.asarray(fv.centroids, float)
    vol = np.asarray(fv.volumes, float)
    Lx = float(cen[:, 0].max() - cen[:, 0].min())
    Ly = float(cen[:, 1].max() - cen[:, 1].min())
    kx, ky = 2.0 * np.pi / Lx, np.pi / Ly
    phi = (np.sin(kx * (cen[:, 0] - cen[:, 0].min()))
           * np.sin(ky * (cen[:, 1] - cen[:, 1].min())))
    den = float(phi @ (vol * phi))
    is_int = fv.neighbor >= 0
    fi = np.where(is_int)[0]
    o, nb, m = fv.owner[fi], fv.neighbor[fi], m_all[fi]
    # 上风：U 侧取 m>0 ? o : nb
    pos = m >= 0.0
    phi_U = np.where(pos, phi[o], phi[nb])
    phi_D = np.where(pos, phi[nb], phi[o])
    lam_up = float(np.sum(m * phi_U * (phi_U - phi_D))) / den
    lam_cen = float(np.sum(0.5 * m * (phi[o] ** 2 - phi[nb] ** 2))) / den
    hx = Lx / NX
    nu_up = -lam_up / (kx ** 2 + ky ** 2)
    nu_cen = -lam_cen / (kx ** 2 + ky ** 2)
    print('网格 %dx%d（hx=%.4f）:' % (NX, NY, hx))
    print('  上风   衰减率 %+.5f /s → ν_num = %+.3e（= %.1f× 分子 ν）'
          % (-lam_up, nu_up, nu_up / NU))
    print('  中心   衰减率 %+.5f /s → ν_num = %+.3e（= %.1f× 分子 ν）'
          % (-lam_cen, nu_cen, nu_cen / NU))
    print('  理论一阶上风 ν_num ≈ U·h/2 = %.3e（＝ %.1f× 分子 ν）'
          % (U * hx / 2.0, U * hx / 2.0 / NU))
    print('  ⇒ 延迟修正可挽回额度 = %.3e（占上风扩散 %.0f%%），但需内迭代收敛'
          % (nu_up - nu_cen, 100.0 * (nu_up - nu_cen) / max(abs(nu_up), 1e-300)))