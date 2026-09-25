# -*- coding: utf-8 -*-
"""S2 诊断：三种对流格式的动量矩阵**结构**对比（解释 upwind2 在封闭盒输运模态下的发散）。

判据（离散对流-扩散矩阵的稳定性）：
  · 非对角元必须 ≤0（M 矩阵性质；正的非对角元 = 反扩散，可直接致发散）；
  · 对称部分 (A+Aᵀ)/2 的最大特征值必须 <0（否则该格式在能量意义上不耗散）；
  · 对角元必须 >0 且行和对角占优（在封闭域 + 零梯度下的近似判据）。
用法：python s2_sou_diag.py [nx]
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver

NX = int(sys.argv[1]) if len(sys.argv) > 1 else 16
U, A, NU, RHO, DT = 0.05, 0.002, 1e-5, 1.0, 0.01

for CONV in ('upwind', 'central', 'upwind2'):
    V, C = cube_tet_mesh(nx=NX, ny=NX, nz=1)
    s = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                       inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                       inlet_zero_gradient=True, wall_slip_axes=(2,),
                       convection=CONV, alpha_momentum=1.0, piso_correctors=0)
    cen = np.asarray(s.fvm.centroids, float)
    Lx = float(cen[:, 0].max() - cen[:, 0].min())
    Ly = float(cen[:, 1].max() - cen[:, 1].min())
    x0, y0 = float(cen[:, 0].min()), float(cen[:, 1].min())
    k = 2.0 * np.pi * 4 / Lx
    xx, yy = cen[:, 0] - x0, cen[:, 1] - y0
    s._u = U + A * np.pi * np.sin(2.0 * np.pi * yy) * np.sin(k * xx)
    s._v = -A * k * np.cos(k * xx) * np.sin(np.pi * yy) ** 2
    s._w = np.zeros_like(s._u); s._p = np.zeros_like(s._u)
    s.enable_transient(DT, snapshot=True)
    r, c, v, rhs, ap = s._assemble_momentum(0)
    n = C.shape[0]
    import scipy.sparse as sp
    Am = sp.csr_matrix((v, (r, c)), shape=(n, n))
    off = Am - sp.diags(Am.diagonal())
    pos = int((off.data > 1e-14).sum()) if off.nnz else 0
    dmin, dmax = float(Am.diagonal().min()), float(Am.diagonal().max())
    Sym = ((Am + Am.T) * 0.5).toarray()
    ev = np.linalg.eigvalsh(Sym)
    rowsum = np.asarray(Am.sum(axis=1)).ravel()
    print('%-8s n=%d  非对角元>0 个数=%-7d 对角[%.3e, %.3e]' % (CONV, n, pos, dmin, dmax))
    print('         对称部分特征值 [%.4e, %.4e]（最大应为负=耗散）  行和 max=%.3e'
          % (float(ev.min()), float(ev.max()), float(np.abs(rowsum).max())))
    del Sym
print('DONE')