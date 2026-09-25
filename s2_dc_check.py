# -*- coding: utf-8 -*-
"""S2 诊断 2：延迟修正（central/limited）在**装配层面**是否真的生效？

背景：干净仪器里 upwind 与 central 的瞬态耗散只差 14%（48× vs 55×ν），而矩阵级测量显示
central 的对流算子数值扩散 ≈1e-19 —— 两者矛盾。若「同一速度场下 central 与 upwind 的装配
矩阵（除对角外）与 RHS 几乎相同」，即说明二阶修正项在装配里被丢弃/未被计入。
用法：python s2_dc_check.py [nx] [step]   （step=0 用初值，step=n 先推进 n 步再比）
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver

NX = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
U, A, NU, RHO, DT = 0.05, 0.002, 1e-5, 1.0, 0.01

def build(conv):
    V, C = cube_tet_mesh(nx=NX, ny=NX, nz=1)
    s = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                       inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                       inlet_zero_gradient=True, wall_slip_axes=(2,),
                       convection=conv, alpha_momentum=1.0, piso_correctors=0)
    cen = np.asarray(s.fvm.centroids, float)
    Lx = float(cen[:, 0].max() - cen[:, 0].min())
    x0, y0 = float(cen[:, 0].min()), float(cen[:, 1].min())
    k = 2.0 * np.pi * 4 / Lx
    xx, yy = cen[:, 0] - x0, cen[:, 1] - y0
    s._u = U + A * np.pi * np.sin(2.0 * np.pi * yy) * np.sin(k * xx)
    s._v = -A * k * np.cos(k * xx) * np.sin(np.pi * yy) ** 2
    s._w = np.zeros_like(s._u); s._p = np.zeros_like(s._u)
    s.enable_transient(DT, snapshot=True)
    if NSTEP > 0:
        for _ in range(NSTEP):
            s.advance(dt=DT, n_inner=1)
    return s

res = {}
for conv in ('upwind', 'central'):
    s = build(conv)
    r, c, v, rhs, ap = s._assemble_momentum(0)
    res[conv] = (np.asarray(r), np.asarray(c), np.asarray(v), np.asarray(rhs), np.asarray(ap))
    print('%-8s nnz=%d  |rhs|=%.6e  max|rhs|=%.4e'
          % (conv, len(v), float(np.linalg.norm(rhs)), float(np.abs(rhs).max())))

r1, c1, v1, b1, a1 = res['upwind']
r2, c2, v2, b2, a2 = res['central']
same_struct = (len(v1) == len(v2) and np.array_equal(r1, r2) and np.array_equal(c1, c2))
print('结构相同：%s' % same_struct)
if same_struct:
    print('矩阵系数最大差 %.4e（相对 %.3e）'
          % (float(np.abs(v1 - v2).max()), float(np.abs(v1 - v2).max() / max(np.abs(v1).max(), 1e-300))))
    print('RHS 差值范数 %.4e（相对 %.3e）'
          % (float(np.linalg.norm(b1 - b2)), float(np.linalg.norm(b1 - b2) / max(np.linalg.norm(b1), 1e-300))))
    print('对角最大差 %.4e' % float(np.abs(a1 - a2).max()))
print('DONE')