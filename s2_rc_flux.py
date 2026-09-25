# -*- coding: utf-8 -*-
"""S2 首要嫌疑判定：动量对流用的面质量通量（Rhie-Chow 修正后）与「理想插值通量」差多少？

动机：本轮实测确认额外耗散「只需对流开启、与粘性/格式/内迭代/投影无关」，与「动量对流项使用
RC 修正面通量」这一实现特征完全吻合。本脚本推进若干步后，直接量化二者之差。
用法：python s2_rc_flux.py [nx] [steps]
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver

NX = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
U, A, NU, RHO, DT = 0.05, 0.002, 1e-5, 1.0, 0.01

V, C = cube_tet_mesh(nx=NX, ny=NX, nz=1)
s = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                   inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                   inlet_zero_gradient=True, wall_slip_axes=(2,),
                   convection='central', alpha_momentum=1.0, piso_correctors=0)
cen = np.asarray(s.fvm.centroids, float)
fv = s.fvm
Lx = float(cen[:, 0].max() - cen[:, 0].min())
x0, y0 = float(cen[:, 0].min()), float(cen[:, 1].min())
k = 2.0 * np.pi * 4 / Lx
xx, yy = cen[:, 0] - x0, cen[:, 1] - y0
s._u = U + A * np.pi * np.sin(2.0 * np.pi * yy) * np.sin(k * xx)
s._v = -A * k * np.cos(k * xx) * np.sin(np.pi * yy) ** 2
s._w = np.zeros_like(s._u); s._p = np.zeros_like(s._u)
s.enable_transient(DT, snapshot=True)

def ideal_mdot():
    """理想插值通量：ρ·(0.5(u_o+u_nb)·n)·A（内部面，中心插值速度）。"""
    o = np.asarray(fv.owner, int); nb = np.asarray(fv.neighbor, int)
    n = np.asarray(fv.face_normal, float); ar = np.asarray(fv.face_area, float)
    vel = np.stack([s._u, s._v, s._w], axis=1)
    vf = 0.5 * (vel[o] + vel[nb])
    return RHO * np.einsum('ij,ij->i', vf, n) * ar

is_int = ~np.asarray(fv.is_boundary, bool)
for step in range(NSTEP + 1):
    s._recompute_mdot_rhie_chow()
    rc = np.asarray(s._mdot, float)
    idl = ideal_mdot()
    d = rc[is_int] - idl[is_int]
    ref = max(float(np.linalg.norm(idl[is_int])), 1e-300)
    print('step %d  t=%.2f  内部面 %d：|Δṁ|/|ṁ_ideal| = %.4e   max|Δṁ|/max|ṁ| = %.4e'
          % (step, float(s.time), int(is_int.sum()), float(np.linalg.norm(d)) / ref,
             float(np.abs(d).max()) / max(float(np.abs(idl[is_int]).max()), 1e-300)))
    if step < NSTEP:
        s.advance(dt=DT, n_inner=1)
print('DONE')