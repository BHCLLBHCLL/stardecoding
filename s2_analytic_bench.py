# -*- coding: utf-8 -*-
"""S2 解析瞬态基准：封闭盒内 sin(πy) 剪切模的纯扩散衰减（对流项恒为零）。

设置：单位立方体 tet 网格（ny=16），y 壁无滑移、z 壁滑移、x 两端**零梯度**（新选项），
初始 u = A·sin(πy)、v=w=0、p=0。精确解 u(y,t) = A·sin(πy)·exp(−ν π² t)（对流项为零），
故测得的衰减率与 νπ² 之差即「时间推进 + 压力-速度耦合」的数值阻尼。
用法：python _s2_analytic.py [steps] [dt] [nu] [ny]
"""
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
NU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
NY = int(sys.argv[4]) if len(sys.argv) > 4 else 16
NI = int(sys.argv[5]) if len(sys.argv) > 5 else 1
NOC = (sys.argv[8] if len(sys.argv) > 8 else "0") not in ("0", "", "false", "no")
CORR = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
PISO = int(sys.argv[6]) if len(sys.argv) > 6 else 1
AM = float(sys.argv[7]) if len(sys.argv) > 7 else 0.7
A = 0.05

V, C = cube_tet_mesh(nx=4, ny=NY, nz=1)
s = PressureSolver(V, C, rho=1.0, mu=NU, inlet_axis=0, inlet_side='min',
                   inlet_velocity=(0.0, 0.0, 0.0), outlet_side='max',
                   inlet_zero_gradient=True, wall_slip_axes=(2,),
                   convection='upwind', piso_correctors=PISO,
                   alpha_momentum=AM, nonorth_corrected=NOC, corr_limit=CORR)
y = np.asarray(s.fvm.centroids, float)[:, 1]
u0 = A * np.sin(np.pi * y)
s._u = u0.copy(); s._v = np.zeros_like(u0); s._w = np.zeros_like(u0)
s._p = np.zeros_like(u0)
mid = int(np.argmin(np.abs(y - 0.5)))
print('cells=%d ni=%d piso=%d a_m=%.2f nonorth=%s corr=%.2f  y=%.4f u0=%.6f'
      % (C.shape[0], NI, PISO, AM, NOC, CORR, y[mid], s._u[mid]))
s.enable_transient(DT, snapshot=True)
lam = NU * np.pi ** 2
print('精确衰减率 νπ² = %.5f /s（理论 u(t)/u(0) = exp(−%.5f t)）' % (lam, lam))
ts, us = [0.0], [float(s._u[mid])]
t0 = time.time()
for k in range(STEPS):
    s.advance(dt=DT, n_inner=NI)
    ts.append(float(s.time)); us.append(float(s._u[mid]))
    if not np.isfinite(us[-1]):
        print('DIVERGED at step %d' % k); break
ts = np.asarray(ts); us = np.asarray(us)
ratio_num = us[-1] / us[0]
ratio_exact = np.exp(-lam * ts[-1])
sigma_num = -np.log(max(ratio_num, 1e-300)) / ts[-1] if ts[-1] > 0 else float('nan')
print('步数 %d  dt=%.4g  终点 t=%.3f s  用时 %.1fs' % (len(ts) - 1, DT, ts[-1], time.time() - t0))
print('数值 u(t)/u(0) = %.5f ；精确 = %.5f ；比值 = %.4f' % (ratio_num, ratio_exact, ratio_num / ratio_exact))
print('数值衰减率 σ = %.5f /s ；精确 λ = %.5f /s ；多出阻尼 = %.5f /s（%.1f%%）'
      % (sigma_num, lam, sigma_num - lam, 100.0 * (sigma_num / lam - 1.0)))
# 分窗速率（看是否随时间变化）
n = len(ts)
for i in range(4):
    a, b = i * n // 4, (i + 1) * n // 4
    if b > a and us[a] > 0 and us[b] > 0:
        sg = -np.log(us[b] / us[a]) / max(ts[b] - ts[a], 1e-12)
        print('  窗%d [%.2f, %.2f]s: σ=%.5f /s' % (i, ts[a], ts[b], sg))