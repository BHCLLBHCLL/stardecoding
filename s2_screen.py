# -*- coding: utf-8 -*-
"""S2 耗散源筛选（小网格，CPU 轻）：同一扰动下比较不同求解器设置的增长/衰减率。

用 hybrid 小网格（m=12/n_r=10，~7.5k 单元）跑 300 步瞬态，输出 CL 包络与对数增长率，
用于在重算（24k 阶梯网格 / 官方对照）之前**廉价筛选**可疑耗散源。
用法：python s2_screen.py <tag> <alpha_p> <convection> <ni> [steps] [alpha_m] [piso]
"""
import io, json, os, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from mesh_hybrid import hybrid_channel_mesh
from pressure_solver import PressureSolver
from aero_forces import force_coefficients

TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
ALPHA_P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
ALPHA_M = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7
PISO = int(sys.argv[7]) if len(sys.argv) > 7 else 1
CONV = sys.argv[3] if len(sys.argv) > 3 else 'central'
NI = int(sys.argv[4]) if len(sys.argv) > 4 else 1
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 300
D, U, NU, RHO, DT = 0.04, 0.05, 1e-5, 1.0, 0.01

MP = dict(D=D, m=24, n_r=16, thickness_D=0.25, h_factor=4.0)   # 与 run4 同参
try:
    mesh = hybrid_channel_mesh(**MP)
except Exception as exc:  # noqa: BLE001
    print('mesh failed: %r' % exc)
    raise
V = np.asarray(mesh['vertices'], float)
C = np.asarray(mesh['cells'], np.int64)
solver = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                        inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                        convection=CONV, wall_slip_axes=(1, 2),
                        piso_correctors=PISO, alpha_pressure=ALPHA_P,
                        alpha_momentum=ALPHA_M,
                        nonorth_corrected=True, corr_limit=0.33)
cc = np.asarray(solver.fvm.centroids, float)[:, :2]
cx0, cy0 = float(mesh['hole_center'][0]), float(mesh['hole_center'][1])
blob = np.exp(-(((cc[:, 0] - cx0) / D) ** 2
                + ((cc[:, 1] - (cy0 + 0.6 * D)) / (0.5 * D)) ** 2))
solver.perturb_velocity(dv=0.2 * U * blob)
solver.enable_transient(DT, snapshot=True)
fv = solver.fvm
hc = np.asarray(mesh['hole_center'], float)[:2]
rh = np.linalg.norm(np.asarray(fv.face_centroid, float)[:, :2] - hc, axis=1)
cyl = np.where(np.asarray(fv.is_boundary, bool)
               & (rh <= float(mesh['hole_r']) + 1e-9))[0]
ts, cls = [], []
t0 = time.time()
for k in range(STEPS):
    solver.advance(dt=DT, n_inner=NI)
    vel = solver.velocity()
    fc = force_coefficients(fv, solver.pressure(), vel[:, 0], vel[:, 1], vel[:, 2],
                            solver.mu, solver.rho, faces=cyl, a_ref=D, u_ref=U,
                            rho_ref=RHO, drag_dir=(1.0, 0.0, 0.0),
                            lift_dir=(0.0, 1.0, 0.0))
    ts.append(float(solver.time)); cls.append(float(fc.get('cl', float('nan'))))
    if not np.isfinite(cls[-1]) or abs(cls[-1]) > 50:
        print('%s DIVERGED at step %d' % (TAG, k)); break
cls = np.asarray(cls, float); ts = np.asarray(ts, float)
n = cls.size; win = max(n // 4, 5)
env = [(float(np.abs(cls[i:i+win] - cls[i:i+win].mean()).max()), float(ts[i]))
       for i in range(0, n - win + 1, win)]
print('%-10s cells=%d conv=%-8s a_m=%.2f a_p=%.2f ni=%d piso=%d steps=%d  %.2fs/step'
      % (TAG, C.shape[0], CONV, ALPHA_M, ALPHA_P, NI, PISO, n,
         (time.time() - t0) / max(n, 1)))
print('   包络: %s' % ['%.5f@%.1fs' % (v, x) for v, x in env])
ok = np.array([v > 1e-7 for v, _ in env])
if ok.sum() >= 2:
    x = np.array([e[1] for e in env])[ok]; y = np.array([e[0] for e in env])[ok]
    p = np.polyfit(x, np.log(y), 1)
    print('   σ = %+.3f /s（%s）' % (p[0], '增长' if p[0] > 0 else '衰减'))
json.dump({'tag': TAG, 'alpha_m': ALPHA_M, 'alpha_p': ALPHA_P, 'conv': CONV, 'ni': NI,
           'piso': PISO,
           'cells': int(C.shape[0]), 't': ts.tolist(), 'cl': cls.tolist()},
          open('_screen_%s.json' % TAG, 'w', encoding='utf-8'))