# -*- coding: utf-8 -*-
"""S2 判别实验（我方一侧）：在**与官方同网格同参数**上跑本仓库瞬态，记录 CL(t)。

网格/CGNS 与 s6_samemesh.prepare 完全一致（channel_tet_mesh_cartesian + MESH_KW），
边界 wall_slip_axes=(1,2) 对应官方侧的四个 SymmetryBoundary，dt=0.01 与官方一致。
用途：与官方同网格 CL(t) 对比 → 判别"网格 vs 求解器"。
"""
import io, json, os, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from official_diff import channel_tet_mesh_cartesian
from pressure_solver import PressureSolver
from aero_forces import force_coefficients

D, U, NU, RHO = 0.04, 0.05, 1e-5, 1.0
MESH_KW = dict(length_D=16.0, height_D=8.0, thickness_D=0.5, h_factor=4.0,
               center_x_D=4.0)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NI = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
CONV = sys.argv[4] if len(sys.argv) > 4 else "central"
ALPHA_P = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
ALPHA_M = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7
DT = 0.01
OUT = 'ours_cl_series%s.json' % (("_" + TAG) if TAG else "")

mesh = channel_tet_mesh_cartesian(D, **MESH_KW)
V = np.asarray(mesh['vertices'], float)
C = np.asarray(mesh['cells'], np.int64)
solver = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                        inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                        convection=CONV, wall_slip_axes=(1, 2),
                        piso_correctors=1, alpha_pressure=ALPHA_P,
                        alpha_momentum=ALPHA_M)
cc = np.asarray(solver.fvm.centroids, float)[:, :2]
cx0, cy0 = float(mesh['hole_center'][0]), float(mesh['hole_center'][1])
blob = np.exp(-(((cc[:, 0] - cx0) / D) ** 2
                + ((cc[:, 1] - (cy0 + 0.6 * D)) / (0.5 * D)) ** 2))
solver.perturb_velocity(dv=0.2 * U * blob)
solver.enable_transient(DT, snapshot=True)
fv = solver.fvm
hc = np.asarray(mesh['hole_center'], float)[:2]
rh = np.linalg.norm(np.asarray(fv.face_centroid, float)[:, :2] - hc, axis=1)
cyl_r = float(mesh['hole_r']) + float(mesh.get('h', 0.0) or 0.0)
cyl = np.where(np.asarray(fv.is_boundary, bool) & (rh <= cyl_r + 1e-9))[0]
print('ours: cells=%d cyl_faces=%d steps=%d ni=%d conv=%s a_m=%s a_p=%s dt=%s'
      % (C.shape[0], cyl.size, STEPS, NI, CONV, ALPHA_M, ALPHA_P, DT), flush=True)
out_dir = os.path.join(ROOT, 's6_samemesh', 'transient')
os.makedirs(out_dir, exist_ok=True)
ts, cls, cds = [], [], []
t0 = time.time()
for k in range(STEPS):
    solver.advance(dt=DT, n_inner=NI)
    vel = solver.velocity()
    fc = force_coefficients(fv, solver.pressure(), vel[:, 0], vel[:, 1], vel[:, 2],
                            solver.mu, solver.rho, faces=cyl, a_ref=D, u_ref=U,
                            rho_ref=RHO, drag_dir=(1.0, 0.0, 0.0),
                            lift_dir=(0.0, 1.0, 0.0))
    ts.append(float(solver.time))
    cls.append(float(fc.get('cl', float('nan'))))
    cds.append(float(fc.get('cd', float('nan'))))
    if not np.isfinite(cls[-1]) or abs(cls[-1]) > 50:
        print('ours: DIVERGED at step %d cl=%s' % (k, cls[-1]), flush=True)
        break
    if k % 100 == 0:
        json.dump({'dt': DT, 'n_inner': NI, 'steps_done': k, 't': ts, 'cl': cls,
                   'cd': cds, 'done': False},
                  open(os.path.join(out_dir, OUT), 'w', encoding='utf-8'))
        print('  ours step %4d t=%.2f cl=%+.5f cd=%.5f %.2fs/step'
              % (k, ts[-1], cls[-1], cds[-1], (time.time() - t0) / (k + 1)),
              flush=True)
json.dump({'dt': DT, 'n_inner': NI, 'steps_done': len(ts), 't': ts, 'cl': cls,
           'cd': cds, 'done': True},
          open(os.path.join(out_dir, OUT), 'w', encoding='utf-8'))
arr = np.asarray(cls, float)
print('ours END steps=%d cl_final=%+.5f cl_min=%+.5f cl_max=%+.5f'
      % (len(arr), arr[-1], np.nanmin(arr), np.nanmax(arr)), flush=True)