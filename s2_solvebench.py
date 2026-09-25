# -*- coding: utf-8 -*-
"""S2/S4 交叉诊断：同网格（24,432 单元阶梯）上「装配 vs 线性求解」耗时分解 + 选路对照。

背景：`stair_am1noc` 长窗跑 >1 小时连第 0 步检查点都未写出。要分清时间花在
装配还是求解、以及直接 LU 与 Krylov/AMG 在**当前负载**下的真实比值。
用法：python s2_solvebench.py [steps]（steps=0 只跑微基准）
"""
import io, os, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from official_diff import channel_tet_mesh_cartesian
from pressure_solver import (PressureSolver, solve_linear, solve_log_clear,
                             solve_log_summary)

D, U, NU, RHO = 0.04, 0.05, 1e-5, 1.0
MESH_KW = dict(length_D=16.0, height_D=8.0, thickness_D=0.5, h_factor=4.0,
               center_x_D=4.0)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1

t0 = time.time()
mesh = channel_tet_mesh_cartesian(D, **MESH_KW)
V = np.asarray(mesh['vertices'], float)
C = np.asarray(mesh['cells'], np.int64)
t_mesh = time.time() - t0
t0 = time.time()
solver = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                        inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                        convection='central', wall_slip_axes=(1, 2),
                        piso_correctors=1, alpha_pressure=0.3,
                        alpha_momentum=0.7)
t_solver = time.time() - t0
n = int(C.shape[0])
print('网格 %.2fs + 求解器构造 %.2fs ；cells=%d' % (t_mesh, t_solver, n), flush=True)
solver.enable_transient(0.01, snapshot=True)

def with_env(dm=None, amg=None):
    old = (os.environ.get('STARDECODING_DIRECT_MAX'),
           os.environ.get('STARDECODING_POISSON_AMG_MIN'))
    if dm is not None:
        os.environ['STARDECODING_DIRECT_MAX'] = str(dm)
    if amg is not None:
        os.environ['STARDECODING_POISSON_AMG_MIN'] = str(amg)
    return old

def restore_env(old):
    for key, val in zip(('STARDECODING_DIRECT_MAX', 'STARDECODING_POISSON_AMG_MIN'), old):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val

# ---------------- 微基准 1：动量装配 vs 三种求解路线 ----------------
old_env = with_env(dm=50000, amg=0)
t0 = time.time()
r, c, v, rhs, ap = solver._assemble_momentum(0)
t_asm = time.time() - t0
print('动量装配 1 次：%.3fs（nnz=%d）' % (t_asm, len(v)), flush=True)

def route(label, dm, system, **kw):
    with_env(dm=dm)
    solve_log_clear()
    t0 = time.time()
    x = solve_linear(r, c, v, rhs, n, tol=1e-9, maxit=6000, system=system)
    dt = time.time() - t0
    print('  %-30s %8.3fs  %s' % (label, dt, solve_log_summary()), flush=True)
    return np.asarray(x, float).ravel(), dt

x_dir, t_dir = route('直接 LU（阈值拉满）', 1000000, 'convection')
x_lad, t_lad = route('Krylov 阶梯（阈值 0）', 0, 'convection')
rel = np.linalg.norm(x_dir - x_lad) / max(np.linalg.norm(x_dir), 1e-300)
print('  两条路线解相对差 %.3e' % rel, flush=True)

# ---------------- 微基准 2：压力修正系统 direct vs AMG ----------------
solver._recompute_mdot_rhie_chow()
aP = ap / solver.alpha_momentum
d_cell = np.asarray(solver.fvm.volumes, float) / np.maximum(aP, 1e-12)
rp, cp, vp, rhsp, o_int, nb_int, gamma, is_int = \
    solver._assemble_pressure_correction(d_cell)
with_env(dm=1000000, amg=0)
t0 = time.time()
xp_dir = solve_linear(rp, cp, vp, rhsp, n, tol=1e-9, maxit=8000, system='poisson')
t_pdir = time.time() - t0
with_env(dm=50000, amg=4000)
t0 = time.time()
xp_amg = solve_linear(rp, cp, vp, rhsp, n, tol=1e-9, maxit=8000, system='poisson')
t_pamg = time.time() - t0
relp = (np.linalg.norm(np.asarray(xp_dir) - np.asarray(xp_amg))
        / max(np.linalg.norm(np.asarray(xp_dir)), 1e-300))
print('压力修正 %d 未知量：直接 LU %.3fs vs AMG 优先 %.3fs ；解相对差 %.3e'
      % (n, t_pdir, t_pamg, relp), flush=True)

# ---------------- 微基准 3：泊松系统的路线分解（AMG 分解 vs 迭代 vs 阶梯） ----------------
import scipy.sparse as sp
import scipy.sparse.linalg as spla
Ap = sp.csr_matrix((vp, (rp, cp)), shape=(n, n))
bp = np.asarray(rhsp, float).ravel()
bpn = max(float(np.linalg.norm(bp)), 1e-300)
try:
    import pyamg
    t0 = time.time()
    ml = pyamg.smoothed_aggregation_solver(Ap, max_coarse=200)
    t_setup = time.time() - t0
    t0 = time.time()
    xc, info = spla.cg(Ap, bp, rtol=1e-6, atol=0.0, M=ml.aspreconditioner(),
                       maxiter=8000)
    t_cg = time.time() - t0
    print('  AMG 分解 %.3fs + AMG-CG %.3fs（info=%s 残差 %.2e）'
          % (t_setup, t_cg, info, np.linalg.norm(Ap @ xc - bp) / bpn), flush=True)
    _d = Ap.diagonal()
    _d = np.where(np.abs(_d) < 1e-300, 1.0, _d)
    t0 = time.time()
    xj, info2 = spla.cg(Ap, bp, rtol=1e-6, atol=0.0,
                        M=spla.LinearOperator(Ap.shape, lambda z: z / _d),
                        maxiter=8000)
    t_jac = time.time() - t0
    print('  Jacobi-CG %.3fs（info=%s 残差 %.2e）'
          % (t_jac, info2, np.linalg.norm(Ap @ xj - bp) / bpn), flush=True)
except Exception as _exc:
    print('  pyamg 不可用：%s' % _exc, flush=True)
with_env(dm=0)
solve_log_clear()
t0 = time.time()
xl = solve_linear(rp, cp, vp, rhsp, n, tol=1e-9, maxit=8000, system='convection')
print('  泊松走 Krylov 阶梯 %.3fs %s' % (time.time() - t0, solve_log_summary()), flush=True)

# ---------------- 整步对照（装配+求解合计） ----------------
for tag, dm, amg in (('old(50000/0)', 50000, 0), ('new(3000/4000)', 3000, 4000)):
    if STEPS <= 0:
        break
    with_env(dm=dm, amg=amg)
    solve_log_clear()
    t0 = time.time()
    for _ in range(STEPS):
        solver.advance(dt=0.01, n_inner=1)
    dt = time.time() - t0
    print('  整步 %-16s ni=1 ×%d：%.3fs/步  %s'
          % (tag, STEPS, dt / STEPS, solve_log_summary()), flush=True)
restore_env(old_env)
print('DONE', flush=True)