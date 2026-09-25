# -*- coding: utf-8 -*-
"""S2 有效数值扩散实测：散度自由移动模态的振幅衰减（对流不改振幅，只有扩散改）。

模态（散度自由，且在 y=0,1 上满足无滑移）：
  u' = A·cos(k(x−Ut))·sin(πy),  v' = −A·(k/π)·sin(k(x−Ut))·cos(πy)
线性化对流把它整体以速度 U 平移（振幅不变），只有粘性 + 数值扩散使其衰减：
  振幅 ∝ exp(−(ν+ν_num)(k²+π²)·t)
故测得衰减率即可反解 ν_num（= 实际运行时的等效数值扩散，含延迟修正实现的份额）。
测量只在中段 x∈[0.25,0.75] 投影，避开入/出口。
用法：python s2_numdiff.py <ni> <conv> <nx> [steps] [alpha_m] [piso] [dt] [nu]

⚠️ **绝对值不可引用（第十八步结论三，实测判定）**：本实现入口是**均匀 Dirichlet**，而解析解要求
入口处带扰动 ⇒ 每步都在「抹掉」扰动，抹除前沿以 U 内推（4 s 推进 0.2，逼近 x=0.25 测量窗）。
该伪影与对流格式无关，正是本脚本给出「三格式 ν_num 相同」的原因；它在 4 s 窗内额外压低约 5×
（upwind 保留率：本脚本 11.9% vs 封闭盒版 `s2_advect_closed.py` 35.8%）。
⇒ **需要定量时请用 `s2_advect_closed.py`**（封闭盒、精确 BC、复傅里叶振幅）；本脚本仅用于
「格式无关」这一结构性观察与历史对照。
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver

NI = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CONV = sys.argv[2] if len(sys.argv) > 2 else 'upwind'
NX = int(sys.argv[3]) if len(sys.argv) > 3 else 32
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
AM = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
PISO = int(sys.argv[6]) if len(sys.argv) > 6 else 1
U, A, RHO = 0.05, 0.002, 1.0
DT = float(sys.argv[7]) if len(sys.argv) > 7 else 0.01
NU = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-5
NWAVE = 4                       # 4 个波长跨域（k = 2π·4/Lx）

V, C = cube_tet_mesh(nx=NX, ny=NX, nz=1)
s = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                   inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                   convection=CONV, wall_slip_axes=(2,),
                   alpha_momentum=AM, piso_correctors=PISO)
cen = np.asarray(s.fvm.centroids, float)
vol = np.asarray(s.fvm.volumes, float)
Lx = float(cen[:, 0].max() - cen[:, 0].min())
Ly = float(cen[:, 1].max() - cen[:, 1].min())
k = 2.0 * np.pi * NWAVE / Lx
ky = np.pi / Ly
x0 = float(cen[:, 0].min()); y0 = float(cen[:, 1].min())
mid = (cen[:, 0] > x0 + 0.25 * Lx) & (cen[:, 0] < x0 + 0.75 * Lx)

def mode_u(x, y, t):
    return A * np.cos(k * (x - U * t)) * np.sin(ky * (y - y0))

def mode_v(x, y, t):
    return -A * (k / ky) * np.sin(k * (x - U * t)) * np.cos(ky * (y - y0))

xu, yu = cen[:, 0], cen[:, 1]
def amp(t, u_field):
    base = mode_u(xu, yu, t)
    num = float(np.sum(vol[mid] * u_field[mid] * base[mid]))
    den = float(np.sum(vol[mid] * base[mid] ** 2))
    return num / den if den > 0 else float('nan')

s._u = U + mode_u(xu, yu, 0.0)
s._v = mode_v(xu, yu, 0.0)
s._w = np.zeros_like(s._u)
s._p = np.zeros_like(s._u)
s.enable_transient(DT, snapshot=True)
lam_th = NU * (k ** 2 + ky ** 2)
ts, amps = [0.0], [amp(0.0, s._u)]
for step in range(STEPS):
    s.advance(dt=DT, n_inner=NI)
    ts.append(float(s.time)); amps.append(amp(s.time, s._u))
    if not np.isfinite(amps[-1]):
        print('DIVERGED at step %d' % step); break
ts = np.asarray(ts); amps = np.asarray(amps)
ok = np.isfinite(amps) & (np.abs(amps) > 1e-12)
if ok.sum() >= 3:
    p = np.polyfit(ts[ok], np.log(np.abs(amps[ok])), 1)
    lam_num = -float(p[0])
    nu_num = lam_num / (k ** 2 + ky ** 2) - NU
    print('ni=%d conv=%-8s a_m=%.2f piso=%d nx=%d  模态 k=%.1f ky=%.1f'
      % (NI, CONV, AM, PISO, NX, k, ky))
    print('  振幅 A(0)=%.5f → A(%.2fs)=%.5f（%.1f%% 保留）'
          % (amps[0], ts[-1], amps[-1], 100.0 * amps[-1] / amps[0]))
    print('  测得衰减率 %.5f /s ；理论粘性 %.5f /s' % (lam_num, lam_th))
    print('  ⇒ 有效数值扩散 ν_num = %+.3e（= %+.1f× 分子 ν=%.1e）'
          % (nu_num, nu_num / NU, NU))
    print('  ⇒ 等效 Re = U·Lx/(ν+ν_num) = %.2f（物理 Re = %.0f）'
          % (U * Lx / (NU + max(nu_num, -NU * 0.99)), U * Lx / NU))
else:
    print('样本不足/发散：%s' % np.round(amps[:5], 6))