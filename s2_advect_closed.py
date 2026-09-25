# -*- coding: utf-8 -*-
"""S2 干净仪器：**封闭盒内被平均流输运的模态**的数值耗散实测。

动机（第十八步）：开域版（s2_numdiff.py）用均匀 Dirichlet 入口，每步都在抹除扰动，
与对流格式无关 ⇒ 三方格式给出同一个 ν_num 无法解释为求解器耗散。本脚本把仪器搬到**封闭域**：
  · 无出入口（x 两端零梯度），扰动不会被入口抹掉；
  · 模态取流函数形式 ψ = A·sin(kx)·sin²(πy)：
      u' = ∂ψ/∂y = A·π·sin(2πy)·sin(kx)
      v' = −∂ψ/∂x = −A·k·cos(kx)·sin²(πy)
    散度恒为零、且 u'、v' 在 y=0,1 上**精确为零**（无滑移精确满足，不激发 Stokes 层）；
  · 叠加平均流 U（无驱动，4 s 内因壁面摩擦的衰减 <0.1%），对流项活跃；
  · 振幅 = 体积加权投影到 sin(2πy)·sin(k(x−Ut))；
  · 理论衰减 λ = ν(k²+4π²) ⇒ ν_num = σ/(k²+4π²) − ν。
用法：python s2_advect_closed.py <ni> <conv> <nx> [steps] [alpha_m] [piso] [dt] [nu] [nwave]
"""
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver

NI = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CONV = sys.argv[2] if len(sys.argv) > 2 else 'central'
NX = int(sys.argv[3]) if len(sys.argv) > 3 else 32
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
AM = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
PISO = int(sys.argv[6]) if len(sys.argv) > 6 else 0
DT = float(sys.argv[7]) if len(sys.argv) > 7 else 0.01
NU = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-5
NWAVE = int(sys.argv[9]) if len(sys.argv) > 9 else 4
U = float(sys.argv[10]) if len(sys.argv) > 10 else 0.05   # U=0 时无对流（配 ν=0 即纯投影实验）
A = float(sys.argv[11]) if len(sys.argv) > 11 else 0.002  # 振幅扫描：非线性自相互作用 ∝A，数值耗散与 A 无关
RHO = 1.0

V, C = cube_tet_mesh(nx=NX, ny=NX, nz=1)
s = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0, inlet_side='min',
                   inlet_velocity=(U, 0.0, 0.0), outlet_side='max',
                   inlet_zero_gradient=True, wall_slip_axes=(2,),
                   convection=CONV, alpha_momentum=AM, piso_correctors=PISO)
cen = np.asarray(s.fvm.centroids, float)
vol = np.asarray(s.fvm.volumes, float)
Lx = float(cen[:, 0].max() - cen[:, 0].min())
Ly = float(cen[:, 1].max() - cen[:, 1].min())
x0, y0 = float(cen[:, 0].min()), float(cen[:, 1].min())
k = 2.0 * np.pi * NWAVE / Lx
ky = 2.0 * np.pi / Ly            # sin²(πy) 的等效波数（u' ∝ sin(2πy)）
mid = (cen[:, 0] > x0 + 0.25 * Lx) & (cen[:, 0] < x0 + 0.75 * Lx)
xx = cen[:, 0] - x0
yy = cen[:, 1] - y0

def uprime(t):
    return A * np.pi * np.sin(2.0 * np.pi * yy) * np.sin(k * (xx - U * t))

# 复傅里叶系数（x 向）—— 实测相位可能与名义 U 漂移，固定相位投影会把色散误差误判为耗散。
_yw = np.sin(2.0 * np.pi * yy)
_wexp = _yw * np.exp(-1j * k * xx) * vol
_den = float(np.sum(vol * _yw ** 2))


def camp(u):
    """复模态系数 c(t)；|c| = 真实振幅（相位无关），arg(c) 给出相速度信息。"""
    return complex(np.sum(_wexp * np.asarray(u, float)) / max(_den, 1e-300))


def amp(t, u):
    return abs(camp(u))

s._u = U + uprime(0.0)
s._v = -A * k * np.cos(k * xx) * np.sin(np.pi * yy) ** 2
s._w = np.zeros_like(s._u)
s._p = np.zeros_like(s._u)
s.enable_transient(DT, snapshot=True)
lam = NU * (k ** 2 + ky ** 2)
ts, amps, camps = [0.0], [amp(0.0, s._u)], [camp(s._u)]
t0 = time.time()
for step in range(STEPS):
    s.advance(dt=DT, n_inner=NI)
    ts.append(float(s.time)); amps.append(amp(s.time, s._u)); camps.append(camp(s._u))
    if not np.isfinite(amps[-1]) or abs(amps[-1]) > 1e3:
        print('DIVERGED at step %d amp=%s' % (step, amps[-1])); break
ts = np.asarray(ts); amps = np.asarray(amps)
ok = np.isfinite(amps) & (np.abs(amps) > 1e-12)
if ok.sum() >= 3:
    p = np.polyfit(ts[ok], np.log(np.abs(amps[ok])), 1)
    lam_num = -float(p[0])
    nu_num = lam_num / (k ** 2 + ky ** 2) - NU
    print('ni=%d conv=%-8s a_m=%.2f piso=%d nx=%d NWAVE=%d 用时%.1fs'
          % (NI, CONV, AM, PISO, NX, NWAVE, time.time() - t0))
    print('  振幅 A(0)=%.5f → A(%.2fs)=%.5f（%.1f%% 保留）'
          % (amps[0], ts[-1], amps[-1], 100.0 * amps[-1] / amps[0]))
    print('  测得衰减率 %.5f /s ；理论粘性 %.5f /s' % (lam_num, lam))
    print('  相位漂移（相对名义 U 平移）：%.4f rad（正=超前，负=滞后）'
          % float(np.angle(np.exp(1j * (np.angle(camps[-1] * np.conj(camps[0])) + k * U * ts[-1])))))
    print('  ⇒ ν_num = %+.3e = %+.1f× 分子 ν（等效 Re=%.1f）'
          % (nu_num, nu_num / max(NU, 1e-30), U * Lx / max(NU + nu_num, 1e-12)))
else:
    print('样本不足：%s' % np.round(amps[:5], 6))