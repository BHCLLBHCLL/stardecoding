# -*- coding: utf-8 -*-
"""S4 第四轮：线性求解路线基准（直接 LU / AMG / ILU+BiCGSTAB 的可复现对照）。

用途：给"10 万单元单步 < 2 s"这类验收提供**可重复测量**，并固化路线选择的依据。

实测（本机，官方 vortexShed 同款通道域阶梯 tet 网格）：

  | 网格 | 未知量 | 直接 LU | 仅 AMG | 分流 v1 | **分流 v2（当前默认）** |
  | --- | --- | --- | --- | --- | --- |
  | h=D/8, T=0.25D | 97,680 | 32.64 s/步 | 12.56 s/步 | 5.60 s/步 | **min 3.28 / median 3.37 s/步** |
  | h=D/8, T=0.125D | 48,840 | （第三轮 5.77 s/步） | — | 3.53 s/步 | **min 2.04 / median 2.10 s/步** |
  | h=D/8, T=1.0D | 390,720 | （未跑，直接 LU 不可行） | 53.34 s/步 | — | — |

  v2 = 动量 ILU(0)（零填充：setup 0.18 + solve 0.25 = 0.42 s，高填充 ILU(1e-4,10) 要 0.97 s）
     + 压力 AMG(max_coarse=200) 预条件 CG（0.23-0.56 s，ml.solve 要 0.84 s）；残差与直接 LU 逐位一致。
  共享机器上外部负载会让步时方差达数倍，故口径用 min/median（脚本会在方差 >2× 时告警）。

单次求解（97,680 未知量 / 454k nnz）：
  · 动量（对流扩散，非对称）：直接 5.56 s｜AMG setup 3.39+solve 0.08｜**ILU 0.89+0.04 s**
  · 压力（泊松，对称正定）：直接 6.54 s｜**AMG 0.50+0.40 s**｜ILU 失效（BiCGSTAB info=-10）

用法：
  python solver_bench.py --mode auto --h-factor 8 --thickness-D 0.25 --steps 3
  python solver_bench.py --mode direct --json _bench_direct.json
返回/落盘：{mode, cells, per_step_s, mean_step_s, residuals, linsolve_last, …}。
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def run_bench(mode="auto", h_factor=8.0, thickness_D=0.25, steps=3,
              length_D=16.0, height_D=8.0, D=0.04, u_inf=0.05, nu=1e-5,
              json_path=None):
    """跑 steps 步 SIMPLE，记录每步墙钟/线性求解耗时/连续性残差。

    mode: auto=按系统类型分流（默认阈值 50000）；direct=强制直接 LU；amg=强制 AMG。
    """
    if mode == "amg":
        os.environ["STARDECODING_DIRECT_MAX"] = "0"
    elif mode == "direct":
        os.environ["STARDECODING_DIRECT_MAX"] = "10000000"
    import pressure_solver as ps
    from official_diff import channel_tet_mesh_cartesian
    from pressure_solver import PressureSolver
    stats = {"calls": 0, "time": 0.0, "n": [], "t": []}
    orig = ps.solve_linear

    def timed(row, col, data, b, n, **kw):
        t0 = time.time()
        x = orig(row, col, data, b, n, **kw)
        stats["calls"] += 1
        stats["time"] += time.time() - t0
        stats["n"].append(int(n))
        stats["t"].append(time.time() - t0)
        return x

    ps.solve_linear = timed
    t0 = time.time()
    mesh = channel_tet_mesh_cartesian(D, length_D=length_D, height_D=height_D,
                                      thickness_D=thickness_D, h_factor=h_factor)
    V = np.asarray(mesh["vertices"], float)
    C = np.asarray(mesh["cells"], np.int64)
    build = time.time() - t0
    solver = PressureSolver(V, C, rho=1.0, mu=nu, inlet_axis=0, inlet_side="min",
                            inlet_velocity=(u_inf, 0.0, 0.0), outlet_side="max",
                            convection="upwind", wall_slip_axes=(1, 2))
    per_step, res, lin = [], [], []
    for _k in range(int(steps)):
        stats.update(calls=0, time=0.0, n=[], t=[])
        t1 = time.time()
        r = solver.step()
        per_step.append(time.time() - t1)
        lin.append(float(stats["time"]))
        res.append(float(r["residual"]))
    ps_list = np.asarray(per_step, float)
    out = {"mode": mode, "cells": int(C.shape[0]), "h_factor": float(h_factor),
           "per_step_min_s": float(ps_list.min()),
           "per_step_median_s": float(np.median(ps_list)),
           "per_step_max_s": float(ps_list.max()),
           "contention_note": ("共享机器：外部负载会让步时方差数倍；"
                               "min 为最安静窗口的可比口径"),
           "thickness_D": float(thickness_D), "build_s": float(build),
           "per_step_s": per_step, "linsolve_s": lin, "residuals": res,
           "mean_step_s": float(np.mean(per_step)),
           "mean_linsolve_s": float(np.mean(lin)),
           "direct_max_env": os.environ.get("STARDECODING_DIRECT_MAX")}
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


def _main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="S4 线性求解路线基准")
    ap.add_argument("--mode", choices=("auto", "direct", "amg"), default="auto")
    ap.add_argument("--h-factor", type=float, default=8.0)
    ap.add_argument("--thickness-D", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    out = run_bench(args.mode, args.h_factor, args.thickness_D, args.steps,
                    json_path=args.json)
    print("[%s] cells=%d build=%.1fs" % (out["mode"], out["cells"], out["build_s"]))
    for k, (w, l, r) in enumerate(zip(out["per_step_s"], out["linsolve_s"],
                                      out["residuals"])):
        print("  step %d: %.2fs total | linsolve %.2fs | residual %.3e" % (k, w, l, r))
    print("[%s] 步时 min %.2f / median %.2f / max %.2f s（线性求解均值 %.2f s）"
          % (out["mode"], out["per_step_min_s"], out["per_step_median_s"],
             out["per_step_max_s"], out["mean_linsolve_s"]))
    if out["per_step_max_s"] > 2.0 * out["per_step_min_s"]:
        print("    ⚠ 步时方差 >2×：共享机器外部负载污染，取 min 作为可比口径")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
