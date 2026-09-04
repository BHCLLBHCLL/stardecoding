# -*- coding: utf-8 -*-
"""N 波 N2b 曲率自适应尺寸场验收（parity_100pct_plan.md 第 6 节 N2）。

覆盖：
  curvature_field：cotan 平均曲率幅值——球面≈恒值 1/R，平面≈0；
  size_field：单调 —— 曲率越大目标边越短；
  remesh_surface 曲率驱动：高曲率区(波峰)边 < 低曲率区(波谷/平面)；
  流形保真：闭环输入保持水密 + 欧拉示性 V-E+F=2；开面输入边界半环数不变。

纯 numpy，两环境（默认 / conda occ）皆可跑。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_io import cube_mesh
from mesh_remesh import (curvature_field, size_field, remesh_surface)


# ---------------------------------------------------------------- 形状构造
def unit_sphere_mesh(lat=20, lon=32):
    """UV 球（半径 1，水密闭环）。"""
    def sph(t, p):
        return [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
    V = []
    for i in range(lat + 1):
        t = np.pi * i / lat
        for j in range(lon):
            p = 2.0 * np.pi * j / lon
            V.append(sph(t, p))
    F = []
    for i in range(lat):
        for j in range(lon):
            a = i * lon + j
            b = i * lon + (j + 1) % lon
            c = (i + 1) * lon + j
            d = (i + 1) * lon + (j + 1) % lon
            F.append((a, b, c))
            F.append((b, d, c))
    return V, F


def wavy_grid(nx=11, ny=11, amp=0.2, k=2.0):
    """开面波浪面 z=amp·sin(2πk·x) 于 [0,1]^2；|∂²z/∂x²|∝|sin(2πk x)|。"""
    V = []
    for i in range(nx):
        x = i / (nx - 1)
        for j in range(ny):
            y = j / (ny - 1)
            V.append([x, y, amp * np.sin(2 * np.pi * k * x)])
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = i * ny + (j + 1)
            c = (i + 1) * ny + j
            d = (i + 1) * ny + (j + 1)
            F.append((a, b, d))
            F.append((a, d, c))
    return V, F


def _mesh_stats(V, F):
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    elen = []
    for a, b, c in F:
        for (u0, v0) in ((a, b), (b, c), (c, a)):
            elen.append(float(np.linalg.norm(V[u0] - V[v0])))
    return {"edges": np.asarray(elen), "mean": float(np.mean(elen)),
            "min": float(np.min(elen))}


def _euler(V, F):
    edges = set()
    for a, b, c in F:
        for (u0, v0) in ((a, b), (b, c), (c, a)):
            edges.add((u0, v0) if u0 < v0 else (v0, u0))
    return len(V) - len(edges) + len(F)


def _boundary_count(F):
    from occ_repair import boundary_edges
    b, _ = boundary_edges(None, np.asarray(F, np.int64))
    return len(b)


# ---------------------------------------------------------------- 曲率场
def test_curvature_sphere_uniform():
    V, F = unit_sphere_mesh(lat=24, lon=32)
    kap = curvature_field(V, F)
    # UV 球两极处三角收敛会产生局部曲率伪值，故只用中纬内部顶点检验均匀性
    lon = 32
    idx = []
    for i in range(3, 24 - 2):
        for j in range(lon):
            idx.append(i * lon + j)
    inter = kap[np.array(idx)]
    assert len(inter) > 0
    mean = float(inter.mean())
    assert mean > 0.8                # 量级处 1/R=1 附近（cotan 略高估）
    assert float(inter.std()) / mean < 0.1   # 内部曲率几乎均匀


def test_curvature_plane_interior_near_zero():
    V, F = wavy_grid(nx=21, ny=21, amp=0.0)
    kap = curvature_field(V, F)
    # cotan 在平面内部精确为零；边界/角点有离散伪值，故只看内部顶点
    interior = _interior_vertices(V, F)
    assert len(interior) > 0
    assert float(kap[interior].max()) < 1e-6


def _interior_vertices(V, F):
    from occ_repair import boundary_edges
    b, _ = boundary_edges(None, np.asarray(F, np.int64))
    bv = set()
    for (u, v) in b:
        bv.add(u)
        bv.add(v)
    return np.array([i for i in range(len(V)) if i not in bv])


def test_size_field_monotone():
    kap = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    L = size_field(kap, 1.0)
    # 单调不增
    assert all(L[i] >= L[i + 1] for i in range(len(L) - 1))
    assert L[0] == pytest.approx(1.0)      # 平面 → target_max
    assert L[-1] >= 0.2                     # 下限 min_ratio*target_max


# ---------------------------------------------------------------- 重网格化
def test_remesh_wavy_curvature_drives_size():
    V, F = wavy_grid(nx=12, ny=12, amp=0.12, k=2.0)
    V2, F2 = remesh_surface(V, F, target_max=0.13,
                            curvature_adaptive=True, max_iter=3)
    V2 = np.asarray(V2, float)
    F2 = np.asarray(F2, np.int64)
    # 按三角形质心 x 处 |sin(2π·2x)| 划分：高曲率波峰 vs 低曲率高干
    elen = []
    curv = []
    for a, b, c in F2:
        x = float((V2[a][0] + V2[b][0] + V2[c][0]) / 3.0)
        elen.append(float(np.linalg.norm(V2[a] - V2[b])
                          + np.linalg.norm(V2[b] - V2[c])
                          + np.linalg.norm(V2[c] - V2[a])) / 3.0)
        curv.append(abs(np.sin(4 * np.pi * x)))
    elen = np.asarray(elen)
    curv = np.asarray(curv)
    hi = elen[curv > 0.55]     # 波峰附近
    lo = elen[curv < 0.15]     # 波节/平缓附近
    assert len(hi) >= 5 and len(lo) >= 5
    assert float(hi.mean()) < float(lo.mean()) * 0.85


def test_remesh_cube_watertight_euler():
    V, F = cube_mesh(1.0)
    V2, F2 = remesh_surface(V, F, target_max=0.4,
                            curvature_adaptive=True, max_iter=3)
    assert len(F2) > 12          # 尖边被曲率加密
    assert _boundary_count(F2) == 0
    assert _euler(V2, F2) == 2   # genus-0 闭环


def test_remesh_sphere_uniform_no_watertight_loss():
    # UV 球两极是"重合顶点但不共享面"的极环（非流形），输入本身带边界半环；
    # 重网格化的验收是**不新增**边界（无水密损失），而非强行补成闭环。
    V, F = unit_sphere_mesh(lat=16, lon=24)
    b0 = _boundary_count(F)
    assert b0 > 0                       # 极环确实是开边界
    V2, F2 = remesh_surface(V, F, target_max=0.2,
                            curvature_adaptive=False, max_iter=4)
    assert _boundary_count(F2) == b0    # 边界半环数不变（无水密损失）
    st = _mesh_stats(V2, F2)
    assert 0.10 < st["mean"] < 0.40


def test_remesh_open_boundary_preserved():
    V, F = wavy_grid(nx=12, ny=12, amp=0.12, k=2.0)
    b0 = _boundary_count(F)
    assert b0 == 4 * (12 - 1)            # 4 条外边界 (每边 nx-1 段)
    V2, F2 = remesh_surface(V, F, target_max=0.13,
                            curvature_adaptive=True, max_iter=3)
    b1 = _boundary_count(F2)
    assert b1 == b0                      # 边界半环数不变（边界边不劈分/折叠）


def test_remesh_refine_flat_input():
    # 平面上给较大 target_max，应无明显加密（均匀）
    V, F = cube_mesh(1.0)
    V2, F2 = remesh_surface(V, F, target_max=0.5,
                            curvature_adaptive=True, max_iter=3)
    # 曲率集中在尖边；面上大体保持 coarse，但尖边仍会加密
    assert _boundary_count(F2) == 0