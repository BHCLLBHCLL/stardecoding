# -*- coding: utf-8 -*-
"""N 波 N5（二）thin mesher 薄壁网格验收（parity_100pct_plan.md 第 N5 节）。

覆盖（mesh_thin 模块 docstring 验收要点）：
  薄盒 [0,1]²×0.05：仅底/顶面薄、底面 2 列贯通、体积=盒体积精确
  （n_through=1/2 一致）；厚立方体：无薄面 ok=False；单薄向盒
  1×1×0.3@阈值0.5：侧壁不薄、体积=0.3 精确；球壳 r=0.9..1.0：
  全部外球面贯通、体积≈壳体积（10% 内）、零翻转。

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
from mesh_thin import thin_mesh


# ---------------------------------------------------------------- 形状构造
def box(width=1.0, depth=1.0, height=0.05, z0=0.0):
    """长方体 [0,width]×[0,depth]×[z0,z0+height]（水密，绕向同 cube_mesh 一致朝内）。"""
    w = width
    d = depth
    h = height
    v = [
        [0, 0, z0], [w, 0, z0], [w, d, z0], [0, d, z0],
        [0, 0, z0 + h], [w, 0, z0 + h], [w, d, z0 + h], [0, d, z0 + h],
    ]
    f = [
        [0, 1, 2], [0, 2, 3],                      # bottom z=z0
        [4, 6, 5], [4, 7, 6],                      # top z=z0+h
        [0, 4, 5], [0, 5, 1],                      # y=0
        [2, 6, 7], [2, 7, 3],                      # y=d
        [0, 3, 7], [0, 7, 4],                      # x=0
        [1, 5, 6], [1, 6, 2],                      # x=w
    ]
    return v, f


def _sph(t, phi, r):
    """球坐标 → 直角坐标：t 极角、phi 方位角、r 半径。"""
    return [r * np.sin(t) * np.cos(phi),
            r * np.sin(t) * np.sin(phi),
            r * np.cos(t)]


def _uv_sphere(r, lat, lon):
    """极点合并的干净 UV 球（单极点顶点，水密，无退化面）。"""
    V = [_sph(0.0, 0.0, r)]                      # 北极 v=0
    ring = [None] * (lat + 1)
    ring[0] = 0
    for i in range(1, lat):
        ring[i] = len(V)
        t = np.pi * i / lat
        for j in range(lon):
            V.append(_sph(t, 2.0 * np.pi * j / lon, r))
    ring[lat] = len(V)
    V.append(_sph(np.pi, 0.0, r))                # 南极
    south = len(V) - 1
    F = []
    r1 = ring[1]
    for j in range(lon):                        # 北极帽
        F.append((0, r1 + j, r1 + (j + 1) % lon))
    for i in range(1, lat - 1):                 # 中纬带（a,b,c / b,d,c）
        a0 = ring[i]
        b0 = ring[i + 1]
        for j in range(lon):
            a = a0 + j
            b = a0 + (j + 1) % lon
            c = b0 + j
            d = b0 + (j + 1) % lon
            F.append((a, b, c))
            F.append((b, d, c))
    last = ring[lat - 1]
    for j in range(lon):                        # 南极帽
        F.append((south, last + (j + 1) % lon, last + j))
    return V, F


def spherical_shell(r_in=0.9, r_out=1.0, lat=12, lon=16):
    """球壳双分量表面：内壳 + 外壳（各自水密，绕向均朝心）。

    外壳面法向朝内(-r̂)：side='+n' 时推进方向朝心，从外壳质心命中
    内壳距离 r_out-r_in=0.1 → 判薄；内壳同样绕向朝心，但其质心沿
    -r̂ 命中对侧内壳距离≈2·r_in=1.8（>阈值）→ 不判薄。源面=外壳
    （面索引小），贯通方向沿 -r̂ 在外壳与内壳间填充壳隙。极点合并
    避免重复顶点退化三角形，薄判定 h≈0.1 一致、零翻转。
    """
    V, F = _uv_sphere(r_out, lat, lon)
    n = len(V)
    Vi, Fi = _uv_sphere(r_in, lat, lon)
    V.extend(Vi)
    for f in Fi:
        F.append(tuple(int(x) + n for x in f))
    return V, F


# ------------------------------------------------------- 薄盒 0.1×0.1×0.05
def test_thin_box_volume_exact_nthrough():
    """薄盒 [0,1]²×0.05：仅底/顶面薄、体积=盒体积精确（n_through=1/2）。"""
    V, F = box(1.0, 1.0, 0.05)
    for n in (1, 2):
        r = thin_mesh(V, F, thickness=0.05, n_through=n, side="inward")
        assert r["ok"]
        # 盒体积 = 1×1×0.05
        assert r["volume_total"] == pytest.approx(0.05, rel=1e-6)
        assert r["n_thin"] > 0
        # n_through 分层的体积一致（贯通全厚）
        assert r["quality"]["n_inverted"] == 0
        assert r["quality"]["sign_consistent"]


def test_thin_box_only_bottom_top_thin():
    """薄盒 [0,1]²×0.05：仅底/顶面判薄，侧壁不薄。"""
    V, F = box(1.0, 1.0, 0.05)
    # 阈值 0.055：底/顶厚 0.05 判薄（浮点上保留 0.0005 余量），侧壁厚 1.0 不薄
    r = thin_mesh(V, F, thickness=0.055, side="inward")
    # 盒共 12 面：底(0,1)、顶(2,3) 应判薄（厚 0.05），侧壁 8 面应不薄
    assert r["n_thin"] == 4          # 底/顶各 2 面
    # 侧壁面对应向质心厚度 = 1.0（不薄）
    assert r["n_sources"] == 2       # 配对取小索引侧贯通（底为源，顶 consumed）
    assert r["n_sources"] > 0
    assert r["volume_total"] == pytest.approx(0.05, rel=1e-6)


def test_thick_cube_no_thin_faces():
    """厚立方体（边长 1）：无薄面 → ok=False。"""
    V, F = cube_mesh(1.0)
    r = thin_mesh(V, F, thickness=0.5, side="inward")
    assert not r["ok"]
    assert r["quality"]["n_cells"] == 0      # 无薄面 → 无单元（n_cells 在 quality 内）
    assert r["n_thin"] == 0


def test_single_thin_direction_box():
    """单薄向盒 1×1×0.3 @ 阈值0.5：仅 z 向薄，侧壁不薄、体积=0.3 精确。"""
    V, F = box(1.0, 1.0, 0.3)
    r = thin_mesh(V, F, thickness=0.5, side="inward")
    assert r["ok"]
    # 底/顶面判薄（厚 0.3），侧壁面质心厚度 = 1.0（>0.5 不判薄）
    assert r["n_thin"] == 4
    assert r["volume_total"] == pytest.approx(0.3, rel=1e-6)
    assert r["quality"]["n_inverted"] == 0


def test_spherical_shell():
    """球壳 r=0.9..1.0：全部外壳贯通、体积≈壳体积（10% 内）、零翻转。"""
    V, F = spherical_shell(0.9, 1.0, lat=12, lon=16)
    r = thin_mesh(V, F, thickness=0.15, side="+n")
    assert r["ok"]
    shell_vol = 4.0 / 3.0 * np.pi * (1.0 ** 3 - 0.9 ** 3)
    assert r["volume_total"] == pytest.approx(shell_vol, rel=0.10)
    assert r["quality"]["n_inverted"] == 0
    assert r["quality"]["sign_consistent"]


def test_invalid_thin_inputs_raise():
    """非法输入抛 ValueError。"""
    V, F = cube_mesh(1.0)
    with pytest.raises(ValueError):
        thin_mesh([], [], thickness=0.1)
    with pytest.raises(ValueError):
        thin_mesh(V, F, thickness=-0.1)
