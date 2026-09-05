# -*- coding: utf-8 -*-
"""N 波 N5（一）：自定义尺寸控制 per part/surface/vertex/region（纯 numpy）。

STAR-CCM+ 自定义控制（Custom Size Controls）的核心思想：背景尺寸为基础，
局部控制（表面/顶点/体积区域）在其影响范围内覆写尺寸，控制区外按增长率
平滑过渡回背景——尺寸场单调有界、无突变。

实现：
  - `resolve_size_field`：多源 Dijkstra 尺寸场解析——
      · 种子：surface_sizes（{面: target 或 {min/target/max}}，种到该面
        全部顶点，target 先夹到面级 min/max）、vertex_sizes（{顶点: 尺寸}，
        per-vertex 自定义值）、region_sizes（[(中心, 半径, 尺寸)]，体积
        区域控制，种到半径内全部顶点）；多种子同顶点取最小；
      · 场：final(v) = min(base, min over 种子 u (s_u + growth·d_geo(v,u)))，
        d_geo = 沿网格边（顶点图）的 Dijkstra 距离，growth = 尺寸单位/
        长度单位（离开控制区线性增长回背景）；邻点尺寸差 ≤ growth·边长
        （梯度有界，重网格化无突变带）；
      · 终值再夹全局 [size_min, size_max]。控制只能局部加密（更小）：
        粗于背景的控制应直接调大 base_size（per part 尺寸 = 该 part
        网格的 base_size，网格器逐 part 调用）。
  - `remesh_with_controls`：解析尺寸场 → `mesh_remesh.remesh_surface(
    sizes=...)`（N5 接入：sizes 模式下目标边长逐顶点给定，随 split 演化
    ——新顶点取两端均值；曲率场旁路）。

验收要点（tests/test_mesh_control.py）：
  立方体单面控制：底面顶点=target 精确、对角顶点=target+growth×边长
  精确（梯度有界且饱和于 base）；min/max 夹取（面级+全局）；顶点/区域
  控制精确；10×10 栅格局部加密：近控制区边长远小于远场、边界边数保持、
  两次运行逐位一致（确定性）。
"""
import heapq

import numpy as np


def _vertex_adjacency(F):
    """无向顶点邻接表 {v: [u, ...]}（双向、去重边）。"""
    sets = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            sets.setdefault(int(u), set()).add(int(v))
            sets.setdefault(int(v), set()).add(int(u))
    return sets


def resolve_size_field(vertices, faces, base_size, surface_sizes=None,
                       vertex_sizes=None, region_sizes=None, growth=0.3,
                       size_min=None, size_max=None):
    """多源 Dijkstra 尺寸场解析 → 每顶点目标尺寸 (N,)。

    参数：
      base_size：背景尺寸（>0）；surface_sizes：{面索引: target 或
        {"min":m,"target":t,"max":M}}；vertex_sizes：{顶点索引: 尺寸}；
      region_sizes：[(中心(3,), 半径, 尺寸)]（欧氏距离 ≤ 半径内顶点播种）；
      growth：离开控制区的尺寸增长率（尺寸单位/长度单位，>0）；
      size_min/size_max：终值全局夹取（None 不夹）。

    空输入抛 ValueError；控制索引越界抛 ValueError。
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or len(V) == 0 or len(F) == 0:
        raise ValueError("resolve_size_field 需要非空表面 (vertices Nx3, faces Mx3)")
    N = len(V)
    base = float(base_size)
    if not (base > 0.0):
        raise ValueError("base_size 须为正数")
    g = float(growth)
    if not (g > 0.0):
        raise ValueError("growth 须为正数（尺寸单位/长度单位）")

    seed = {}                                   # 顶点 → 播种尺寸（多种子取最小）

    def _put(v, s):
        s = float(s)
        if not (s > 0.0):
            raise ValueError("控制尺寸须为正数")
        if v not in seed or s < seed[v]:
            seed[v] = s

    if surface_sizes:
        for fi, spec in surface_sizes.items():
            fi = int(fi)
            if not (0 <= fi < len(F)):
                raise ValueError("surface_sizes 面索引越界: %d" % fi)
            if isinstance(spec, dict):
                if "target" not in spec:
                    raise ValueError("surface_sizes 字典形式须含 target")
                t = float(spec["target"])
                if "min" in spec:
                    t = max(t, float(spec["min"]))
                if "max" in spec:
                    t = min(t, float(spec["max"]))
            else:
                t = float(spec)
            for v in F[fi]:
                _put(int(v), t)
    if vertex_sizes:
        for v, s in vertex_sizes.items():
            v = int(v)
            if not (0 <= v < N):
                raise ValueError("vertex_sizes 顶点索引越界: %d" % v)
            _put(v, s)
    if region_sizes:
        for center, radius, s in region_sizes:
            c = np.asarray(center, float)
            rad = float(radius)
            if not (rad > 0.0):
                raise ValueError("region_sizes 半径须为正数")
            dist = np.linalg.norm(V - c, axis=1)
            for v in np.where(dist <= rad + 1e-12)[0]:
                _put(int(v), s)

    best = np.full(N, base)
    adj = _vertex_adjacency(F)
    heap = []
    for v, s in seed.items():                   # 多源 Dijkstra：边费 = growth×边长
        if s < best[v]:
            best[v] = s
            heapq.heappush(heap, (s, v))
    VA = np.asarray(V, float)
    while heap:
        d, v = heapq.heappop(heap)
        if d > best[v] + 0.0:
            continue                            # 陈旧堆项
        if d >= base:
            break                               # 堆序剩余 ≥ base → 全夹到 base
        for u in adj.get(v, ()):                # 去重边列表
            nd = d + g * float(np.linalg.norm(VA[u] - VA[v]))
            if nd < best[u] and nd < base:
                best[u] = nd
                heapq.heappush(heap, (nd, u))
    if size_min is not None or size_max is not None:
        lo = 0.0 if size_min is None else float(size_min)
        hi = float("inf") if size_max is None else float(size_max)
        best = np.clip(best, lo, hi)
    return best


def remesh_with_controls(vertices, faces, base_size, surface_sizes=None,
                         vertex_sizes=None, region_sizes=None, growth=0.3,
                         size_min=None, size_max=None, **remesh_kwargs):
    """自定义尺寸控制的表面重网格化：解析尺寸场 → remesh_surface(sizes=…)。

    remesh_kwargs 透传 mesh_remesh.remesh_surface（refine/collapse/smooth/
    flip/max_iter 等）。返回 (V', F')。
    """
    from mesh_remesh import remesh_surface
    V = np.asarray(vertices, float)
    sizes = resolve_size_field(V, faces, base_size, surface_sizes,
                               vertex_sizes, region_sizes, growth,
                               size_min, size_max)
    return remesh_surface([list(map(float, p)) for p in V], faces,
                          target_max=base_size, curvature_adaptive=False,
                          sizes=sizes, **remesh_kwargs)
