# -*- coding: utf-8 -*-
"""C 波 C5：表面包裹 wrapper（收缩包裹 + 特征捕捉 + 局部尺寸）。

occ_wrap 为纯 numpy 自研，**两环境（默认 3.14 / conda occ）皆真跑**：
    pytest tests/test_occ_wrap.py -q
    <conda>/envs/occ/python.exe -m pytest tests/test_occ_wrap.py -q
核心验收：收缩包裹对含孔开放网格仍输出**水密**闭合表面；wrap_mesh 局部
加密（cell/2 / cell/4）且混合分辨率仍水密；detect_features 依二面角阈值
捕捉 sharp 特征边。
"""
import numpy as np

from occ_wrap import shrink_wrap, wrap_mesh, detect_features
from mesh_io import cube_mesh


def _open_box():
    """立方体挖掉底面 → 4 边方孔的开盒（输入本身不闭合）。"""
    V, F = cube_mesh(2.0)
    return V, F[2:]


def _tube():
    """开口筒：侧面 4 条竖缝的开放网格（一个「应该被包裹封住」的病态输入）。"""
    th = np.arange(0, 2 * np.pi, np.pi / 4)
    V = []
    for z in (0.0, 2.0):
        for a in th:
            V.append([np.cos(a) * 1.0, np.sin(a) * 1.0, z])
    F = []
    n = len(th)
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, j + n])
        F.append([i, j + n, i + n])
    return np.array(V, float), np.array(F, int)


def test_c5_shrink_wrap_seals_open_box():
    """收缩包裹：对开底方孔(4 边)的开盒输入 → 输出闭合水密表面。"""
    V, F = _open_box()
    assert len(shrink_wrap(V, F, 0.5, gap=0.0)["faces"]) > 0
    out = shrink_wrap(V, F, 0.40)
    assert out["watertight"], "收缩包裹必须输出水密闭合表面"
    assert out["n_triangles"] > 0
    assert out["volume_estimate"] > 0
    assert out["cell_sizes"] == [round(0.40, 9)]


def test_c5_shrink_wrap_seals_gappy_tube():
    """收缩包裹：对含竖缝的开口筒同样密封成闭合体（体积>0、水密）。"""
    V, F = _tube()
    out = shrink_wrap(V, F, 0.7)
    assert out["watertight"]
    assert out["volume_estimate"] > 0


def test_c5_wrap_mesh_local_refinement():
    """wrap_mesh：refine 球区局部加密 → 面密度更高，且整体仍水密。"""
    V, F = _open_box()
    base = shrink_wrap(V, F, 0.6)
    adapt = wrap_mesh(V, F, 0.6, refine=[((1.0, 1.0, 1.0), 0.6)])
    # 局部加密：refine 区三角被切割 → 面数多于均匀包裹
    assert adapt["n_triangles"] > base["n_triangles"]
    assert adapt["watertight"]


def test_c5_wrap_mesh_feature_capture():
    """wrap_mesh：features 特征点附近加密（面密度更高），仍水密。"""
    V, F = _tube()
    base = shrink_wrap(V, F, 0.8)
    out = wrap_mesh(V, F, 0.8, features=[(1.0, 0.0, 1.0)])
    assert out["n_triangles"] > base["n_triangles"], \
        "特征点附近应加密（局部面数增加）"
    assert out["watertight"]


def test_c5_detect_features_angle_threshold():
    """特征检测：立方体有 sharp 褶皱；阈值越大召回越少。"""
    V, F = cube_mesh(2.0)
    lo = detect_features(V, F, angle_deg=10.0)["n_feature"]
    hi = detect_features(V, F, angle_deg=80.0)["n_feature"]
    assert lo > 0            # 立方体存在直角褶皱
    assert hi <= lo          # 阈值提高 → 不再判定为特征


def test_c5_detect_features_returns_edges():
    """特征检测返回边三元组 (a,b,dihedral)，且角度在 (0,180)。"""
    V, F = cube_mesh(2.0)
    r = detect_features(V, F, angle_deg=30.0)
    assert "edges" in r and isinstance(r["edges"], list)
    for (a, b, dihed) in r["edges"]:
        assert isinstance(a, int) and isinstance(b, int)
        assert 0 < dihed <= 180
    assert r["n_feature"] == len(r["edges"])


def test_c5_watertight_no_degenerate_duplicate():
    """收缩包裹产物：无退化三角，文件可导出（STL 往返面数一致）。"""
    import os
    import tempfile
    from mesh_io import write_ascii_stl, read_stl
    from occ_repair import quality_metrics
    V, F = _open_box()
    out = shrink_wrap(V, F, 0.5)
    q = quality_metrics(out["vertices"], out["faces"])
    assert q["n_degenerate"] == 0
    assert q["closed"]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c5.stl")
        write_ascii_stl(p, out["vertices"], out["faces"])
        V2, F2 = read_stl(p)
        assert len(F2) == len(out["faces"])  # STL 往返面数一致