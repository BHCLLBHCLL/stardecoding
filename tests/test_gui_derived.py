# -*- coding: utf-8 -*-
"""V 波遗留项：派生零件谱系（官方类型表 + 分类 + 真实语料发现 + 树/图标接线）。

依据 doc_javadoc_catalog.md §4（star.vis）、§5（star.post）、§8（star.meshing）。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

import star_gui_derived as dg  # noqa: E402

ADJ = os.path.join(ROOT, "adjointWing_start.sim")
AIRFOIL = os.path.join(ROOT, "resaved_airfoil.sim")
PIPE = os.path.join(ROOT, "resaved_vibratingPipe_start.sim")


class _Stub(object):
    def __init__(self, **kw):
        self.dict = {}
        for k, v in kw.items():
            setattr(self, k, v)


def _model(path):
    from sim_parser import SimFile
    from star_gui_model import StarSceneModel
    return StarSceneModel(SimFile(path))


# ---------------------------------------------------------------------------
# 类型表 / 分类
# ---------------------------------------------------------------------------
def test_type_table_grounded_in_javadoc():
    for short in ("ClipPlane", "PlaneManager", "PlaneSection", "IsoPart",
                  "ThresholdPart", "StreamlineCreator", "ScalarWarpSurface",
                  "PartDataSource", "FvRecordedPart", "ExtractedPart"):
        assert short in dg.DERIVED_TYPES
        t = dg.DERIVED_TYPES[short]
        assert t.short == short and t.cn and t.package.startswith("star.")
    assert len(dg.DERIVED_TYPES) >= 20


def test_classify_positive_and_negative():
    assert dg.is_derived("star.vis.ClipPlane")
    assert dg.is_derived("star.vis.ThresholdPart")
    assert dg.is_derived("star.post.FvRecordedSurface")
    assert dg.is_derived("star.meshing.ExtractedPart")
    assert not dg.is_derived("star.common.Region")
    assert not dg.is_derived("star.vis.Scene")
    assert not dg.is_derived("")
    assert not dg.is_derived(None)


def test_excluded_classes_not_derived():
    for cn in ("star.meshing.FaceQualityThreshold",
               "star.meshing.FaceProximityThreshold",
               "star.meshing.FreeEdgesThreshold",
               "star.meshing.NonManifoldEdgesThreshold",
               "star.meshing.NonManifoldVerticesThreshold",
               "star.meshing.PiercedFacesThreshold",
               "star.meshing.SurfaceMeshWidgetThresholdManager",
               "star.cadmodeler.CanonicalSketchPlane"):
        assert dg.short_class(cn) in dg._EXCLUDED
        assert dg.classify(cn) is None
        assert not dg.is_derived(cn)
        assert not dg.is_tree_member(cn)


def test_tree_member_kinds():
    assert dg.is_tree_member("star.vis.ClipPlane")          # part
    assert dg.is_tree_member("star.vis.PartDataSource")     # source
    assert not dg.is_tree_member("star.vis.PlaneManager")   # manager
    assert not dg.is_tree_member("star.vis.IsoCreator")     # creator
    assert not dg.is_tree_member("star.vis.IsoValue")       # value


def test_cn_labels():
    assert dg.type_cn("star.vis.ClipPlane") == "切片平面"
    assert dg.category_of("star.vis.IsoPart") == "iso"
    assert dg.category_cn("iso") == "等值面"
    assert dg.category_cn("clip") == "切面"
    assert dg.category_cn("unknown-cat") == "unknown-cat"
    assert dg.type_cn("star.common.Region") is None
    assert dg.category_of("star.common.Region") is None


# ---------------------------------------------------------------------------
# 真实语料谱系发现（.sim 确实不含 DerivedPartManager，靠 ClipPlane/PlaneManager 谱系）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path,count", [(ADJ, 6), (AIRFOIL, 12), (PIPE, 6)])
def test_derived_members_real_corpus(path, count):
    from sim_parser import SimFile
    sim = SimFile(path)
    members = dg.derived_members(sim)
    clips = [o for o in members if dg.short_class(o.class_name) == "ClipPlane"]
    assert len(clips) == count
    assert len(members) == count          # 无阈值/草图基准面误入
    s = dg.summary(sim)
    assert s["total"] == count
    assert s["categories"] == {"clip": count}
    assert s["types"] == ["ClipPlane"]
    assert all(dg.is_tree_member(o.class_name) for o in members)


def test_lineage_parent_and_scene():
    from sim_parser import SimFile
    sim = SimFile(ADJ)
    clip = [o for o in dg.derived_members(sim)
            if dg.short_class(o.class_name) == "ClipPlane"][0]
    info = dg.lineage(sim, clip)
    assert info["id"] == clip.id
    assert info["category"] == "clip"
    assert info["cn"] == "切片平面"
    assert info["parent"] is not None
    assert dg.short_class(info["parent"]["class_name"]) == "PlaneManager"
    assert info["scene"] is not None
    assert dg.short_class(info["scene"]["class_name"]) == "Scene"


# ---------------------------------------------------------------------------
# 树接线（StarSceneModel._derived_folder）
# ---------------------------------------------------------------------------
def test_tree_derived_folder_surfaces_clip_planes():
    m = _model(ADJ)
    root = m.sim_tree()[0]
    folder = [c for c in root.children if c.label == "Derived Parts"]
    assert folder, [c.label for c in root.children]
    folder = folder[0]
    assert folder.layer == "derived"
    labels = [c.label for c in folder.children]
    assert labels, "派生零件文件夹为空（谱系未显现）"
    assert all(l.startswith("Plane ") for l in labels)
    assert all(c.layer == "derived" for c in folder.children)
    assert all("ClipPlane" in (c.class_name or "") for c in folder.children)
    assert all(m.object_by_id(c.obj_id) is not None for c in folder.children)


def test_tree_derived_multi_category_subfolders(monkeypatch):
    def fake_members(sim):
        return [_Stub(id=1, class_name="star.vis.ClipPlane", name="Plane A"),
                _Stub(id=2, class_name="star.vis.IsoPart", name="Iso 1")]
    monkeypatch.setattr(dg, "derived_members", fake_members)
    m = _model(ADJ)
    root = m.sim_tree()[0]
    folder = [c for c in root.children if c.label == "Derived Parts"][0]
    subs = {c.label: c for c in folder.children}
    assert set(subs) == {"切面", "等值面"}
    assert subs["切面"].layer == "derived"
    assert [c.label for c in subs["切面"].children] == ["Plane A"]
    assert [c.label for c in subs["等值面"].children] == ["Iso 1"]


# ---------------------------------------------------------------------------
# 图标接线
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_derived_icon_and_key(app):
    from star_gui_icons import AppIcons
    from star_gui_panes import SimulationTree
    from star_gui_model import Node
    icons = AppIcons()
    assert icons._has("derived")
    assert not icons.get("derived").pixmap(24, 24).isNull()
    tree = SimulationTree()
    folder_node = Node("folder:Derived Parts", "Derived Parts", None, "folder",
                       layer="derived")
    assert tree._icon_key(folder_node) == "derived"
    clip_node = Node("obj:1", "Plane 1", 1, "star.vis.ClipPlane", layer="derived")
    assert tree._icon_key(clip_node) == "derived"
