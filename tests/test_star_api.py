# -*- coding: utf-8 -*-
"""A2：star.* Python 脚本 API —— 对象模型镜像 / 管理器 Keys / 脚本执行。

对象图取自教程件 adjointWing_start.sim（与 E0 测试同源）。
"""
import os

import pytest

import star_api as sa
from sim_parser import SimFile
from semantic_dict import layer_of, resolve_class

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "adjointWing_start.sim")


def _sim():
    return sa.Simulation(sim=SimFile(SIM))


# -- 集合 + 包装 -----------------------------------------------------------
def test_simulation_collections_and_wrap_types():
    S = _sim()
    assert [r.name for r in S.regions] == ["Fluid Domain"]
    assert isinstance(S.regions[0], sa.Region)
    assert [s.name for s in S.scenes] == ["Mesh Scene 1"]
    assert isinstance(S.scenes[0], sa.Scene)
    assert [c.name for c in S.continua] == ["Physics 1"]
    assert isinstance(S.continua[0], sa.Continuum)
    assert sorted(p.name for p in S.parts) == ["Fluid Domain", "Large Block",
                                               "Small Block"]
    assert [p.name for p in S.plots] == ["Residuals"]


def test_wrap_maps_semantic_layers_and_suffixes():
    sim = SimFile(SIM)
    S = sa.Simulation(sim=sim)
    phys = next(o for o in sim.objects if o.name == "Physics 1")
    assert resolve_class(phys.class_name) == "star.common.PhysicsContinuum"
    assert layer_of(phys.class_name) == "core"
    assert isinstance(sa.wrap(S, phys), sa.Continuum)

    region = next(o for o in sim.objects if o.name == "Fluid Domain")
    assert isinstance(sa.wrap(S, region), sa.Region)
    assert sa.wrap(S, None) is None


# -- 管理器 Keys + 反向访问器 ----------------------------------------------
def test_region_boundaries_via_manager_keys():
    S = _sim()
    region = S.get_by_name("Fluid Domain")
    assert isinstance(region, sa.Region)
    names = [b.name for b in region.boundaries]
    assert names == ["Wing Lower Element", "Symmetry", "Wing Upper Element",
                     "Inlet", "Far Field", "Outlet", "Wing End Plate"]
    assert all(b.class_name.endswith("Boundary") for b in region.boundaries)


def test_boundary_region_and_displayer_scene_reverse():
    S = _sim()
    inlet = S.get_by_name("Inlet")
    assert isinstance(inlet, sa.Boundary)
    assert inlet.region is not None
    assert inlet.region.name == "Fluid Domain"

    scene = S.get_by_name("Mesh Scene 1")
    assert isinstance(scene, sa.Scene)
    displayers = scene.displayers
    assert displayers
    assert all(d.class_name.endswith("Displayer") for d in displayers)
    assert displayers[0].scene.name == "Mesh Scene 1"


# -- 定位：Key / get / find_all -------------------------------------------
def test_client_server_object_key_resolution():
    sim = SimFile(SIM)
    S = sa.Simulation(sim=sim)
    by_name = sa.ClientServerObjectKey(name="Fluid Domain")
    assert by_name.resolve(sim.objects).id == S.get_by_name("Fluid Domain").id
    assert S.get(by_name).name == "Fluid Domain"

    by_class = sa.ClientServerObjectKey(class_name="star.common.Region")
    assert S.get(by_class).name == "Fluid Domain"
    assert sa.ClientServerObjectKey(name="Nope").resolve(sim.objects) is None
    assert S.get("Nope", "fallback") == "fallback"


def test_get_by_class_name_and_find_all_and_len():
    sim = SimFile(SIM)
    S = sa.Simulation(sim=sim)
    assert S.get("star.common.Region").name == "Fluid Domain"
    found = S.find_all(sa.ClientServerObjectKey(class_name="star.common.Region"))
    assert found and all(isinstance(o, sa.Region) for o in found)
    assert len(S) == len(sim.objects)
    assert S[sa.ClientServerObjectKey(name="Fluid Domain")].name == "Fluid Domain"
    region = S.get_by_name("Fluid Domain")
    assert S.get_object(region.id) == region


# -- 写操作（经 CommandBus，可撤销） --------------------------------------
def test_rename_and_set_property_undo_redo():
    S = _sim()
    region = S.get_by_name("Fluid Domain")
    old = region.get("PresentationName")

    region.rename("Renamed Domain")
    assert region.get("PresentationName") == "Renamed Domain"
    assert S.document.bus.can_undo()
    assert S.document.undo()
    assert region.get("PresentationName") == old
    assert S.document.redo()
    assert region.get("PresentationName") == "Renamed Domain"

    region.set("SomeCustomKey", 5)
    assert region.get("SomeCustomKey") == 5


def test_visibility_and_copy_delete_roundtrip():
    S = _sim()
    region = S.get_by_name("Fluid Domain")
    assert region.is_visible()
    region.set_visible(False)
    assert not region.is_visible()
    region.set_visible(True)
    assert region.is_visible()

    clone = region.copy()
    assert clone is not None and clone.id != region.id
    assert clone.name.endswith("Copy")


# -- 脚本执行入口 ----------------------------------------------------------
def test_run_python_script_injects_namespaces():
    g = sa.run_python_script(
        "n = len(sim.regions)\n"
        "first = sim.regions[0].name\n"
        "key = ClientServerObjectKey(name='Fluid Domain')\n"
        "resolved = sim.get(key).name\n"
        "SimCls = star.Simulation\n"
        "flow_name = sim.regions[0].boundaries[3].name\n",
        sim=SimFile(SIM))
    assert g["n"] == 1
    assert g["first"] == "Fluid Domain"
    assert g["resolved"] == "Fluid Domain"
    assert g["SimCls"] is sa.Simulation
    assert g["flow_name"] == "Inlet"


def test_star_namespace_exposes_classes_and_subpackages():
    ns = sa.make_star_namespace(_sim())
    assert ns.Simulation is sa.Simulation
    assert ns.Region is sa.Region
    assert ns.ClientServerObjectKey is sa.ClientServerObjectKey
    assert ns.common.Solver is sa.Solver
    assert ns.vis.Scene is sa.Scene
    assert ns.meshing.Part is sa.Part
    assert ns.base.neo is sa.ClientServerObject
    assert "Simulation" in repr(ns.common)
