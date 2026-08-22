# -*- coding: utf-8 -*-
"""star_gui_model.py — StarSceneModel：SimFile 的只读 GUI 适配层（纯逻辑，无 Qt）。

职责：
- 仿真树节点生成（对象图 build_tree + 语义过滤；U1 起另提供 STAR-CCM+ 用户树 sim_tree）
- 属性表数据生成（SimObject.dict + 引用解析 + 别名）
- 场景/显示器摘要（供 M3 使用）
"""

import re
from collections import defaultdict

from semantic_dict import layer_of, resolve_class, LAYER_CN


class Node:
    """树节点（不可变数据）。"""

    __slots__ = ("key", "label", "obj_id", "class_name", "layer", "children")

    def __init__(self, key, label, obj_id=None, class_name=None, layer=None,
                 children=None):
        self.key = key
        self.label = label
        self.obj_id = obj_id
        self.class_name = class_name
        self.layer = layer or layer_of(class_name or "")
        self.children = children or []


# 无名字且无意义展开的叶子容器类（树中隐藏）
_SKIP_LEAF_CLASSES = {
    "NameManager", "InactiveList", "Domain", "MethodVector",
    "ClassVersions", "PrintedMonitors", "SerializableTimeStamper",
}
_CONTAINER_CLASSES = {
    "InactiveList", "MethodVector", "SerializableList", "SerializableVector",
    "MasterArray", "ExportFunctionVector", "ExportPartVector",
    "ExportBoundaryVector", "ExportRegionVector", "AutoExportFunctionVector",
}


def short_class(cn):
    """类名 → 简短显示名（去掉 star.common 等包前缀）。"""
    if not cn:
        return "?"
    parts = cn.split(".")
    return parts[-1]


def friendly_name(obj):
    """STAR-CCM+ 风格显示名：PresentationName 优先，否则驼峰拆词。"""
    if obj is None:
        return "?"
    if obj.name:
        return obj.name
    short = short_class(obj.class_name)
    for suf in ("Solver", "Model", "Function", "Monitor", "Displayer", "Option"):
        if short.endswith(suf) and len(short) > len(suf):
            short = short[: -len(suf)]
            break
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", short).strip()
    return spaced or short


# STAR-CCM+ 用户树文件夹 → 语义层（图标）
_FOLDER_LAYER = {
    "Geometry": "cad-geometry",
    "Continua": "physics",
    "Regions": "core",
    "Solvers": "solver",
    "Reports": "post-processing",
    "Plots": "post-processing",
    "Monitors": "post-processing",
    "Scenes": "visualization",
    "Tools": "query",
}


def _is_container(cn):
    return any(cn.startswith(c) for c in _CONTAINER_CLASSES)


def _keep_node(obj, has_named_child):
    """树节点保留策略：有名字 / 有命名子节点 / 管理器容器。"""
    if obj.class_name in _SKIP_LEAF_CLASSES and not has_named_child:
        return False
    if obj.name is not None:
        return True
    if _is_container(obj.class_name) and not has_named_child:
        return False
    if obj.name is None and not has_named_child and not obj.class_name.endswith("Manager"):
        return False
    return True


class StarSceneModel:
    """把一个 SimFile 适配为 GUI 数据。"""

    def __init__(self, sim):
        self.sim = sim
        self.objmap = sim.objmap
        self.nodes_by_id = {}
        self._nodes_by_key = {}

    # ---------------- 树 ----------------
    def tree_roots(self):
        """仿真树根节点（Simulation + 游离对象组）。"""
        roots = []
        for obj in self.sim.roots:
            kids = self._node_children(obj)
            if not _keep_node(obj, bool(kids)):
                continue
            label = obj.name if obj.name is not None else short_class(obj.class_name)
            n = Node(self._key("obj", obj.id), label, obj.id, obj.class_name,
                     children=kids)
            self._register(n)
            roots.append(n)
        return roots

    def _node_children(self, obj, depth=0):
        if depth > 8:
            return []
        kids = []
        for c in self.sim.children.get(obj.id, []):
            ck = self._node_children(c, depth + 1)
            if not _keep_node(c, bool(ck)):
                continue
            label = c.name if c.name is not None else short_class(c.class_name)
            n = Node(self._key("obj", c.id), label, c.id, c.class_name,
                     children=ck)
            self._register(n)
            kids.append(n)
        kids.sort(key=lambda n: (n.obj_id or 0))
        return kids

    def _key(self, kind, oid):
        return "%s:%s" % (kind, oid)

    def _register(self, node):
        if node.obj_id is not None:
            self.nodes_by_id.setdefault(node.obj_id, node)
        self._nodes_by_key[node.key] = node

    # ---------------- STAR-CCM+ 用户树（U1） ----------------
    def _manager(self, *suffixes):
        """按类名短名精确匹配管理器（避免 SolverManager 误配 StoppingCriterionManager）。"""
        for suf in suffixes:
            for o in self.sim.objects:
                short = (o.class_name or "").split(".")[-1]
                if short == suf:
                    return o
        return None

    def _keys(self, manager):
        if manager is None:
            return []
        out = []
        for k in manager.dict.get("Keys") or []:
            o = self.objmap.get(k)
            if o is not None:
                out.append(o)
        return out

    def _obj_node(self, obj, children=None, label=None):
        n = Node(self._key("obj", obj.id),
                 label if label is not None else friendly_name(obj),
                 obj.id, obj.class_name, children=children or [])
        self._register(n)
        return n

    def _folder_node(self, key, label, children, layer=None):
        n = Node(self._key("folder", key), label, obj_id=None,
                 class_name="folder", layer=layer or _FOLDER_LAYER.get(key, "core"),
                 children=children)
        self._register(n)
        return n

    def _group_children(self, manager, child_builder=None):
        kids = []
        builder = child_builder or (lambda o: self._obj_node(o))
        for o in self._keys(manager):
            kids.append(builder(o))
        return kids

    def _part_node(self, obj):
        surfaces = []
        psm = self.objmap.get(obj.dict.get("PartSurfaces") or -1)
        if psm is None:
            psm = self.objmap.get(obj.dict.get("PartSurfaceManager") or -1)
        for s in self._keys(psm):
            if s.name:
                surfaces.append(self._obj_node(s))
        return self._obj_node(obj, children=surfaces)

    def _region_node(self, obj):
        bm = self.objmap.get(obj.dict.get("BoundaryManager") or -1)
        bounds = [self._obj_node(b) for b in self._keys(bm)]
        kids = []
        if bounds:
            kids.append(self._folder_node("Boundaries:%s" % obj.id, "Boundaries",
                                          bounds, layer="core"))
        return self._obj_node(obj, children=kids)

    def _continuum_node(self, obj):
        mm = self.objmap.get(obj.dict.get("ModelManager") or -1)
        models = [self._obj_node(m) for m in self._keys(mm)]
        return self._obj_node(obj, children=models)

    def _scene_node(self, obj):
        dm = self.objmap.get(obj.dict.get("DisplayerManager") or -1)
        disp = [self._obj_node(d) for d in self._keys(dm)
                if (d.class_name or "").startswith("star.vis")]
        return self._obj_node(obj, children=disp)

    def sim_tree(self):
        """STAR-CCM+ 用户仿真树（Geometry / Regions / Scenes / …）。

        数据源为各 *Manager.Keys，与官方客户端树及 semantic_report() 同源。
        原 tree_roots()（对象图森林）保留供调试。
        """
        sim_obj = None
        for o in self.sim.objects:
            if o.class_name == "star.common.Simulation":
                sim_obj = o
                break
        folders = [
            self._folder_node("Geometry", "Geometry",
                              self._group_children(
                                  self._manager("SimulationPartManager", "PartManager"),
                                  self._part_node)),
            self._folder_node("Continua", "Continua",
                              self._group_children(self._manager("ContinuumManager"),
                                                   self._continuum_node)),
            self._folder_node("Regions", "Regions",
                              self._group_children(self._manager("RegionManager"),
                                                   self._region_node)),
            self._folder_node("Solvers", "Solvers",
                              self._group_children(self._manager("SolverManager"))),
            self._folder_node("Reports", "Reports",
                              self._group_children(self._manager("ReportManager"))),
            self._folder_node("Plots", "Plots",
                              self._group_children(self._manager("PlotManager"))),
            self._folder_node("Monitors", "Monitors",
                              self._group_children(self._manager("MonitorManager"))),
            self._folder_node("Scenes", "Scenes",
                              self._group_children(self._manager("SceneManager"),
                                                   self._scene_node)),
            self._tools_folder(),
        ]
        if sim_obj is None:
            return folders
        root = self._obj_node(sim_obj, children=folders)
        return [root]

    def _tools_folder(self):
        kids = []
        ff = self._manager("FieldFunctionManager")
        if ff is not None:
            kids.append(self._folder_node("Field Functions", "Field Functions",
                                          self._group_children(ff), layer="query"))
        cs = self._manager("CoordinateSystemManager")
        if cs is not None:
            kids.append(self._folder_node("Coordinate Systems", "Coordinate Systems",
                                          self._group_children(cs), layer="query"))
        um = self._manager("UnitsManager")
        if um is not None:
            kids.append(self._obj_node(um, children=[], label="Units"))
        tm = None
        for o in self.sim.objects:
            if o.class_name == "star.common.TableManager":
                tm = o
                break
        if tm is not None:
            kids.append(self._obj_node(tm, children=self._group_children(tm),
                                       label="Tables"))
        return self._folder_node("Tools", "Tools", kids)

    def node_by_id(self, oid):
        return self.nodes_by_id.get(oid)

    def node_by_key(self, key):
        return self._nodes_by_key.get(key)

    def object_by_id(self, oid):
        return self.objmap.get(oid)

    # ---------------- 属性 ----------------
    @staticmethod
    def _fmt_value(v, objmap):
        if isinstance(v, dict):
            return "{...%d keys}" % len(v)
        if isinstance(v, list):
            return "[%d items]" % len(v)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return v if len(v) <= 120 else v[:117] + "..."
        if isinstance(v, int) and v in objmap:
            o = objmap[v]
            return "-> %s %s" % (short_class(o.class_name), o.name or "id:%d" % o.id)
        if isinstance(v, float):
            return "%.8g" % v
        return str(v)

    def properties(self, obj):
        """属性行列表：[(attr, value_text, raw_value)]。"""
        rows = []
        for k, v in obj.dict.items():
            rows.append((k, self._fmt_value(v, self.objmap), v))
        return rows

    # ---------------- 场景摘要（M3 使用） ----------------
    def scenes(self):
        out = []
        for o in self.sim.objects:
            if o.class_name == "star.vis.Scene":
                dm = self.objmap.get(o.dict.get("DisplayerManager") or -1)
                displayers = []
                if dm is not None:
                    for k in dm.dict.get("Keys") or []:
                        d = self.objmap.get(k)
                        if d is not None and (d.class_name or "").startswith("star.vis"):
                            displayers.append({"id": d.id, "class": d.class_name,
                                               "name": d.name})
                view = self.objmap.get(o.dict.get("CurrentView") or -1)
                out.append({"id": o.id, "name": o.name,
                            "displayers": displayers,
                            "view": (view.class_name, view.name) if view else None})
        return out

    # ---------------- 统计 ----------------
    def stats(self):
        census, _ = self.sim.layer_census()
        return dict(census.most_common())


def short_text(v):
    return StarSceneModel._fmt_value(v, {})
