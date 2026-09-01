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
    "Operations": "meshing",
    "Derived Parts": "visualization",
    "3D-CAD": "cad-geometry",
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
        self._g7_cache = None

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
        disp = [self._displayer_node(d) for d in self._keys(dm)
                if (d.class_name or "").startswith("star.vis")]
        return self._obj_node(obj, children=disp)

    def _displayer_node(self, obj):
        """PartDisplayer：下级 Parts 筛选列表（对齐 STAR-CCM+ 场景树）。"""
        col = self.objmap.get(obj.dict.get("Collector") or -1)
        kids = []
        if col is not None and (col.dict.get("Keys") or []):
            kids.append(self._parts_filter_node(col))
        return self._obj_node(obj, children=kids)

    def _parts_filter_node(self, part_group):
        """Collector.Keys 按所属几何 Part / Region 分组。"""
        grouped = []  # (parent_obj, [child_obj, ...])
        seen_parent = {}
        loose = []
        for src in self._keys(part_group):
            parent = owning_mesh_part(self.sim, src)
            if parent is None:
                loose.append(src)
                continue
            if parent.id not in seen_parent:
                seen_parent[parent.id] = len(grouped)
                grouped.append((parent, []))
            # 自身就是网格 Part 时不再嵌一层同名
            if src.id != parent.id:
                grouped[seen_parent[parent.id]][1].append(src)
        kids = []
        for parent, children in grouped:
            kids.append(self._obj_node(parent, children=[self._obj_node(c) for c in children]))
        for src in loose:
            kids.append(self._obj_node(src))
        return self._obj_node(part_group, children=kids, label="Parts")

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
            self._folder_node("Operations", "Operations",
                              self._group_children(
                                  self._manager("MeshOperationManager")),
                              layer="meshing"),
            self._derived_folder(),
            self._cad_folder(),
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

    def _derived_folder(self):
        mgr = self._manager("DerivedPartManager")
        if mgr is not None:
            return self._folder_node("Derived Parts", "Derived Parts",
                                     self._group_children(mgr), layer="visualization")
        kids = []
        seen = set()
        for o in self.sim.objects:
            cn = o.class_name or ""
            if "Manager" in cn:
                continue
            if any(tag in cn for tag in ("PlaneSection", "ThresholdPart", "IsoPart",
                                         "DerivedPart", "ProbePart")):
                if o.id not in seen:
                    seen.add(o.id)
                    kids.append(self._obj_node(o))
        return self._folder_node("Derived Parts", "Derived Parts", kids,
                                 layer="visualization")

    def _cad_folder(self):
        kids = []
        mgr = self._manager("CadModelManager", "CadObjectManager")
        if mgr is not None:
            kids.extend(self._group_children(mgr))
        if not kids:
            for o in self.sim.objects:
                cn = o.class_name or ""
                if cn.endswith("CadModel") or cn == "star.cadmodeler.CadModel":
                    kids.append(self._obj_node(o))
        return self._folder_node("3D-CAD", "3D-CAD", kids, layer="cad-geometry")

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
            if v and all(isinstance(x, int) for x in v) and objmap:
                names = []
                for x in v[:8]:
                    o = objmap.get(x)
                    names.append((o.name or short_class(o.class_name)) if o else str(x))
                extra = ", …" if len(v) > 8 else ""
                return "[" + ", ".join(names) + extra + "]"
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
        rows.extend(self._g7_rows(obj))
        return rows

    # ---------------- G7 物理语义行 ----------------
    def _g7(self):
        """extract_physics() 惰性缓存（G7）。"""
        if self._g7_cache is None:
            try:
                ph = self.sim.extract_physics()
            except Exception:
                ph = {"ok": False}
            if not ph.get("ok"):
                ph = {"continua": [], "materials": [], "motion": []}
            self._g7_cache = ph
        return self._g7_cache

    def _g7_rows(self, obj):
        """G7 语义行：连续体模型参数 / 材料属性 / 运动规格。

        行名统一加 "G7:" 前缀。P1 写侧：可编辑叶子（物理量/选项/标量参数）
        单独成行，raw 携带编辑描述符 {"kind","oid","key",...}，属性面板据此
        开放编辑并路由到目标对象；无描述符的行保持只读。
        """
        from sim_parser import g7_format_value
        cn = obj.class_name or ""
        tail = cn.rsplit(".", 1)[-1]
        rows = []
        if cn == "star.common.PhysicsContinuum":
            c = next((c for c in self._g7()["continua"]
                      if c["id"] == obj.id), None)
            if c:
                for m in c["models"]:
                    ps = " ".join("%s=%s" % (k, g7_format_value(v))
                                  for k, v in m["params"].items())
                    label = (m["class"] or "").rsplit(".", 1)[-1]
                    if m["name"]:
                        label += " %r" % m["name"]
                    rows.append(("G7:模型 " + label,
                                 ps if ps else "（无参数）", None))
                    rows.extend(self._g7_edit_rows(m["params"], m["id"]))
        elif cn.startswith("star.material.") and \
                tail in type(self.sim)._G7_MAT_TAILS:
            mt = next((m for m in self._g7()["materials"]
                       if m["id"] == obj.id), None)
            if mt:
                for p in mt["properties"]:
                    line = "[%s" % p.get("method", "?")
                    tag = p.get("method_tag", "")
                    if tag:
                        line += "/%s" % tag
                    line += "]"
                    if "value" in p:
                        line += " = %s" % g7_format_value(
                            {"value": p["value"], "units": p.get("units", "")})
                    rows.append(("G7:属性 " + p["name"], line, None))
                    if p.get("kind") == "quantity" and "value" in p:
                        rows.append(("G7:值 " + p["name"],
                                     g7_format_value(p), p))
        elif cn == "star.motion.MotionSpecification":
            mo = next((m for m in self._g7()["motion"]
                       if m["id"] == obj.id), None)
            if mo:
                rows.append(("G7:运动 Region", mo.get("region", "") or "—", None))
                rows.append(("G7:运动 Continuum",
                             mo.get("continuum", "") or "—", None))
                mc = mo.get("motion_class", "") or ""
                mv = mc.rsplit(".", 1)[-1] if mc else "—"
                if mo.get("motion_name"):
                    mv += " %r" % mo["motion_name"]
                rows.append(("G7:运动 Motion", mv, None))
                rc = mo.get("ref_frame_class", "") or ""
                rv = rc.rsplit(".", 1)[-1] if rc else "—"
                if mo.get("ref_frame_name"):
                    rv += " %r" % mo["ref_frame_name"]
                rows.append(("G7:运动 ReferenceFrame", rv, None))
                for k in ("RotationRate", "AxisVector", "OriginVector"):
                    q = mo.get(k)
                    if isinstance(q, dict) and "oid" in q:
                        rows.append(("G7:" + k, g7_format_value(q), q))
        return rows

    def _g7_edit_rows(self, params, holder_oid, prefix=""):
        """P1 编辑行：把参数叶子展开为带编辑描述符的 G7 行。

        物理量/选项 dict 自带 oid/key/kind 锚点；原始标量（bool/int/float）
        的锚点是直接持有者（holder_oid）；嵌套参数组按 _oid 递归。
        """
        from sim_parser import g7_format_value
        rows = []
        for k, v in params.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and "kind" in v and "oid" in v:
                if "value" in v or "selected" in v:
                    rows.append(("G7:%s%s" % (prefix, k),
                                 g7_format_value(v), v))
            elif isinstance(v, dict):
                rows.extend(self._g7_edit_rows(
                    v, v.get("_oid", holder_oid), prefix + k + "."))
            elif isinstance(v, (bool, int, float)):
                rows.append(("G7:%s%s" % (prefix, k), g7_format_value(v),
                             {"kind": "scalar", "oid": holder_oid,
                              "key": k, "value": v}))
        return rows

    def invalidate_g7(self):
        """P1：物理参数编辑后失效 G7 语义缓存（下次访问重建）。"""
        self._g7_cache = None

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


def _psm_to_part_map(sim):
    """PartSurfaceManager id → 拥有它的几何 Part（MeshPart / CadPart / Block）。"""
    cache = getattr(sim, "_psm_to_part", None)
    if cache is not None:
        return cache
    cache = {}
    for o in sim.objects:
        cn = o.class_name or ""
        if "Part" not in cn and not isinstance(o.dict.get("TriangleCount"), int):
            continue
        for attr in ("PartSurfaces", "PartSurfaceManager"):
            v = o.dict.get(attr)
            if isinstance(v, int):
                cache[v] = o
    sim._psm_to_part = cache
    return cache


def owning_mesh_part(sim, obj):
    """把 Displayer Parts 里的 Boundary / PartSurface / Part 解析到带网格的几何 Part。"""
    if obj is None:
        return None
    om = sim.objmap
    cn = obj.class_name or ""
    if isinstance(obj.dict.get("TriangleCount"), int) and obj.dict["TriangleCount"] > 0:
        return obj
    if "PartSurface" in cn and "Manager" not in cn and "Group" not in cn:
        psm_id = obj.dict.get("Parent")
        if isinstance(psm_id, int):
            return _psm_to_part_map(sim).get(psm_id)
        return None
    if cn.endswith("Boundary") or cn == "star.common.Boundary":
        psg = om.get(obj.dict.get("PartSurfaces") or -1)
        for k in (psg.dict.get("Keys") if psg is not None else None) or []:
            ps = om.get(k)
            part = owning_mesh_part(sim, ps)
            if part is not None:
                return part
        region = om.get(obj.dict.get("Region") or -1)
        pg = om.get(region.dict.get("Parts") or -1) if region is not None else None
        for k in (pg.dict.get("Keys") if pg is not None else None) or []:
            p = om.get(k)
            if p is not None and isinstance(p.dict.get("TriangleCount"), int) and p.dict["TriangleCount"] > 0:
                return p
        return None
    if cn.endswith("Region") or cn == "star.common.Region":
        pg = om.get(obj.dict.get("Parts") or -1)
        for k in (pg.dict.get("Keys") if pg is not None else None) or []:
            p = om.get(k)
            if p is not None and isinstance(p.dict.get("TriangleCount"), int) and p.dict["TriangleCount"] > 0:
                return p
    return None


def collector_sources(sim, displayer):
    """PartDisplayer.Collector.Keys → 源对象列表。"""
    if displayer is None:
        return []
    om = sim.objmap
    col = om.get(displayer.dict.get("Collector") or -1)
    if col is None:
        return []
    out = []
    for k in col.dict.get("Keys") or []:
        o = om.get(k)
        if o is not None:
            out.append(o)
    return out


def mesh_part_ids_for_displayer(sim, displayer):
    """该显示器 Parts 筛选对应的网格 Part id 集合。"""
    ids = []
    seen = set()
    for src in collector_sources(sim, displayer):
        part = owning_mesh_part(sim, src)
        if part is not None and part.id not in seen:
            seen.add(part.id)
            ids.append(part.id)
    return ids


def _surface_patch_types(sim):
    cache = getattr(sim, "_surface_patch_types", None)
    if cache is not None:
        return cache
    cache = {}
    for o in sim.objects:
        if o.class_name != "star.meshing.PartSurfacePatches":
            continue
        psid = o.dict.get("PartSurface")
        types = o.dict.get("Types") or []
        if not isinstance(psid, int) or not types:
            continue
        s = set(int(x) for x in types)
        if psid not in cache or len(s) > len(cache[psid]):
            cache[psid] = s
    sim._surface_patch_types = cache
    return cache


def part_surface_cad_ids(sim, obj):
    """Boundary / PartSurface → CAD FaceTypes 集合（用于从整 Part 网格里切面子块）。"""
    if obj is None:
        return set()
    om = sim.objmap
    cn = obj.class_name or ""
    surfaces = []
    if "PartSurface" in cn and "Manager" not in cn and "Group" not in cn:
        surfaces = [obj]
    elif cn.endswith("Boundary") or cn == "star.common.Boundary":
        psg = om.get(obj.dict.get("PartSurfaces") or -1)
        for k in (psg.dict.get("Keys") if psg is not None else None) or []:
            s = om.get(k)
            if s is not None:
                surfaces.append(s)
    patches = _surface_patch_types(sim)
    ids = set()
    for s in surfaces:
        ids |= patches.get(s.id, set())
    return ids
