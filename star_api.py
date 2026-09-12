# -*- coding: utf-8 -*-
"""A2：Python 脚本 API —— 镜像 star.* ClientServerObject 对象模型。

以 `semantic_dict`（包→语义层）+ `sim_parser.SimObject`（对象图）为骨架：
  - `ClientServerObjectKey`：官方同名键，按 (类名 / 语义层 / 名) 惰性定位对象。
  - `ClientServerObject`：包一个 SimObject + 所属 Simulation，提供
    get/set/rename/delete/copy/set_visible/parent/children/descendants 等
    与官方 ClientServerObject 对齐的访问面；所有写操作经 CommandBus（可撤销、
    可被 A1 录制器捕获）。
  - `Simulation`：根对象，提供 regions/scenes/parts/reports/... 集合与 get()。
  - `run_python_script`：脚本执行入口，注入 `star` / `sim` 命名空间。
"""

import copy as _copy

from semantic_dict import LAYER_CN, layer_of, resolve_class
from sim_parser import build_tree
from star_gui_commands import (CopyObjectCommand, DeleteObjectCommand,
                               RenameCommand, SetPropertyCommand,
                               VisibilityCommand)


class ClientServerObjectKey(object):
    """按 (类名 / 语义层 / 名) 定位对象的键（对应官方 ClientServerObjectKey）。"""

    def __init__(self, class_name=None, name=None, layer=None):
        self.class_name = class_name
        self.name = name
        self.layer = layer

    def matches(self, obj):
        if self.name is not None and (obj.name or "") != self.name:
            return False
        if self.class_name is not None:
            cls = resolve_class(obj.class_name or "")
            want = self.class_name
            if cls != want and not cls.endswith("." + want):
                return False
        if self.layer is not None and layer_of(obj.class_name or "") != self.layer:
            return False
        return True

    def resolve(self, objects):
        for o in objects:
            if self.matches(o):
                return o
        return None

    def __repr__(self):
        return "ClientServerObjectKey(class_name=%r, name=%r, layer=%r)" % (
            self.class_name, self.name, self.layer)


class ClientServerObject(object):
    """ClientServerObject 基类：SimObject 的脚本侧视图。"""

    layer = None

    def __init__(self, sim, obj):
        self._sim = sim
        self._obj = obj

    # -- 只读属性 ----------------------------------------------------------
    @property
    def id(self):
        return self._obj.id

    @property
    def class_name(self):
        return self._obj.class_name

    @property
    def resolved_class(self):
        return resolve_class(self._obj.class_name or "")

    @property
    def name(self):
        return self._obj.name

    @property
    def obj_layer(self):
        return layer_of(self._obj.class_name or "")

    @property
    def layer_cn(self):
        return LAYER_CN.get(self.obj_layer, self.obj_layer)

    # -- 读写 --------------------------------------------------------------
    def get(self, key, default=None):
        return self._obj.dict.get(key, default)

    def set(self, key, value):
        """改属性 → 经 CommandBus（可撤销 / 可录制）。"""
        return self._sim.set_property(self, key, value)

    def rename(self, new_name):
        return self._sim.rename(self, new_name)

    def delete(self):
        return self._sim.delete(self)

    def copy(self):
        return self._sim.copy(self)

    def set_visible(self, visible):
        return self._sim.set_visible(self, visible)

    def is_visible(self):
        return self._sim.is_visible(self)

    # -- 图关系 ------------------------------------------------------------
    def parent(self):
        pid = self._sim._tree()[2].get(self._obj.id)
        if pid is None:
            return None
        po = self._sim.sim.objmap.get(pid)
        return wrap(self._sim, po) if po is not None else None

    def children(self):
        kids = self._sim._tree()[1].get(self._obj.id, [])
        return [wrap(self._sim, o) for o in kids]

    def descendants(self):
        out = []
        stack = list(self.children())
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(node.children())
        return out

    def keys(self):
        """管理器 Keys 列表（按图顺序）并包装为对象。"""
        raw = self._obj.dict.get("Keys")
        if not isinstance(raw, list):
            return []
        out = []
        for ref in raw:
            o = self._sim.sim.objmap.get(ref) if self._sim.sim else None
            if o is not None:
                out.append(wrap(self._sim, o))
        return out

    def ref(self, key):
        """把某个属性值解析成被引用对象（int 引用）。"""
        v = self._obj.dict.get(key)
        if isinstance(v, int) and self._sim.sim is not None:
            o = self._sim.sim.objmap.get(v)
            if o is not None:
                return wrap(self._sim, o)
        return None

    def child(self, class_suffix):
        """按类名后缀找直接子对象（如对应管理器）。"""
        for c in self.children():
            if (c.class_name or "").endswith(class_suffix):
                return c
        return None

    def _manager_keys(self, manager_suffix):
        """走官方管理器路径取子项：XManager.Keys -> 子对象列表。

        找不到管理器返回 None（由调用方回退到图遍历）。
        """
        for c in self.children():
            if (c.class_name or "").endswith(manager_suffix):
                return c.keys()
        for c in self.descendants():
            if (c.class_name or "").endswith(manager_suffix):
                return c.keys()
        return None

    def __eq__(self, other):
        return isinstance(other, ClientServerObject) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return "<%s id=%d name=%r>" % (self.__class__.__name__, self.id, self.name)


# -- 语义层/类名 -> 专用子类 ------------------------------------------------
class Region(ClientServerObject):
    layer = "core"

    @property
    def boundaries(self):
        items = self._manager_keys("BoundaryManager")
        if items is not None:
            return items
        return [c for c in self.descendants()
                if (c.class_name or "").endswith("Boundary")
                and not (c.class_name or "").endswith("Manager")]


class Boundary(ClientServerObject):
    layer = "core"

    @property
    def region(self):
        for r in self._sim.regions:
            if any(b.id == self.id for b in r.boundaries):
                return r
        node = self.parent()
        while node is not None:
            if (node.class_name or "").endswith("Region"):
                return node
            node = node.parent()
        return None


class Part(ClientServerObject):
    layer = "meshing"


class Scene(ClientServerObject):
    layer = "visualization"

    @property
    def displayers(self):
        items = self._manager_keys("DisplayerManager")
        if items is not None:
            return items
        return [c for c in self.descendants() if "Displayer" in (c.class_name or "")
                and not (c.class_name or "").endswith("Manager")]


class Displayer(ClientServerObject):
    layer = "visualization"

    @property
    def scene(self):
        for s in self._sim.scenes:
            if any(d.id == self.id for d in s.displayers):
                return s
        node = self.parent()
        while node is not None:
            if (node.class_name or "").endswith("Scene"):
                return node
            node = node.parent()
        return None


class Report(ClientServerObject):
    layer = "post-processing"


class Monitor(ClientServerObject):
    layer = "post-processing"


class Plot(ClientServerObject):
    layer = "post-processing"


class Table(ClientServerObject):
    layer = "post-processing"


class FieldFunction(ClientServerObject):
    layer = "post-processing"


class Continuum(ClientServerObject):
    layer = "physics"

    @property
    def models(self):
        items = self._manager_keys("ModelManager")
        if items is not None:
            return items
        return [c for c in self.descendants() if "Model" in (c.class_name or "")
                and not (c.class_name or "").endswith("Manager")]


class Model(ClientServerObject):
    layer = "physics"

    @property
    def continuum(self):
        for c in self._sim.continua:
            if any(m.id == self.id for m in c.models):
                return c
        node = self.parent()
        while node is not None:
            if "Continuum" in (node.class_name or ""):
                return node
            node = node.parent()
        return None


class Solver(ClientServerObject):
    layer = "solver"


class Material(ClientServerObject):
    layer = "materials"


_BY_SIMPLE = {
    "Region": Region, "Boundary": Boundary, "Part": Part, "Scene": Scene,
    "Displayer": Displayer, "Report": Report, "Monitor": Monitor, "Plot": Plot,
    "Table": Table, "FieldFunction": FieldFunction, "Continuum": Continuum,
    "Model": Model, "Solver": Solver, "Material": Material,
}

_BY_LAYER = {
    "meshing": Part,
    "visualization": Scene,
    "post-processing": Report,
    "physics": Continuum,
    "solver": Solver,
    "materials": Material,
}


def wrap(sim, obj):
    """按类名/语义层把 SimObject 包装为对应的 ClientServerObject 子类。

    `star.common.PhysicsContinuum` 之类语义层归 core 的类，靠类名后缀匹配
    （`_BY_SIMPLE`）兜底，避免退回基类。
    """
    if obj is None:
        return None
    simple = resolve_class(obj.class_name or "").rsplit(".", 1)[-1]
    cls = _BY_SIMPLE.get(simple)
    if cls is None:
        for key in sorted(_BY_SIMPLE, key=len, reverse=True):
            if simple.endswith(key):
                cls = _BY_SIMPLE[key]
                break
    if cls is None:
        cls = _BY_LAYER.get(layer_of(obj.class_name or ""), ClientServerObject)
    return cls(sim, obj)


class Simulation(object):
    """根对象：包一层 SimDocument，脚本从这里访问一切。"""

    def __init__(self, doc=None, sim=None):
        if doc is None and sim is not None:
            from star_gui_document import SimDocument
            doc = SimDocument(sim)
        self.document = doc
        self.sim = doc.sim if doc is not None else None
        self._cache = None

    # -- 内部：树缓存 ------------------------------------------------------
    def _tree(self):
        if self._cache is None:
            objs = self.sim.objects if self.sim is not None else []
            roots, children = build_tree(objs)
            parent = {}
            for pid, kids in children.items():
                for k in kids:
                    parent[k.id] = pid
            self._cache = (roots, children, parent)
        return self._cache

    def refresh(self):
        self._cache = None

    # -- 集合 --------------------------------------------------------------
    def objects(self, layer=None, class_name=None):
        if self.sim is None:
            return []
        out = []
        for o in self.sim.objects:
            if class_name is not None and resolve_class(o.class_name or "") != class_name:
                continue
            if layer is not None and layer_of(o.class_name or "") != layer:
                continue
            out.append(wrap(self, o))
        return out

    def by_layer(self, layer):
        return self.objects(layer=layer)

    def roots(self):
        return [wrap(self, o) for o in self._tree()[0]]

    @property
    def regions(self):
        return [o for o in self.objects()
                if (o.class_name or "").endswith("Region") and o.name]

    @property
    def scenes(self):
        return [o for o in self.objects()
                if (o.class_name or "").endswith("Scene") and o.name]

    @property
    def parts(self):
        return [o for o in self.objects(layer="meshing")
                if (o.class_name or "").endswith("Part") and o.name]

    @property
    def reports(self):
        return [o for o in self.objects()
                if (o.class_name or "").endswith("Report") and o.name]

    @property
    def monitors(self):
        return [o for o in self.objects()
                if "Monitor" in (o.class_name or "") and o.name]

    @property
    def plots(self):
        return [o for o in self.objects()
                if (o.class_name or "").endswith("Plot") and o.name]

    @property
    def continua(self):
        return [o for o in self.objects()
                if "Continuum" in (o.class_name or "") and o.name]

    @property
    def models(self):
        return [o for o in self.objects()
                if "Model" in (o.class_name or "") and o.name]

    # -- 查找 --------------------------------------------------------------
    def get_object(self, oid):
        if self.sim is None or oid is None:
            return None
        return wrap(self, self.sim.objmap.get(oid))

    def get(self, key, default=None):
        """按 ClientServerObjectKey / 名 / 类名 取一个对象。"""
        if isinstance(key, ClientServerObjectKey):
            o = key.resolve(self.sim.objects if self.sim else [])
            return wrap(self, o) if o is not None else default
        if isinstance(key, int):
            got = self.get_object(key)
            return got if got is not None else default
        objs = self.objects()
        for o in objs:
            if o.name == key:
                return o
        for o in objs:
            if resolve_class(o.class_name or "") == key:
                return o
        return default

    def get_by_name(self, name, default=None):
        return self.get(ClientServerObjectKey(name=name), default)

    def find_all(self, key):
        if isinstance(key, ClientServerObjectKey):
            return [wrap(self, o) for o in (self.sim.objects if self.sim else [])
                    if key.matches(o)]
        return [o for o in self.objects() if o.name == key]

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return len(self.sim.objects) if self.sim is not None else 0

    def __iter__(self):
        return iter(self.objects())

    # -- 写操作（经 CommandBus） -------------------------------------------
    def set_property(self, cso, key, value):
        if self.document is None:
            cso._obj.dict[key] = value
            return True
        return bool(self.document.execute(
            SetPropertyCommand(cso.id, key, value, cso._obj.dict.get(key))))

    def rename(self, cso, new_name):
        if self.document is None:
            cso._obj.dict["PresentationName"] = new_name
            return True
        return bool(self.document.execute(
            RenameCommand(cso.id, new_name, cso._obj.dict.get("PresentationName"))))

    def delete(self, cso):
        if self.document is None:
            return False
        return bool(self.document.execute(DeleteObjectCommand(cso.id)))

    def copy(self, cso):
        if self.document is None:
            return None
        cmd = CopyObjectCommand(cso.id)
        if not self.document.execute(cmd):
            return None
        return self.get_object(cmd.new_id)

    def set_visible(self, cso, visible):
        if self.document is None:
            return False
        return bool(self.document.execute(VisibilityCommand(cso.id, visible)))

    def is_visible(self, cso):
        if self.document is not None:
            return self.document.is_visible(cso.id)
        return True

    def save(self, path):
        from sim_writer import save_sim
        if self.document is None:
            return False
        doc = self.document
        save_sim(self.sim, path, patches=doc.patches, created=doc.created,
                 src_path=doc.path, deleted=doc.deleted,
                 array_patches=doc.array_patches)
        return True


# ---------------------------------------------------------------------------
# 脚本命名空间 + 执行入口
# ---------------------------------------------------------------------------
class _SubNamespace(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return "<star.%s>" % ",".join(sorted(self.__dict__))


class StarNamespace(object):
    """镜像 `star.*`：顶层暴露常用 ClientServerObject 子类 + 子包。"""

    def __init__(self, simulation):
        self.simulation = simulation
        self.Simulation = Simulation
        self.ClientServerObject = ClientServerObject
        self.ClientServerObjectKey = ClientServerObjectKey
        for cls in (Region, Boundary, Part, Scene, Displayer, Report, Monitor,
                    Plot, Table, FieldFunction, Continuum, Model, Solver, Material):
            setattr(self, cls.__name__, cls)
        self.common = _SubNamespace(Simulation=Simulation, Region=Region,
                                    Boundary=Boundary, Solver=Solver)
        self.vis = _SubNamespace(Scene=Scene, Displayer=Displayer)
        self.meshing = _SubNamespace(Part=Part)
        self.base = _SubNamespace(ClientServerObject=ClientServerObject,
                                  ClientServerObjectKey=ClientServerObjectKey,
                                  neo=ClientServerObject)


def make_star_namespace(simulation):
    return StarNamespace(simulation)


def run_python_script(source, simulation=None, doc=None, sim=None, extra=None):
    """执行一段 Python 脚本，注入 `star` / `sim` / `objects`。

    返回脚本命名空间（globals）。脚本内可直接调用 `sim.regions`、
    `star.common.Simulation`、`ClientServerObjectKey(...)` 等。
    """
    if simulation is None:
        simulation = Simulation(doc=doc, sim=sim)
    star = make_star_namespace(simulation)
    g = {
        "__name__": "__star_script__",
        "__builtins__": __builtins__,
        "star": star,
        "sim": simulation,
        "objects": simulation.objects,
        "ClientServerObjectKey": ClientServerObjectKey,
    }
    if extra:
        g.update(extra)
    exec(compile(source, "<star-script>", "exec"), g)
    return g


def _deepcopy_dict(d):
    return _copy.deepcopy(d)
