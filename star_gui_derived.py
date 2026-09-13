# -*- coding: utf-8 -*-
"""star_gui_derived.py — 派生零件（Derived Parts）类型表与谱系发现（纯逻辑，无 Qt/无 numpy）。

依据 doc_javadoc_catalog.md：
  §4 star.vis  L125-127 —— ClipPlane / PlaneManager、PlaneSection、
      IsoPart / IsoCreator / IsoValue、ThresholdPart / ThresholdCreator、
      StreamlineCreator、ScalarWarpSurface、
      PartDataSource / BoundaryDataSource / ThresholdDataSource / IsoDataSource /
      GridDataDataSource
  §5 star.post L136 —— FvRecordedObject 系列
      （FvRecordedPart / FvRecordedSurface / FvRecordedVolume /
        FvRecordedPointCloud / FvRecordedParcelCloud）
  §8 star.meshing L154 —— ExtractedPart

分类采用「短类名精确匹配」，避免把 star.meshing.*Threshold（面质量/接近度等阈值）
与 star.cadmodeler.CanonicalSketchPlane（草图基准面）误判为派生零件。
"""


class DerivedType(object):
    """一种官方派生零件类型。kind ∈ {part, source, creator, manager, value}。"""

    __slots__ = ("short", "package", "category", "cn", "kind")

    def __init__(self, short, package, category, cn, kind):
        self.short = short
        self.package = package
        self.category = category
        self.cn = cn
        self.kind = kind

    def __repr__(self):
        return "DerivedType(%s, %s, %s)" % (self.short, self.category, self.kind)


_DEFS = (
    # ---- star.vis 切面 / 等值面 / 阈值（§4 L125） ----
    ("ClipPlane", "star.vis", "clip", "切片平面", "part"),
    ("PlaneManager", "star.vis", "clip", "切片管理器", "manager"),
    ("PlaneSection", "star.vis", "section", "剖面", "part"),
    ("IsoPart", "star.vis", "iso", "等值面", "part"),
    ("IsoCreator", "star.vis", "iso", "等值面创建器", "creator"),
    ("IsoValue", "star.vis", "iso", "等值", "value"),
    ("ThresholdPart", "star.vis", "threshold", "阈值零件", "part"),
    ("ThresholdCreator", "star.vis", "threshold", "阈值创建器", "creator"),
    # ---- star.vis 流线 / 变形面（§4 L126） ----
    ("StreamlineCreator", "star.vis", "streamline", "流线创建器", "creator"),
    ("ScalarWarpSurface", "star.vis", "warp", "标量变形曲面", "part"),
    # ---- star.vis 场景数据源（§4 L127） ----
    ("PartDataSource", "star.vis", "data-source", "零件数据源", "source"),
    ("BoundaryDataSource", "star.vis", "data-source", "边界数据源", "source"),
    ("ThresholdDataSource", "star.vis", "data-source", "阈值数据源", "source"),
    ("IsoDataSource", "star.vis", "data-source", "等值数据源", "source"),
    ("GridDataDataSource", "star.vis", "data-source", "网格数据源", "source"),
    # ---- 旧启发式保留：探针/基类派生零件 ----
    ("ProbePart", "star.vis", "probe", "探针零件", "part"),
    ("DerivedPart", "star.vis", "derived", "派生零件", "part"),
    # ---- star.post 记录数据对象（§5 L136） ----
    ("FvRecordedPart", "star.post", "recorded", "记录零件", "part"),
    ("FvRecordedSurface", "star.post", "recorded", "记录表面", "part"),
    ("FvRecordedVolume", "star.post", "recorded", "记录体积", "part"),
    ("FvRecordedPointCloud", "star.post", "recorded", "记录点云", "part"),
    ("FvRecordedParcelCloud", "star.post", "recorded", "记录颗粒云", "part"),
    # ---- star.meshing 抽取部件（§8 L154） ----
    ("ExtractedPart", "star.meshing", "extracted", "抽取零件", "part"),
)

DERIVED_TYPES = {}
for _short, _pkg, _cat, _cn, _kind in _DEFS:
    DERIVED_TYPES[_short] = DerivedType(_short, _pkg, _cat, _cn, _kind)

_TREE_KINDS = ("part", "source")

_EXCLUDED = {
    "CanonicalSketchPlane",
    "FaceQualityThreshold",
    "FaceProximityThreshold",
    "FreeEdgesThreshold",
    "NonManifoldEdgesThreshold",
    "NonManifoldVerticesThreshold",
    "PiercedFacesThreshold",
    "SurfaceMeshWidgetThresholdManager",
}


def short_class(class_name):
    return (class_name or "").split(".")[-1]


def classify(class_name):
    """ClassName -> DerivedType；非派生零件返回 None（短类名精确匹配）。"""
    short = short_class(class_name)
    if not short or short in _EXCLUDED:
        return None
    return DERIVED_TYPES.get(short)


def is_derived(class_name):
    return classify(class_name) is not None


def is_tree_member(class_name):
    """是否应作为「派生零件」树节点出现（零件/数据源；排除创建器/管理器/枚举值）。"""
    t = classify(class_name)
    return t is not None and t.kind in _TREE_KINDS


def type_cn(class_name):
    t = classify(class_name)
    return t.cn if t is not None else None


def category_of(class_name):
    t = classify(class_name)
    return t.category if t is not None else None


CATEGORY_CN = {
    "clip": "切面",
    "section": "剖面",
    "iso": "等值面",
    "threshold": "阈值",
    "streamline": "流线",
    "warp": "变形面",
    "data-source": "数据源",
    "probe": "探针",
    "derived": "派生",
    "recorded": "记录",
    "extracted": "抽取",
}


def category_cn(category):
    return CATEGORY_CN.get(category, category)


def _manager(sim, short):
    for o in sim.objects:
        if short_class(o.class_name) == short:
            return o
    return None


def _resolve(sim, oid):
    if not isinstance(oid, int) or oid < 0:
        return None
    return sim.objmap.get(oid)


def derived_members(sim):
    """派生零件谱系成员（DerivedPartManager.Keys ∪ 全图精确类型命中），按 id 去重。"""
    out = []
    seen = set()
    mgr = _manager(sim, "DerivedPartManager")
    if mgr is not None:
        for oid in (mgr.dict.get("Keys") or []):
            o = _resolve(sim, oid)
            if o is not None and o.id not in seen:
                seen.add(o.id)
                out.append(o)
    for o in sim.objects:
        if o.id in seen or not is_tree_member(o.class_name):
            continue
        seen.add(o.id)
        out.append(o)
    return out


def lineage(sim, obj):
    """派生零件谱系：自身类型 + 父管理器 + 所属场景（ClipPlane 经 PlaneManager.Scene）。"""
    t = classify(obj.class_name)
    info = {
        "id": obj.id,
        "class_name": obj.class_name,
        "category": t.category if t is not None else None,
        "cn": t.cn if t is not None else None,
        "parent": None,
        "scene": None,
    }
    parent = _resolve(sim, obj.dict.get("Parent"))
    if parent is not None:
        info["parent"] = {"id": parent.id, "class_name": parent.class_name,
                          "name": parent.name}
    scene = _resolve(sim, obj.dict.get("Scene"))
    if scene is None and parent is not None:
        scene = _resolve(sim, parent.dict.get("Scene"))
    if scene is not None:
        info["scene"] = {"id": scene.id, "class_name": scene.class_name,
                         "name": scene.name}
    return info


def summary(sim):
    """谱系统计：总数 / 按类别 / 命中的类型短名（供锚点与自检）。"""
    members = derived_members(sim)
    cats = {}
    types = []
    for o in members:
        t = classify(o.class_name)
        cats[t.category] = cats.get(t.category, 0) + 1
        if t.short not in types:
            types.append(t.short)
    return {"total": len(members), "categories": cats, "types": types}
