# -*- coding: utf-8 -*-
"""
对象图语义字典：包→语义层映射、旧类名→新类名别名表、属性→引用方向规则。

依据:
  - doc_javadoc_catalog.md（574 个 star.* 包的语义目录，来自官方 Javadoc）
  - 21 个 .sim 语料实测（1286 种 ClassName，6.x–10.x 时代旧类名）
"""

# ---------------------------------------------------------------------------
# 1. 包前缀 -> 语义层
# ---------------------------------------------------------------------------
PACKAGE_LAYERS = {
    # 根/对象模型
    "star.base.neo": "object-model",
    "star.base.generic": "object-model",
    "star.base.query": "query",
    "star.base.report": "post-processing",
    # 主干
    "star.common": "core",
    "star.common.dom": "serialization",
    # 可视化
    "star.vis": "visualization",
    "star.vis.dom": "visualization",
    # 后处理/解历史
    "star.post": "post-processing",
    # 网格/几何
    "star.meshing": "meshing",
    "star.meshing.geometryrepair": "meshing",
    "star.trimmer": "meshing",
    "star.prismmesher": "meshing",
    "star.delaunaymesher": "meshing",
    "star.dualmesher": "meshing",
    "star.solidmesher": "meshing",
    "star.sweptmesher": "meshing",
    "star.extruder": "meshing",
    "star.twodmesher": "meshing",
    "star.bodyfittedmesher": "meshing",
    "star.surfacewrapper": "meshing",
    "star.resurfacer": "meshing",
    "star.solvermeshing": "meshing",
    "star.morpher": "meshing",
    "star.overset": "meshing",
    # 3D-CAD
    "star.cadmodeler": "cad-geometry",
    "star.brep": "cad-geometry",
    "star.starcad2": "cad-geometry",
    # 材料
    "star.material": "materials",
    # 物理模型（分散在多个包）
    "star.flow": "physics",
    "star.energy": "physics",
    "star.turbulence": "physics",
    "star.keturb": "physics",
    "star.kwturb": "physics",
    "star.segregatedflow": "physics",
    "star.coupledflow": "physics",
    "star.lagrangian": "physics",
    "star.multiphase": "physics",
    "star.vof": "physics",
    "star.eulerianmultiphasemasstransfer": "physics",
    "star.eulerianmultiphaseturb": "physics",
    "star.mixturemultiphase": "physics",
    "star.dmp": "physics",
    "star.radiation.common": "physics",
    "star.species": "physics",
    "star.segregatedspecies": "physics",
    "star.combustion": "physics",
    "star.battery": "physics",
    "star.casting": "physics",
    "star.fea.common.models": "physics",
    "star.fea": "physics",
    "star.walldistance": "physics",
    "star.metrics": "physics",
    "star.emp": "physics",
    "star.abl": "physics",
    "star.acoustics": "physics",
    "star.electromagnetism": "physics",
    "star.electrochemistry": "physics",
    "star.electrochemicalspecies": "physics",
    "star.coal": "physics",
    "star.emissions": "physics",
    "star.electroniccooling": "physics",
    "star.electropainting": "physics",
    "star.atomic": "physics",
    "star.stabilization": "solver",
    "star.segregatedenergy": "solver",
    # 求解器
    "star.solve": "solver",
    # 运动/DFBI
    "star.motion": "motion",
    "star.sixdof": "motion",
    # 协同仿真
    "star.cosimulation": "co-simulation",
    # 自动化/助理/UI
    "star.automation": "automation",
    "star.assistant": "automation",
    "star.coremodule": "client-ui",
    "star.coremodule.ui.layout": "client-ui",
    "star.cae": "cad-clients",
    "star.mapping": "mapping",
    # 容器类（无包前缀）
    "Domain": "containers",
    "InactiveList": "containers",
    "SerializableList": "containers",
    "SerializableVector": "containers",
    "MethodVector": "containers",
    "MasterArray": "containers",
    "ExportFunctionVector": "containers",
    "ExportPartVector": "containers",
    "ExportBoundaryVector": "containers",
    "ExportRegionVector": "containers",
    "PrintedMonitors": "containers",
    "MeanAnalysisVariable": "post-processing",
    "NameManager": "object-model",
    "ClassVersions": "serialization",
}

# 包前缀 -> 中文描述
LAYER_CN = {
    "object-model": "对象模型",
    "core": "仿真主干",
    "cad-geometry": "CAD 几何",
    "meshing": "网格",
    "materials": "材料",
    "physics": "物理模型",
    "solver": "求解器",
    "motion": "运动/DFBI",
    "visualization": "场景可视化",
    "post-processing": "绘图/监视器/报告",
    "co-simulation": "协同仿真",
    "automation": "自动化",
    "client-ui": "界面布局",
    "query": "查询",
    "mapping": "数据映射",
    "containers": "容器",
    "serialization": "序列化元信息",
    "cad-clients": "CAD 客户端",
    "unknown": "未分类",
}


def layer_of(class_name):
    """ClassName -> 语义层 key。"""
    if not class_name:
        return "unknown"
    if class_name.startswith("star."):
        parts = class_name.split(".")
        for i in range(len(parts) - 1, 1, -1):
            pkg = ".".join(parts[:i])
            if pkg in PACKAGE_LAYERS:
                return PACKAGE_LAYERS[pkg]
        pkg2 = ".".join(parts[:2])
        if pkg2 in PACKAGE_LAYERS:
            return PACKAGE_LAYERS[pkg2]
        return "unknown"
    for key in PACKAGE_LAYERS:
        if class_name == key or class_name.startswith(key + "<"):
            return PACKAGE_LAYERS[key]
    return "unknown"


# ---------------------------------------------------------------------------
# 2. 旧类名 -> 新类名 别名表（语料 6.x–10.x vs Javadoc 现名）
# ---------------------------------------------------------------------------
ALIASES = {
    # 绘图/视图（star.vis / star.common）
    "star.common.XyPlot": "star.common.Cartesian2DPlot",
    "star.vis.View": "star.vis.VisView",
    "star.vis.ScalarScene": "star.vis.ScalarDisplayer",
    "star.vis.VectorScene": "star.vis.VectorDisplayer",
    # 网格器
    "star.meshing.PolyhedralMesher": "star.dualmesher.DualAutoMesher",
    "star.meshing.TetrahedralMesher": "star.dualmesher.DualAutoMesher",
    "star.meshing.PrismLayerMesher": "star.prismmesher.PrismAutoMesher",
    "star.meshing.SurfaceRemesher": "star.resurfacer.ResurfacerAutoMesher",
    "star.meshing.SurfaceRepairMesher": "star.resurfacer.AutomaticSurfaceRepairAutoMesher",
    "star.meshing.TrimmerMesher": "star.trimmer.TrimmerAutoMesher",
    "star.meshing.ThinMesher": "star.solidmesher.ThinAutoMesher",
    "star.meshing.DirectedMesher": "star.sweptmesher.DirectedMesher",
    "star.meshing.ExtruderMesher": "star.extruder.ExtruderMesher",
    "star.meshing.AdvancingLayerMesher": "star.bodyfittedmesher.AdvancingLayerAutoMesher",
    "star.meshing.DelaunayMesher": "star.delaunaymesher.DelaunayAutoMesher",
    # 区域/边界旧名
    "star.common.ContinuumManager": "star.common.PhysicsContinuumManager",
}

# 自动别名：语料中的短类名（如 XyPlot）在 Javadoc 现名里按后缀匹配
JAVADOC_KNOWN = {
    # 短名 -> 现名（用于语料类名自动升级）
    "XyPlot": "star.common.Cartesian2DPlot",
    "View": "star.vis.VisView",
    "PolyhedralMesher": "star.dualmesher.DualAutoMesher",
    "TetrahedralMesher": "star.dualmesher.DualAutoMesher",
    "PrismLayerMesher": "star.prismmesher.PrismAutoMesher",
    "SurfaceRemesher": "star.resurfacer.ResurfacerAutoMesher",
    "TrimmerMesher": "star.trimmer.TrimmerAutoMesher",
}


def resolve_class(class_name):
    """把（可能是旧版本的）ClassName 解析为现名。"""
    if class_name in ALIASES:
        return ALIASES[class_name]
    short = class_name.rsplit(".", 1)[-1]
    if short in JAVADOC_KNOWN:
        return JAVADOC_KNOWN[short]
    return class_name


# ---------------------------------------------------------------------------
# 3. 属性 -> 引用方向 规则（用于全量建树）
#    方向: "up"   = 被引用对象是父（属性指向所属对象/关联对象）
#          "down" = 被引用对象是子（属性是管理器/容器）
# ---------------------------------------------------------------------------
# 子方向（对象持有这些属性 → 属性值是它的子对象）
# 注意：只保留确证的管理器/容器名；数值列表类属性（指数向量、选项索引等）不在此列
DOWN_ATTRS = {
    "Keys", "NameManager", "ManagerManager", "ClientServerObjectGroupManager",
    "Objects", "ObjectsManager", "Parts", "Models", "Methods", "Boundaries",
    "Interfaces", "Planes", "Lights", "Regions", "regions", "interfaces",
    "BoundaryManager", "RegionManager", "ContinuumManager", "PartManager",
    "ModelManager", "DisplayerManager", "FieldFunctionManager", "ReportManager",
    "MonitorManager", "PlotManager", "SceneManager", "TableManager",
    "CoordinateSystemGroup", "CreatorGroup", "Displayer", "HighlightDisplayer",
    "CurrentView", "CoordinateSystemManager", "SystemOption", "NoUnits",
    "TaskManager", "FeatureCurveManager", "BodyHandles", "PartSurfaces",
    "PartCurves", "PartContacts", "PartContactGroup", "TagGroup", "MetaData",
    "TransferRecord", "Stamper", "EdgeOption", "Descriptions", "Domain",
    "domain", "RampCalculatorManager", "UpdateEventManager", "Quantity", "Dimensions",
}
# 父方向（属性指向所属/关联对象 → 被引用对象作为父节点）
UP_ATTRS = {
    "Parent", "Simulation", "System", "Region", "Boundary", "Part", "Plot",
    "Monitor", "Table", "Units", "Dimensions", "CoordinateSystem",
    "FieldFunction", "Function", "Type", "Style", "DerivedFrom", "Units0",
    "Units1", "Units2", "Quantity", "Model", "UpdateEvent", "Iteration",
    "TimeStamp", "MonitorNormalization", "Selected", "InterpolationOption",
    "LogicOption", "FilterModel", "Tag", "Solver", "Continuum", "Scene",
}

# 已知歧义: 数值/枚举等非引用属性（建树时跳过）
NON_REF_ATTRS = {
    "Index", "Priority", "Seniority", "SampleFrequency", "StartCount",
    "EvaluatedIteration", "NotifyExtractIteration", "EvaluatedTimeStep",
    "Start", "AbsoluteSize", "size", "Conversion", "Offset", "Preferred",
    "TriangleCount", "Value", "Count", "N", "Level", "State",
    "SharpEdgeAngle", "TessellationDensity", "TessellationTimeStamp",
    "LastClosedSurfaceCheckResult", "LastClosedSurfaceCheckTime",
    "LastManifoldSurfaceCheckResult", "LastManifoldSurfaceCheckTime",
    "refreshRate", "interactorStyle", "ModeName", "Offscreen", "DepthPeel",
    "HighlightOpacity", "AxesVisible", "BackgroundColorMode", "SolidBackgroundColor",
    "GradientBackgroundColor", "AxesTextColor", "AxesViewport", "Time",
    "PhysicalTime", "isModified", "isReadOnly", "parallel", "name", "path",
    "PresentationName", "ReleaseDate", "ReleaseNumber", "BuildArch", "BuildEnv",
    "MachineConfig", "Description", "PresentationName", "Comments",
}


def attr_direction(attr_name):
    """属性名 -> 'up' / 'down' / None（非引用）。"""
    if attr_name in NON_REF_ATTRS:
        return None
    if attr_name in DOWN_ATTRS:
        return "down"
    if attr_name in UP_ATTRS:
        return "up"
    # 其余未知 → None（保守，避免把数值属性误当引用）
    return None
