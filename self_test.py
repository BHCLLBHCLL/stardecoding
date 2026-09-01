# -*- coding: utf-8 -*-
"""sim_parser.py 自检：结构不变量 + 导出有效性。"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile, walk_sections
from collections import Counter

sim = SimFile("D:/training/caedecoder/stardecoding/adjointWing_start.sim")

assert sim.header["StatePosition"] == 221991
assert sim.header["ClassName"] == "STAR"
assert len(sim.arrays) == 33
assert sim.arrays[0]["count"] == 36153
assert len(sim.state_text) == 36153
assert sim.state_magic[0] == "CD-adapco_STAR-CCM+_ID"
assert sim.state_magic[2] == "36120"
assert len(sim.tokens) == 7217, len(sim.tokens)
assert len(sim.objects) == 2076
assert sim.object_by_id(2).class_name == "star.common.Simulation"
assert sim.object_by_id(2).name == "adjointWing_start"
assert sim.object_by_id(3).class_name == "star.common.ManagerManager"
assert sim.object_by_id(4).class_name == "NameManager"

# walker 覆盖整个文件：最后一个分区（ClassVersions）的行延伸到文件末尾
sections = walk_sections(sim.blob)
last_d, last_payload, last_start, _ = sections[-1]
assert last_d.get("ClassName") == "ClassVersions"
tail = sim.blob[last_start:].decode("latin-1").strip()
assert tail.startswith("{") and tail.endswith("}")
print("sections: %d, last section (ClassVersions) covers %d-byte tail" % (
    len(sections), len(tail)))

kinds = Counter(r["kind"] for r in sim.records)
print("record kinds:", dict(kinds))
assert kinds.get("other", 0) == 0

fmts = sorted({r["fmt"] for r in sim.records if r.get("fmt")})
print("formats:", fmts)

ptrs = [r for r in sim.records if r["kind"] == "pointer"]
unres = sorted({r["ref"] for r in ptrs if r["ref"] not in sim.objmap})
print("pointers: %d, unresolved refs: %s" % (len(ptrs), unres))

sim.export("D:/training/caedecoder/stardecoding/export_check")
objs = json.load(open("D:/training/caedecoder/stardecoding/export_check/objects.json", encoding="utf-8"))
recs = json.load(open("D:/training/caedecoder/stardecoding/export_check/state_records.json", encoding="utf-8"))
assert len(objs) == 2076 and len(recs) == len(sim.records)
print("export JSON OK:", len(objs), "objects,", len(recs), "records")

# --- 改进①：语义字典 / 分层 / 全量建树 / 校验 ---
from semantic_dict import layer_of, resolve_class, attr_direction
assert layer_of("star.vis.Scene") == "visualization"
assert layer_of("star.cadmodeler.CadModel") == "cad-geometry"
assert layer_of("star.material.Gas") == "materials"
assert layer_of("star.common.Region") == "core"
assert resolve_class("star.common.XyPlot") == "star.common.Cartesian2DPlot"
assert resolve_class("star.meshing.PolyhedralMesher") == "star.dualmesher.DualAutoMesher"
assert attr_direction("Keys") == "down" and attr_direction("Parent") == "up"
assert attr_direction("PostSweeps") is None  # 数值属性不当引用

census, named = sim.layer_census()
assert census["core"] > 1000 and census["materials"] > 0
print("layer census:", dict(census.most_common(6)), "...")

main_roots = [r for r in sim.roots if sim.children.get(r.id)]
assert len(main_roots) == 1 and main_roots[0].id == 2  # 全量建树：Simulation 为唯一主根
loose = [r for r in sim.roots if not sim.children.get(r.id)]
assert len(loose) < 100  # 语义字典建树后游离对象应大幅减少（原 304）
print("tree: single main root", main_roots[0].id, "; loose:", len(loose))

v = sim.validate_class_versions()
assert v["status"] == "diagnostic" and v["expected_classes"] > 400
print("classversions diagnostic:", v["expected_classes"], "classes,",
      v["matched"], "matched,", v["expected_total"], "/", v["actual_total"])

# --- 改进②：网格抽取 ---
m = sim.extract_mesh()
assert m["faces"] is not None and m["faces"].shape[0] == 2824
assert m["vertices"] is not None and m["vertices"].shape[0] == 1412
assert m["consistent"] is True
print("mesh:", m["faces"].shape[0], "faces,", m["vertices"].shape[0],
      "vertices, flags:", m["face_flag"], "/", m["vertex_flag"])

# --- 改进⑤⑥：版本指纹 / 长度自校验 ---
fp = sim.version_fingerprint()
assert fp["banner_version"] == "250020723" and fp["release"] == "8.03.076"
chk = sim.check_state_length()
assert chk["ok"] is True
print("fingerprint:", fp["banner_version"], "/", fp["release"], "/", fp["state_mode"],
      "; length check:", chk["detail"])

# --- 改进⑦：语义层报告 ---
rep = sim.semantic_report()
assert len(rep["regions"]) == 1 and rep["regions"][0]["name"] == "Fluid Domain"
assert rep["regions"][0]["parts"][0]["triangles"] == 2824
assert rep["continua"][0]["name"] == "Physics 1" and len(rep["continua"][0]["models"]) >= 20
assert rep["scenes"][0]["name"] == "Mesh Scene 1"
print("report: %d regions, %d continua, %d scenes, %d parts" % (
    len(rep["regions"]), len(rep["continua"]), len(rep["scenes"]), len(rep["parts"])))
# --- G1：状态表结构化语义树 / 文法统计 ---
from sim_parser import decode_state_tree, state_grammar_report

tree = decode_state_tree(sim)
assert len(tree) == len(sim.records)
kinds_t = Counter(n["head"].get("kind") for n in tree)
assert kinds_t["pointer"] == 112
# 指针三分法：对象引用可命中 objmap 并带 target
obj_refs = [n for n in tree if n.get("ref", {}).get("role") == "object-ref"]
assert obj_refs and all("target" in n["ref"] for n in obj_refs)
assert any(n["ref"]["target"]["class"] == "star.common.CoordinateSystemManager"
           for n in obj_refs)
# T 块分段（255 分隔）与几何验证（29 标记顶点三元组 100% 命中）
t_nodes = [n for n in tree if n["head"].get("fmt") == "T"
           and n["head"].get("kind") == "anonymous"]
assert t_nodes and all("segments" in n for n in t_nodes)
geo = [n["geometry_check"] for n in tree if n.get("geometry_check")]
assert sum(g["triples"] for g in geo) == 30
assert sum(g["vertex_hits"] for g in geo) == 30
g = state_grammar_report(sim)
assert g["state_mode"] == "ascii" and g["n_records"] == 152
assert g["pointer_resolved_pct"] == 67.0
assert g["fmt_distribution"]["A"] == 4
assert g["geometry_vertex_check"] == {"triples": 30, "hits": 30}
print("G1 state tree: %d nodes, %d object-refs, %d T-blocks, geo 30/30" % (
    len(tree), len(obj_refs), len(t_nodes)))

# --- G2：数组块语义标注（A<n> 引用 × 网格规模自洽性） ---
from sim_parser import array_annotation_report

g2r = array_annotation_report(sim)
assert g2r["n_arrays"] == len(sim.arrays) == 33
assert g2r["labeled_pct"] == 100.0      # adjointWing 每个数组块都有名称+用途标注
assert g2r["face_tables"] == 2          # 面索引表 count == TriangleCount×3 自洽
assert g2r["vertex_span_matched"] >= 2  # 顶点坐标表跨度与面索引表互相印证
assert g2r["a_refs_resolved"] == 7      # ascii 变体 A<n> 索引引用全部解析到数组
roles_g2 = g2r["roles"]
assert roles_g2["state-table"] == 1 and roles_g2["state-referenced"] == 6
assert roles_g2["face-indices"] == 2 and roles_g2["vertex-coords"] == 4
assert roles_g2.get("unclassified", 0) == 0
print("G2 arrays: %d annotated %.1f%%, refs=%d, face=%d, vspan=%d" % (
    g2r["n_arrays"], g2r["labeled_pct"], g2r["a_refs_resolved"],
    g2r["face_tables"], g2r["vertex_span_matched"]))

# --- G3：体网格抽取（存储体系驱动，DuplicateStorageManager + 面→单元反演） ---
import glob as _glob
CORPUS = r"D:/training/starccm/startutorialsdata"


def _find(name):
    hits = _glob.glob(CORPUS + "/**/" + name, recursive=True)
    return hits[0] if hits else None


VOL_EXPECT = {"airfoil.sim": 16987, "pipeBlockage.sim": 14882,
              "pipeMixingBlockage.sim": 14720, "methaneOnPt.sim": 1750}
for _name, _want in VOL_EXPECT.items():
    _p = _find(_name)
    assert _p, "语料缺少 %s" % _name
    _vol = SimFile(_p).extract_volume_mesh()
    assert _vol.get("ok") and _vol.get("count") == _want, (
        "%s: %s != %d (%s)" % (_name, _vol.get("count"), _want, _vol.get("reason")))
    assert _vol.get("kind") == "poly"
    assert _vol.get("points") is not None and _vol["points"].shape[1] == 3
    _fv = _vol.get("face_verts")
    assert _fv is not None and int(_fv.max()) < _vol["points"].shape[0]
    assert sum(1 for fs in _vol["cell_faces"] if not fs) == 0  # orphan=0
print("G3 volume: 4 个体网格文件单元数精确 %s, 拓扑 orphan=0 全通" % (
    {k: v for k, v in VOL_EXPECT.items()}))

# 纯表面网格文件诚实拒绝（旧启发式曾把直升机顶点标志表误判为 tet）
_hp = _find("genericHelicopter_start.sim")
assert _hp
_vol = SimFile(_hp).extract_volume_mesh()
assert not _vol.get("ok"), "直升机应为纯表面网格（无体网格存储组）"
print("G3 volume: %s 诚实拒绝（%s）" % ("genericHelicopter_start.sim",
                                       _vol.get("reason")))

# VTK UnstructuredGrid 导出结构：offsets 累计 / types=41 / connectivity 合法
import os as _os, re as _re
_meth = SimFile(_find("methaneOnPt.sim"))
_vtu = _meth.export_volume_vtu(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "_g3_selfcheck.vtu"))
assert _vtu and _os.path.exists(_vtu)
_txt = open(_vtu, encoding="utf-8").read()
assert 'NumberOfPoints="3780"' in _txt and 'NumberOfCells="1750"' in _txt
_conn = [int(x) for x in _re.search(
    r'<DataArray[^>]*Name="connectivity"[^>]*>(.*?)</DataArray>',
    _txt, _re.S).group(1).split()]
_offs = [int(x) for x in _re.search(
    r'<DataArray[^>]*Name="offsets"[^>]*>(.*?)</DataArray>',
    _txt, _re.S).group(1).split()]
assert len(_offs) == 1750 and _offs[-1] == len(_conn)
assert _re.search(r'Name="types"[^>]*>\s*41(?:\s+41)*\s*<', _txt)
_os.remove(_vtu)
print("G3 vtu: VTK_POLYHEDRON 导出结构校验通过（1750 cells, offsets 累计, types 全 41）")

# --- G4：体网格边界 ↔ Boundary 精确映射（FvBoundary → Boundary 对象链） ---
G4_EXPECT = {"pipeBlockage.sim": (3050, 4), "airfoil.sim": (336, 7),
             "methaneOnPt.sim": (3778, 6), "pipeMixingBlockage.sim": (4478, 5)}
_g4 = {}
for _name, (_want_f, _want_b) in G4_EXPECT.items():
    _p = _find(_name)
    assert _p, "语料缺少 %s" % _name
    _s = SimFile(_p)
    _vol = _s.extract_volume_mesh()
    _bf = _s.extract_boundary_faces(_vol)
    _g4[_name] = _bf
    assert _bf.get("ok"), "%s: %s" % (_name, _bf.get("reason"))
    assert _bf.get("total_faces") == _want_f, (
        "%s: total=%s != %d" % (_name, _bf.get("total_faces"), _want_f))
    assert len(_bf["boundaries"]) == _want_b, (
        "%s: boundaries=%d != %d" % (_name, len(_bf["boundaries"]), _want_b))
    for _b in _bf["boundaries"]:
        assert _b["face_count"] == len(_b["owner_cells"]) == len(_b["rings"])
        assert 0 <= min(_b["owner_cells"]) and max(_b["owner_cells"]) < _vol["count"]
        assert max(max(_r) for _r in _b["rings"]) < _vol["points"].shape[0]
assert sum(b["face_count"] for b in _g4["pipeBlockage.sim"]["boundaries"]) == 3050
# psi == Boundary.PartSurfaces 组 Keys 所指 PartSurface.Index（对象级闭合，逐边界吻合）
assert all(b["part_surfaces"]
           and b["part_surface_index"] == b["part_surfaces"][0]["index"]
           for n in ("pipeBlockage.sim", "airfoil.sim", "pipeMixingBlockage.sim")
           for b in _g4[n]["boundaries"])
# pipeBlockage 1:1 场景：psi == Boundary.Index == 1/2/3/4
assert sorted(b["part_surface_index"]
              for b in _g4["pipeBlockage.sim"]["boundaries"]) == [1, 2, 3, 4]
# airfoil：Boundary.Index 是 Region 内序号（24..30）≠ 全局 PartSurface.Index（47..55）；
# PartSurface 名与边界名同构
assert sorted(b["index"] for b in _g4["airfoil.sim"]["boundaries"]) == list(range(24, 31))
assert all(b["name"].endswith(b["part_surfaces"][0]["name"])
           for b in _g4["airfoil.sim"]["boundaries"])
# methaneOnPt 型：无 PartSurface 通道（ProstarBounId 代替），FvBoundary 链兜底
assert all(b["part_surface_index"] is None and not b["part_surfaces"]
           for b in _g4["methaneOnPt.sim"]["boundaries"])
# airfoil 2D 多边形网格：边界面=边界边（环长 2）
assert all(max(len(r) for r in b["rings"]) <= 2
           for b in _g4["airfoil.sim"]["boundaries"])
print("G4 boundary: 4 个体网格文件 22 边界 / 11642 边界面精确"
      "（psi==PartSurface.Index 对象级闭合；methaneOnPt 无 psi 链兜底；airfoil 2D 环长=2）")

# 纯表面网格文件诚实拒绝
_hp = SimFile(_find("genericHelicopter_start.sim")).extract_boundary_faces()
assert not _hp.get("ok")
print("G4 boundary: %s 诚实拒绝（%s）" % ("genericHelicopter_start.sim", _hp.get("reason")))

# GUI 着色证据（3D→polys / 2D→lines，与表面路径 boundary_colored_polydata 同构）
try:
    from star_gui_vtk import boundary_colored_volume_polydata
    _pbc = boundary_colored_volume_polydata(SimFile(_find("pipeBlockage.sim")))
    assert _pbc and _pbc["kind"] == "polys" and len(_pbc["label_names"]) == 4
    assert _pbc["polydata"].GetNumberOfPolys() == 3050
    assert _pbc["polydata"].GetCellData().GetScalars().GetNumberOfTuples() == 3050
    _abc = boundary_colored_volume_polydata(SimFile(_find("airfoil.sim")))
    assert _abc and _abc["kind"] == "lines" and len(_abc["label_names"]) == 7
    assert _abc["polydata"].GetNumberOfLines() == 336
    print("G4 gui: pipeBlockage=polys/3050 面 4 边界, airfoil=lines/336 边 7 边界")
except ImportError:
    print("G4 gui: vtk 不可用，跳过着色冒烟")

# --- G5：内嵌解场抽取（SolutionRepresentation → FvRegion cells 组） ---
_G5_DIR = r"D:/training/openfoam/benchmark"
_g5p = None
for _n in sorted(os.listdir(_G5_DIR)):
    if not (_n.startswith("vortexShed_tutor") and _n.endswith(".sim")):
        continue
    _cand = os.path.join(_G5_DIR, _n)
    try:
        if SimFile(_cand).extract_solution_fields().get("ok"):
            _g5p = _cand
            break
    except Exception:
        continue
assert _g5p, "语料缺少含解场的 vortexShed_tutor*.sim"
_g5 = SimFile(_g5p).extract_solution_fields()
assert _g5.get("ok"), _g5.get("reason")
assert _g5["cell_count"] == 20245
assert _g5["n_fields"] >= 8
assert _g5.get("region_name") == "Fluid_Domain"
_gnames = {f["name"]: f for f in _g5["fields"]}
assert "Pressure" in _gnames and _gnames["Pressure"]["components"] == 1
assert _g5["data"]["Pressure"].shape == (20245,)
assert abs(float(_g5["data"]["Pressure"].mean())) < 0.01  # 不可压 ΔP 均值≈0
assert float(_g5["data"]["W_Velocity"].max()) == 0.0      # 2D：W 全 0
assert _gnames["VelocityFieldFunction"]["components"] == 3
assert _g5["data"]["VelocityFieldFunction"].shape == (20245, 3)
assert float(_g5["data"]["U_Velocity"].mean()) > 0.001     # 来流 U>0
print("G5 solution: %s 单元=20245 字段=%d 区域=%s"
      "（Pressure ΔP 均值≈0, W=0 二维, 矢量 x3）" % (
          os.path.basename(_g5p), _g5["n_fields"], _g5.get("region_name")))

# 无解场文件诚实拒绝（教程 *_start / 未求解）
_g5n = SimFile(_find("pipeBlockage.sim")).extract_solution_fields()
assert not _g5n.get("ok") and "SolutionRepresentation" in _g5n.get("reason")
print("G5 solution: pipeBlockage.sim 诚实拒绝（%s）" % _g5n.get("reason"))

# GUI 解场着色冒烟（真解场 → 单元标量）
try:
    from star_gui_vtk import solution_colored_volume_polydata
    _sr = solution_colored_volume_polydata(SimFile(_g5p))
    assert _sr and _sr["kind"] == "lines" and _sr["field"] == "Pressure"
    assert _sr["polydata"].GetCellData().GetScalars().GetNumberOfTuples() > 0
    print("G5 gui: 解场着色 kind=%s 面=%d field=%s range=[%.4g..%.4g]" % (
        _sr["kind"], _sr["total_faces"], _sr["field"], _sr["min"], _sr["max"]))
except ImportError:
    print("G5 gui: vtk 不可用，跳过着色冒烟")

# --- G6：监视器曲线重建（MonitorManager → XAxisData/YAxisValues 双 MasterArray，两代子格式通吃） ---
_g6 = SimFile(_g5p).extract_monitor_curves()
assert _g6.get("ok"), _g6.get("reason")
_m6 = {e["name"]: e for e in _g6["monitors"]}
assert "Continuity" in _m6 and "Iteration" in _m6 and "Physical Time" in _m6
assert _m6["Iteration"]["last"] == _m6["Iteration"]["n"]      # 残差每迭代记录
_cv = _m6["Continuity"]["cur_value"]
assert isinstance(_cv, float) and abs(
    _cv - _m6["Continuity"]["last"]) <= 1e-9 * max(1.0, abs(_cv))
assert _m6["Physical Time"]["index_first"] == 15.0            # StarUpdate 间隔 15
_lf = _m6["升力系数 Monitor"]
assert _lf["y_min"] < -0.1 and _lf["y_max"] > 0.1             # 涡脱落振荡
print("G6 curves: %s %d 监视器（Continuity ==CurrentValue, 升力系数 涡脱落 [%.3f..%.3f]）" % (
    os.path.basename(_g5p), len(_g6["monitors"]), _lf["y_min"], _lf["y_max"]))

# 绘图关联 + 按绘图导出对齐 XY CSV（G2 标注：标题/轴标题/图例）
_g6q = SimFile(_g5p).extract_plots(_g6)
assert _g6q.get("ok"), _g6q.get("reason")
_r6 = next(q for q in _g6q["plots"] if q["title"] == "Residuals")
assert _r6["x_title"] == "Iteration"
assert len(_r6["series"]) >= 3
assert all(s["kind"] == "monitor" for s in _r6["series"])
_csv6 = "_g6_selftest.csv"
assert SimFile(_g5p).export_plot_csv(_csv6, _g6, _r6) == _csv6
with open(_csv6, encoding="utf-8") as fh:
    _rr6 = fh.read().splitlines()
assert _rr6[0] == "Iteration,Continuity,X-momentum,Y-momentum"
assert len(_rr6) == _m6["Continuity"]["n"] + 1
os.remove(_csv6)
print("G6 plots: Residuals x%d，CSV %d 行（X=迭代号）" % (
    len(_r6["series"]), len(_rr6) - 1))

# 新版子格式精确锚点（v3_0.05_2502：300000 迭代 / 200s / Farassat1A 表数据）
_g6v3 = os.path.join(_G5_DIR, "vortexShed_tutor_v3_0.05_2502.sim")
if os.path.isfile(_g6v3):
    _g6b = SimFile(_g6v3).extract_monitor_curves()
    assert _g6b.get("ok"), _g6b.get("reason")
    _m6b = {e["name"]: e for e in _g6b["monitors"]}
    assert len(_g6b["monitors"]) == 6
    assert _m6b["Iteration"]["n"] == 300000
    assert _m6b["Iteration"]["last"] == 300000.0
    assert _m6b["Physical Time"]["n"] == 20000
    assert abs(_m6b["Physical Time"]["y_min"] - 0.01) < 1e-12
    assert abs(_m6b["Physical Time"]["y_max"] - 200.0) < 1e-9
    assert _m6b["Continuity"]["n"] == 300000
    assert _m6b["升力系数 Monitor"]["n"] == 20000
    assert _m6b["升力系数 Monitor"]["y_min"] < -0.27
    assert _m6b["升力系数 Monitor"]["y_max"] > 0.28
    _g6qb = SimFile(_g6v3).extract_plots(_g6b)
    _t6b = {q["title"]: q for q in _g6qb["plots"]}
    assert set(_t6b) == {"Residuals", "升力系数 Monitor 绘图", "Monitor Plot"}
    assert _t6b["升力系数 Monitor 绘图"]["x_units"] == "s"
    assert len(_t6b["升力系数 Monitor 绘图"]["series"]) == 1
    _tab6 = [s for s in _t6b["Monitor Plot"]["series"] if s["kind"] == "tabular"]
    assert len(_tab6) == 3 and any(
        "Farassat1A-Patch-time.dat" in (s.get("table_file") or "") for s in _tab6)
    assert SimFile(_g6v3).export_plot_csv(_csv6, _g6b, _t6b["Residuals"]) == _csv6
    with open(_csv6, encoding="utf-8") as fh:
        _rr6b = fh.read().splitlines()
    assert len(_rr6b) == 300001
    assert SimFile(_g6v3).export_plot_csv(
        _csv6, _g6b, _t6b["升力系数 Monitor 绘图"]) == _csv6
    with open(_csv6, encoding="utf-8") as fh:
        _lr6 = fh.read().splitlines()
    assert _lr6[0] == "Physical Time,升力系数 Monitor"
    assert len(_lr6) == 20001
    os.remove(_csv6)
    print("G6 anchor v3: 300000 迭代 / 200s，Residuals %d 行 + 升力系数 %d 行"
          "（X=迭代号/物理时间, tabular x%d）" % (
              len(_rr6b) - 1, len(_lr6) - 1, len(_tab6)))

# 未求解文件诚实拒绝（结构完整但监视器无数据）
_g6n = SimFile(_find("pipeBlockage.sim")).extract_monitor_curves()
assert not _g6n.get("ok") and "未求解" in _g6n.get("reason")
print("G6 curves: pipeBlockage.sim 诚实拒绝（%s）" % _g6n.get("reason"))

# GUI 曲线冒烟（真实采样 → 降采样 X 定位 + 标注行）
try:
    from star_gui_plots import monitor_curve_items, monitor_report_lines
    _s6 = SimFile(_g5p)
    _items6 = monitor_curve_items(_s6)
    assert len(_items6) == 6
    assert all(len(it[2]) >= 2 and it[4] is not None for it in _items6)
    _lines6 = monitor_report_lines(_s6)
    assert any("Residuals" in ln for ln in _lines6)
    print("G6 gui: 监视器曲线 %d 条（X 定位降采样）+ 标注行 %d" % (
        len(_items6), len(_lines6)))
except ImportError:
    print("G6 gui: PyQt5 不可用，跳过曲线冒烟")

# --- G7：物理模型/材料/运动参数解码（Javadoc 属性名 ↔ 语料属性值对照） ---
_g7kw = os.path.join(_G5_DIR, "vortexShed_tutor_v3_0.025_k-omega.sim")
assert os.path.isfile(_g7kw), "语料缺少 vortexShed_tutor_v3_0.025_k-omega.sim"
_g7s = SimFile(_g7kw)
_g7 = _g7s.extract_physics()
assert _g7.get("ok"), _g7.get("reason")
_g7c = _g7["continua"][0]
_g7m = {m["class"].rsplit(".", 1)[-1]: m for m in _g7c["models"]}
_sst7 = _g7m["SstKwTurbModel"]["params"]
assert abs(_sst7["A1"] - 0.31) < 1e-12 and abs(_sst7["BetaStar"] - 0.09) < 1e-12
assert _sst7["KwTurbCompressibilityParameters"]["ZetaStar"] == 1.5
assert _sst7["VorticityTimeParameter"]["Value"] == 0.075
_air7 = {p["name"]: p for p in
         next(m for m in _g7["materials"] if m["name"] == "Air")["properties"]}
assert abs(_air7["DynamicViscosityProperty"]["value"] - 2e-05) < 1e-12
assert _air7["DynamicViscosityProperty"]["units"] == "Pa-s"
assert abs(_air7["MolecularWeightProperty"]["value"] - 28.9664) < 1e-12
assert _air7["DrhoDpProperty"]["method"] == "NullMaterialPropertyMethod"
assert _g7["motion"] and _g7["motion"][0]["region"] == "Fluid_Domain"
print("G7 physics: %s 模型=%d（SstKw A1=0.31/BetaStar=0.09）Air 属性=%d"
      "（DynamicViscosity=2e-05 Pa-s）" % (
          os.path.basename(_g7kw), len(_g7c["models"]), len(_air7)))

# MRF 旋转：openWaterPropeller 旋转域 RotationRate=15 rps Axis=(1,0,0)
_g7p = _find("openWaterPropeller_start.sim")
assert _g7p, "语料缺少 openWaterPropeller_start.sim"
_g7rs = SimFile(_g7p)
_g7r = _g7rs.extract_physics()
assert _g7r.get("ok"), _g7r.get("reason")
_rot7 = next(m for m in _g7r["motion"] if m["region"] == "Rotating Region")
assert _rot7["ref_frame_class"].endswith("UserRotatingReferenceFrame")
assert abs(_rot7["RotationRate"]["value"] - 15.0) < 1e-12
assert _rot7["RotationRate"]["units"] == "rps"
assert _rot7["AxisVector"]["value"] == [1.0, 0.0, 0.0]
print("G7 motion: openWaterPropeller 旋转域 RotationRate=15 rps Axis=(1,0,0)")

# 纯几何 CAD 诚实拒绝（无 PhysicsContinuum；_start 文件即使无网格也带连续体）
_g7n = SimFile(_find("directedMeshCAD.sim")).extract_physics()
assert not _g7n.get("ok") and "PhysicsContinuum" in _g7n.get("reason")
print("G7 physics: directedMeshCAD.sim 诚实拒绝（%s）" % _g7n.get("reason"))

# GUI 属性面板语义行（G7: 前缀 + raw=None，StarSceneModel 纯逻辑层）
from star_gui_model import StarSceneModel
_g7g = StarSceneModel(_g7s)
_g7pc = next(o for o in _g7s.objects
             if o.class_name == "star.common.PhysicsContinuum")
_g7k = [k for k, _t, _r in _g7g.properties(_g7pc)]
assert sum(k.startswith("G7:") for k in _g7k) == len(_g7c["models"])
assert "G7:模型 SstKwTurbModel" in _g7k
_g7air = next(o for o in _g7s.objects if o.class_name == "star.material.Gas")
assert any(k == "G7:属性 DynamicViscosityProperty" and "2e-05 Pa-s" in t
           for k, t, _r in _g7g.properties(_g7air))
_g7mo = StarSceneModel(_g7rs)
_g7mr = dict((k, t) for k, t, _r in _g7mo.properties(_g7mo.object_by_id(_rot7["id"])))
assert _g7mr.get("G7:RotationRate") == "15 rps"
assert _g7mr.get("G7:AxisVector") == "(1, 0, 0) Dimensionless"
assert _g7mr.get("G7:运动 Region") == "Rotating Region"
print("G7 gui: 属性面板 G7 行 连续体=%d 模型行 + 材料属性行 + 旋转帧 15 rps" % (
    len(_g7c["models"])))

print("ALL CHECKS PASSED")
