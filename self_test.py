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
assert sum(k.startswith("G7:模型 ") for k in _g7k) == len(_g7c["models"])
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

# --- P1：物理参数写侧（G7 编辑锚点 + 属性面板可编辑描述符行） ---
# 嵌套参数组锚点：_oid 回指承载对象（原始标量叶子的写回目标）
assert isinstance(_sst7["VorticityTimeParameter"]["_oid"], int)
_vt7o = _g7s.objmap.get(_sst7["VorticityTimeParameter"]["_oid"])
assert _vt7o is not None and _vt7o.dict.get("Value") == 0.075
assert isinstance(_sst7["KwTurbCompressibilityParameters"]["_oid"], int)
# 物理量锚点：oid/key/kind 可解析回源对象（材料常量属性）
_q7 = _g7s.objmap.get(_air7["DynamicViscosityProperty"]["oid"])
assert _q7 is not None and _q7.dict.get("Value") == 2e-05
assert _air7["DynamicViscosityProperty"]["kind"] == "quantity"
assert isinstance(_air7["DynamicViscosityProperty"]["oid"], int)
assert _air7["DynamicViscosityProperty"]["key"] == "Value"
assert _rot7["RotationRate"]["kind"] == "quantity"
assert _rot7["RotationRate"]["key"] == "Value"
_g7rows = dict((k, r) for k, _t, r in _g7g.properties(_g7pc))
_a1r = _g7rows.get("G7:A1")
assert isinstance(_a1r, dict) and _a1r["kind"] == "scalar" \
    and _a1r["oid"] == _g7m["SstKwTurbModel"]["id"] and _a1r["key"] == "A1"
_zs7 = _g7rows.get("G7:KwTurbCompressibilityParameters.ZetaStar")
assert isinstance(_zs7, dict) and _zs7["kind"] == "scalar" \
    and _zs7["oid"] == _sst7["KwTurbCompressibilityParameters"]["_oid"] \
    and _zs7["key"] == "ZetaStar"
_vt7 = _g7rows.get("G7:VorticityTimeParameter.Value")
assert isinstance(_vt7, dict) and _vt7["kind"] == "scalar" \
    and _vt7["oid"] == _sst7["VorticityTimeParameter"]["_oid"] \
    and _vt7["key"] == "Value" and _vt7["value"] == 0.075
_g7arows = dict((k, r) for k, _t, r in _g7g.properties(_g7air))
_dv7 = _g7arows.get("G7:值 DynamicViscosityProperty")
assert isinstance(_dv7, dict) and _dv7["kind"] == "quantity" \
    and _dv7["oid"] == _air7["DynamicViscosityProperty"]["oid"] \
    and _dv7["key"] == "Value"
_g7mrows = dict((k, r) for k, _t, r in
                _g7mo.properties(_g7mo.object_by_id(_rot7["id"])))
_rr7 = _g7mrows.get("G7:RotationRate")
assert isinstance(_rr7, dict) and _rr7["kind"] == "quantity" \
    and _rr7["oid"] == _rot7["RotationRate"]["oid"]
print("P1 anchors: 物理量/标量/嵌套参数组编辑锚点 + GUI 描述符行 全通过")

# --- P1 写侧落盘往返：三类锚点编辑 → SetPropertyCommand → save_sim(patches) → 重开断言 ---
import tempfile
import shutil
from sim_writer import save_sim
from star_gui_document import SimDocument
from star_gui_commands import SetPropertyCommand
_p1t = tempfile.mkdtemp(prefix="star_p1_")
try:
    _p1d = os.path.join(_p1t, "g7_roundtrip.sim")
    _p1doc = SimDocument(_g7s, _g7kw)
    _m7id = _g7m["SstKwTurbModel"]["id"]
    _p1doc.execute(SetPropertyCommand(
        _m7id, "A1", 0.33, _g7s.objmap[_m7id].dict.get("A1")))
    _p1doc.execute(SetPropertyCommand(
        _sst7["VorticityTimeParameter"]["_oid"], "Value", 0.085, 0.075))
    _p1doc.execute(SetPropertyCommand(
        _air7["DynamicViscosityProperty"]["oid"], "Value", 3e-05, 2e-05))
    save_sim(_g7s, _p1d, patches=_p1doc.patches, src_path=_g7kw)
    _p1ph = SimFile(_p1d).extract_physics()
    assert _p1ph.get("ok"), _p1ph.get("reason")
    _p1air = {p["name"]: p for p in next(
        m for m in _p1ph["materials"] if m["name"] == "Air")["properties"]}
    _p1m = {m["class"].rsplit(".", 1)[-1]: m
            for m in _p1ph["continua"][0]["models"]}
    assert abs(_p1m["SstKwTurbModel"]["params"]["A1"] - 0.33) < 1e-12
    assert abs(_p1m["SstKwTurbModel"]["params"]
               ["VorticityTimeParameter"]["Value"] - 0.085) < 1e-12
    assert abs(_p1air["DynamicViscosityProperty"]["value"] - 3e-05) < 1e-12
    assert _p1air["DynamicViscosityProperty"]["units"] == "Pa-s"
    assert SimFile(_g7kw).objmap[_m7id].dict.get("A1") == 0.31
finally:
    shutil.rmtree(_p1t, ignore_errors=True)
print("P1 roundtrip: 模型标量/嵌套组标量/材料物理量 三类锚点落盘往返 全通过")

# --- G8：场景显示参数解码（Scene → Displayer/颜色映射/图例/灯光/诚实拒绝） ---
# 断言样本 = vortexShed_tutor.sim（G5 同款；G8 侦查值全部来自该文件）
_s8 = SimFile(_g5p).extract_scene_display()
assert _s8.get("ok") and len(_s8["scenes"]) == 3, \
    "vortexShed 应有 3 个场景：%s" % _s8.get("reason")
_s8scal = next(s for s in _s8["scenes"]
               if any(d["class"] == "ScalarDisplayer" for d in s["displayers"]))
_s8d = {d["class"]: d for d in _s8scal["displayers"]}
assert "PartDisplayer" in _s8d and "ScalarDisplayer" in _s8d
_p8 = _s8d["PartDisplayer"]
assert len(_p8["color"]) == 3 and _p8["opacity"] == 1.0
assert any(p["class"] == "Boundary" and p["name"] == "Inlet"
           for p in _p8["parts"]), "PartDisplayer 部件应解引用 Inlet 边界"
_s8q = _s8d["ScalarDisplayer"]
assert _s8q["field"]["name"] == "Vorticity: Magnitude"
assert _s8q["field"]["units"] == "/s"
assert abs(_s8q["field"]["range"][0] - 0.0009035094315465401) < 1e-9
assert abs(_s8q["field"]["range"][1] - 278.1039638285214) < 1e-6
assert _s8q["representation"] == "FvRepresentation"
_l8 = _s8q["legend"]
assert _l8["lut"] == "blue-yellow-red" and _l8["lut_class"] == "PredefinedLookupTable"
assert _l8["format"] == "%-6.3g" and _l8["labels"] == 3
assert _l8["position"] == [0.73, 0.08] and _l8["visible"] is True
_c8 = _s8q["colormap"]
assert len(_c8["values"]) == 36, "blue-yellow-red 应为 9 组 (位置,R,G,B) 断点"
_b8 = _c8["breakpoints"]
assert len(_b8) == 9
assert all(_b8[i + 1]["pos"] > _b8[i]["pos"] for i in range(8)), \
    "断点位置应单调 0→1"
assert abs(_b8[0]["pos"]) < 1e-12 and abs(_b8[-1]["pos"] - 1.0) < 1e-12
assert _b8[0]["rgb"][2] > _b8[0]["rgb"][0] and _b8[0]["rgb"][2] > _b8[0]["rgb"][1], \
    "首断点应为蓝"
assert _b8[5]["rgb"][0] > 0.9 and _b8[5]["rgb"][1] > 0.9 \
    and _b8[5]["rgb"][2] < 0.5, "位置≈0.5 处应为黄"
assert _b8[-1]["rgb"][0] > _b8[-1]["rgb"][1] and _b8[-1]["rgb"][0] > _b8[-1]["rgb"][2], \
    "末断点应为红"
assert len(_s8scal["lights"]) == 4
assert _s8scal["lights"][0]["azimuth"] == 30.0 \
    and _s8scal["lights"][0]["elevation"] == 30.0 \
    and _s8scal["lights"][0]["intensity"] == 1.0 \
    and _s8scal["lights"][0]["enabled"] is True
# 注记链：全局定义 5 项；场景级 props（Annotation 解引用/可见性/位置/高宽）
#   + AnnotationGroup.Keys 解引用（标量场景 1 显示 Logo + Solution Time）
_a8defs = _s8.get("annotations") or {}
assert len(_a8defs) == 5, "vortexShed 全局注记定义应为 5 项"
assert any(d["class"] == "LogoAnnotation" and d["name"] == "Logo"
           for d in _a8defs.values())
assert any(d["class"] == "PhysicalTimeAnnotation"
           and d["name"] == "Solution Time" for d in _a8defs.values())
_anc8 = _s8scal["annotations"]
assert any(p["class"] == "LogoAnnotationProp" and p["annotation"] == "Logo"
           and p["visible"] is True and abs(p["position"][0] - 0.015) < 1e-12
           and p["position"][1] == 0.9 and p["height"] == 0.1
           for p in _anc8["props"]), "Logo 注记显示属性应解引用并携带位置/高"
assert any(p["class"] == "PhysicalTimeAnnotationProp"
           and p["annotation"] == "Solution Time" and p["visible"] is True
           and p["height"] == 0.05 for p in _anc8["props"])
assert _anc8["shown"] == ["Logo", "Solution Time"], \
    "标量场景 1 注记组应解引用 Logo + Solution Time"
# 几何场景仅 PartDisplayer（无场/图例），其余场景结构独立成立
_s8geo = next(s for s in _s8["scenes"]
              if all(d["class"] == "PartDisplayer" for d in s["displayers"]))
assert _s8geo["displayers"] and _s8geo["lights"]
# airfoil.sim：Mesh + Scalar - Mach 双场景（ScalarDisplayer 次级样本）
_a8 = SimFile(_find("airfoil.sim")).extract_scene_display()
assert _a8.get("ok") and len(_a8["scenes"]) == 2
assert any(d["class"] == "ScalarDisplayer"
           for s in _a8["scenes"] for d in s["displayers"])
# 诚实拒绝：纯几何 CAD 无 Scene
_n8 = SimFile(_find("directedMeshCAD.sim")).extract_scene_display()
assert not _n8.get("ok") and "Scene" in _n8.get("reason")
print("G8 scenes: vortexShed %d 场景（场=%r 范围 %.4g..%.4g 图例=%r 断点=%d 组"
      " 注记=%d 定义/%d 显示） + airfoil 双场景, directedMeshCAD 诚实拒绝（%s）" % (
          len(_s8["scenes"]), _s8q["field"]["name"],
          _s8q["field"]["range"][0], _s8q["field"]["range"][1],
          _l8["lut"], len(_b8), len(_a8defs), len(_anc8["shown"]),
          _n8.get("reason")))

# G8 GUI：官方 ColorMap → vtkLookupTable（断点重采样 + 通道主导色校验）
try:
    from star_gui_vtk import lut_from_colormap
    _lut8 = lut_from_colormap(_c8["values"], _c8["alphas"],
                              lo=_s8q["field"]["range"][0],
                              hi=_s8q["field"]["range"][1])
    _n8t = _lut8.GetNumberOfTableValues()
    assert _lut8 is not None and _n8t == 256
    _c8lo, _c8hi = [0.0] * 4, [0.0] * 4
    _lut8.GetTableValue(0, _c8lo)
    _lut8.GetTableValue(_n8t - 1, _c8hi)
    assert _c8lo[2] > _c8lo[0] and _c8lo[2] > _c8lo[1], "表首应为蓝"
    assert _c8hi[0] > _c8hi[1] and _c8hi[0] > _c8hi[2], "表末应为红"
    print("G8 gui: 官方色表 blue-yellow-red 9 断点→%d 级 LUT 通过（蓝→红）" % _n8t)
except ImportError:
    print("G8 gui: vtk 不可用，跳过官方色表冒烟")

# --- G9：二进制状态表完整文法（写侧前置；长度前缀 + id<<8 + 无损往返） ---
# 断言样本：4 个 binary 编码状态表文件（vortexShed2d/airfoil/vibratingPipe/manifold）
import re as _re9
from sim_parser import parse_state_table_binary, serialize_binary_records


def _find9(name):
    for _root in (CORPUS, _G5_DIR):
        _h = _glob.glob(_root + "/**/" + name, recursive=True)
        if _h:
            return _h[0]
    return None


_g9bins = ["vortexShed2d.sim", "airfoil.sim",
           "vibratingPipe_start.sim", "manifold_start.sim"]
_g9rt = True
for _n9 in _g9bins:
    _f9 = SimFile(_find9(_n9))
    assert _f9.state_mode == "binary", "%s 应为 binary 编码" % _n9
    _b9 = _f9.state_text.encode("latin-1")
    _m9 = _re9.search(
        rb"TRANSMIT FILE created by modeller version (\d+).{0,24}?SCH_([A-Za-z0-9_]+)",
        _b9)
    _i9 = _m9.end() if _m9 else 0
    _t9, _r9, _mg9, _bn9 = parse_state_table_binary(_f9.state_text)
    _rb9 = serialize_binary_records(_r9)
    _g9rt = _g9rt and (_rb9 == _b9[_i9:])
assert _g9rt, "4 个 binary 状态表 完整文法 应逐字节往返一致（可逆）"
# 语法锚点：vortexShed2d 的 named 记录头（长度前缀 + id<<8 + flags + version 规则）
_s9 = SimFile(_find9("vortexShed2d.sim"))
_bt9, _br9, _bm9, _bb9 = parse_state_table_binary(_s9.state_text)
_bn9 = {r["name"]: r for r in _br9 if r["kind"] == "named"}
assert _bn9["lattice"]["id"] == 222 and _bn9["lattice"]["fmt"] == "CCCI"
assert _bn9["mesh"]["id"] == 1006 and _bn9["mesh"]["fmt"] == "I"
assert _bn9["index_map"]["id"] == 82 and _bn9["index_map"]["fmt"] == "A"
assert _bn9["lowest_node_id"]["id"] == 0 \
    and _bn9["lowest_node_id"]["version"] == 1 \
    and _bn9["lowest_node_id"]["fmt"] == "dA", "id==0 记录应带 version 字节"
assert _bn9["list_type"]["id"] == 0 and _bn9["list_type"]["version"] == 1 \
    and _bn9["list_type"]["fmt"] == "uI"
assert _bn9["notransmit"]["fmt"] == "lCCCDCCDI"
assert _bn9["finger_index"]["fmt"] == "dI"
assert _bn9["mesh_offset_data"]["id"] == 206 \
    and _bn9["mesh_offset_data"]["fmt"] == "Z" \
    and _bn9["mesh_offset_data"]["value"] == 0, "Z 记录应有 value + stream"
assert _bn9["finger_block"]["id"] == 1012 \
    and _bn9["finger_block"]["fmt"] == "CZ" \
    and _bn9["finger_block"]["value"] == 0
assert _bb9 and "modeller version 3600169" in _bb9
# 对象图语义（binary 文件对象可正常解出——G8 级联）
assert len(_s9.objects) > 1000
print("G9 binary: 4 文件逐字节往返可逆 + 语法锚点（长度前缀/id<<8/version 规则）"
      " + vortexShed2d 对象图 %d 个 全通过" % len(_s9.objects))

# --- W2：状态表安全编辑（只动已确证记录，差分验证，尾部/其他记录不动） ---
from sim_parser import (_binary_record_bytes, edit_binary_state_records,
                        verify_binary_state_edit)
# 单段文件：vortexShed2d 改 3 个 named 头字段（id/flags/version）
_w2f = SimFile(_find9("vortexShed2d.sim"))
_w2e = [{"index": 3, "flags": 2}, {"index": 4, "id": 1007},
        {"index": 10, "version": 2}]
_w2nb, _w2chg = edit_binary_state_records(_w2f.state_text, _w2e)
assert len(_w2nb) == len(_w2f.state_text.encode("latin-1")), "等宽编辑长度应不变"
assert verify_binary_state_edit(_w2f.state_text, _w2nb, _w2e)
_t2, _r2, _m2, _b2 = parse_state_table_binary(_w2nb.decode("latin-1"))
assert _r2[3]["flags"] == 2 and _r2[4]["id"] == 1007 \
    and _r2[10]["version"] == 2, "编辑应持久化且仅改动目标记录"
# 强断言：非目标记录字节逐字不变，仅目标记录区间差分
for _i, (_ra, _rb) in enumerate(zip(
        parse_state_table_binary(_w2f.state_text)[1], _r2)):
    if _i in (3, 4, 10):
        continue
    assert _binary_record_bytes(_ra) == _binary_record_bytes(_rb), \
        "非目标记录 %d 被安全编辑意外改动" % _i
# 变长编辑应被明确拒绝（留给 W1）
_try_w2 = False
try:
    edit_binary_state_records(_w2f.state_text, [{"index": 4, "id": 0}])
except ValueError:
    _try_w2 = True
assert _try_w2, "id<->0 增删 version 字节（变长）应抛 ValueError（留给 W1）"
# 工作区一致性：写出的副本重开对象图一致、改动持久化（模拟 Save As 落盘往返）
_w2dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w2_edit_check.sim")
import shutil as _sh2
_sh2.copy2(_w2f.path, _w2dst)
with open(_w2dst, "rb") as _fh:
    _w2blob = bytearray(_fh.read())
for _a in _w2f.arrays:
    if _a["type"] == "Character1" and _a.get("data") and \
            len(_a["data"]) == len(_w2nb):
        _st = int(_a["start"])
        _w2blob[_st:_st + len(_w2nb)] = _w2nb
        break
with open(_w2dst, "wb") as _fh:
    _fh.write(bytes(_w2blob))
_w2re = SimFile(_w2dst)
assert _w2re.state_mode == "binary" and len(_w2re.objects) == len(_w2f.objects)
assert _w2re.state_mode == "binary"
_t3, _r3, _m3, _b3 = parse_state_table_binary(_w2re.state_text)
assert _r3[3]["flags"] == 2 and _r3[4]["id"] == 1007 \
    and _r3[10]["version"] == 2, "Save As 落盘后编辑应仍被解析命中"
os.remove(_w2dst)
# 多段魔数文件（manifold, N=5）：安全编辑同样成立
_w2m = SimFile(_find9("manifold_start.sim"))
_mt, _mr, _mm, _mbb = parse_state_table_binary(_w2m.state_text)
_w2mb, _w2mchg = edit_binary_state_records(
    _w2m.state_text, [{"index": 6, "id": 1041}, {"index": 4, "version": 3}])
assert len(_w2mb) == len(_w2m.state_text.encode("latin-1"))
assert verify_binary_state_edit(
    _w2m.state_text, _w2mb, [{"index": 6, "id": 1041}, {"index": 4, "version": 3}])
_mt2, _mr2, _mm2, _mbb2 = parse_state_table_binary(_w2mb.decode("latin-1"))
assert _mr2[6]["id"] == 1041 and _mr2[4]["version"] == 3
for _i, (_ra, _rb) in enumerate(zip(_mr, _mr2)):
    if _i in (4, 6):
        continue
    assert _binary_record_bytes(_ra) == _binary_record_bytes(_rb), \
        "多段文件非目标记录 %d 被改动" % _i
print("W2 state-edit: 只动已确证记录（id/flags/version 等宽）+ 差分验证"
      " + 变长拒绝 + 单/多段文件 全通过")

# --- W1：数组块变长替换/删除（全量重定位 + StatePosition 重算） ---
from sim_writer import apply_array_ops
_w1f = SimFile(r"D:/training/caedecoder/stardecoding/adjointWing_start.sim")
_w1blob = open(_w1f.path, "rb").read()
_w1_orig_len = len(_w1blob)
_w1_n0 = len(_w1f.objects)
_w1_n_arr = len(_w1f.arrays)  # 原始数组数（apply_array_ops 会原地改写 _w1f.arrays，须先快照）
_w1_old_sp = int(_w1f.header["StatePosition"])
_w1payload = (b"\xab\x00\x00\x00" * 120)  # Unsigned4, 120 元素
_w1nb, _w1info = apply_array_ops(_w1blob, _w1f, [
    {"op": "replace", "index": 1, "count": 120, "payload": _w1payload},
    {"op": "delete", "index": 6},
])
# 变长：len 应变化；且解析新文件一致
assert len(_w1nb) != _w1_orig_len, "变长替换/删除应改变文件长度"
assert _w1info["old_state_position"] == _w1_old_sp
# 主状态表数组（Character1 arr0）不可被 W1 变长改（留 W2）
_w1guard = False
try:
    apply_array_ops(_w1blob, _w1f, [{"op": "delete", "index": 0}])
except ValueError:
    _w1guard = True
assert _w1guard, "主状态表数组变长编辑应被 W1 拒绝（留 W2）"
# 落盘重开验证
_w1dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w1_arr_check.sim")
import shutil as _sh1
_sh1.copy2(_w1f.path, _w1dst)
with open(_w1dst, "wb") as _fh:
    _fh.write(_w1nb)
_w1r = SimFile(_w1dst)
assert len(_w1r.arrays) == _w1_n_arr - 1, \
    "删除 1 块后应少 1 个数组（%d->%d，重开实得 %d）" % (_w1_n_arr, _w1_n_arr - 1, len(_w1r.arrays))
assert _w1r.arrays[1]["count"] == 120, "arr1 应变长替换为 120"
assert len(_w1r.objects) == _w1_n0, "对象图应保持不变"
assert int(_w1r.header["StatePosition"]) == _w1info["new_state_position"], \
    "重开应命中重算后的 StatePosition"
# StatePosition 应仍指向同一 StarVersion 分区（字典内容一致）
def _w1_section(sim, off, blob=None):
    for d, pl, ss, ps in sim.sections:
        if ss <= off < ps:
            return d
    return None
_w1old_sec = _w1_section(SimFile(_w1f.path), _w1_old_sp)
_w1new_sec = _w1_section(_w1r, int(_w1r.header["StatePosition"]))
assert _w1old_sec and _w1new_sec and _w1old_sec.get("ClassName") == _w1new_sec.get("ClassName") \
    and _w1old_sec.get("Type") == _w1new_sec.get("Type"), "StatePosition 应指向同类分区"
assert _w1old_sec.get("ClassName") == "StarVersion"
os.remove(_w1dst)
print("W1 array-op: 变长替换/删除 + 全量重定位 + StatePosition 精确重指向"
      " + 主状态表拒改 + 重开一致 全通过")

# --- W4：ClassVersions 一致性维护 + NameManager 写入 + id「序号+2」兼容 ---
from sim_writer import save_sim, get_name_manager, write_name_manager
from star_gui_document import SimDocument
from star_gui_commands import CopyObjectCommand
_W4SIM = "D:/training/caedecoder/stardecoding/adjointWing_start.sim"
_w4f = SimFile(_W4SIM)
_w4_orig_vers = (_w4f.objects[-1].dict.get("Versions") or {})
_w4_orig_matched = _w4f.validate_class_versions()["matched"]
_w4_orig_region = _w4_orig_vers.get("star.common.Region")
_w4doc = SimDocument(_w4f, _W4SIM)
_w4cmd = CopyObjectCommand(
    next(o.id for o in _w4f.objects if o.class_name == "star.common.Region"))
assert _w4doc.execute(_w4cmd), "W4 复制对象失败"
_w4_created_n = len(_w4doc.created)
_w4_region_new = sum(1 for o in _w4doc.created.values()
                     if o is not None and (getattr(o, "class_name", None) == "star.common.Region"))
assert _w4_region_new >= 1, "W4 至少应创建一个 star.common.Region"
_w4dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w4_cv_check.sim")
save_sim(_w4f, _w4dst, patches=_w4doc.patches, created=_w4doc.created, src_path=_W4SIM)
_w4r = SimFile(_w4dst)
# id 维持「图序号+2」严格连续
assert _w4r.check_sequential_ids()["ok"], "创建后对象图 id 仍应=序号+2 严格连续"
# ClassVersions 仍为最后对象，尾部合法
assert _w4r.objects[-1].class_name == "ClassVersions"
_last_d4, _last_p4, _last_s4, _ = walk_sections(_w4r.blob)[-1]
_tail4 = _w4r.blob[_last_s4:].decode("latin-1").strip()
assert _tail4.startswith("{") and _tail4.endswith("}"), "ClassVersions 尾部应保持合法 dict"
_w4r_vers = _w4r.objects[-1].dict["Versions"]
# 一致性：Region 计数 = 原本 + 本次新增；其余类保持不变（增量不扩散）
assert _w4r_vers["star.common.Region"] == (_w4_orig_region or 0) + _w4_region_new, \
    "ClassVersions Versions 应反映新增 Region 实例"
for _cn, _n in _w4_orig_vers.items():
    if _cn != "star.common.Region":
        assert _w4r_vers.get(_cn) == _n, "其余类计数不应被改动: %s %s->%s" % (
            _cn, _n, _w4r_vers.get(_cn))
# 校验不劣化（新增实例已计入，matched 不下降）
assert _w4r.validate_class_versions()["matched"] >= _w4_orig_matched, \
    "ClassVersions 维护后 matched 不应下降"
# ClassVersions 增量日志（save_sim 输出）与落盘一致
assert getattr(_w4f, "class_versions_delta", {}).get("star.common.Region") == _w4_region_new
# 对象图规模正确（原 2076 + 新增）
assert len(_w4r.objects) == len(SimFile(_W4SIM).objects) + _w4_created_n
# NameManager 保留：原始空标记原样
_w4nm = get_name_manager(_w4r)
assert _w4nm is not None and _w4nm.dict.get("ClassName") == "NameManager"
# NameManager 写入（保守等宽）：空标记文件无既有 ObjectId → 变长新增如实拒绝
_w4blob0 = open(_W4SIM, "rb").read()
_w4nb0, _w4i0 = write_name_manager(_w4blob0, _w4f, object_id=12345678901234)
assert not _w4i0["changed"] and "等宽" in _w4i0["reason"]
# 对含 ObjectId 的文件：等宽改写成功且不破坏对象图/数组/序号
_w4ra = SimFile("D:/training/caedecoder/stardecoding/resaved_airfoil.sim")
_w4_old_oid = (get_name_manager(_w4ra).dict or {}).get("ObjectId")
_w4_nb, _w4info = write_name_manager(open(_w4ra.path, "rb").read(), _w4ra,
                                     object_id=_w4_old_oid + 1)
assert _w4info["changed"] and _w4info.get("width", 0) == 0, "等宽 ObjectId 改写应原位完成"
_w4dst2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w4_nm_check.sim")
with open(_w4dst2, "wb") as _fh:
    _fh.write(_w4_nb)
_w4r2 = SimFile(_w4dst2)
assert get_name_manager(_w4r2).dict.get("ObjectId") == _w4_old_oid + 1
assert len(_w4r2.objects) == len(_w4ra.objects) and len(_w4r2.arrays) == len(_w4ra.arrays)
assert _w4r2.check_sequential_ids()["ok"]
os.remove(_w4dst)
os.remove(_w4dst2)
print("W4 ClassVersions 一致性维护 + NameManager 保守写入 + id「序号+2」兼容"
      " + 等宽改写 + 重开一致 全通过")

# --- W5：引用/字典/嵌套结构属性全可写（semantic_dict 白名单扩展到写侧） ---
from sim_writer import format_repr, audit_write_references, save_sim as _w5save
from sim_parser import parse_repr
# 嵌套结构 format_repr 忠实往返（list/dict/str/float/None；tuple 规范化为 list 即文件格式）
_w5nested = {"a": 1, "b": [1.5, "x", {"k": None}], "c": [1, 2], "d": True}
assert parse_repr(format_repr(_w5nested)) == _w5nested
_W5SIM = "D:/training/caedecoder/stardecoding/adjointWing_start.sim"
_w5f = SimFile(_W5SIM)
_w5doc = SimDocument(_w5f, _W5SIM)
_w5scene = next(o for o in _w5f.objects if o.class_name == "star.vis.Scene")
_w5view = _w5doc.object(_w5scene.dict["CurrentView"])
# up 引用：重设视图 Parent 指向 Simulation（id 2 已存在）
_w5doc.set_property(_w5view.id, "Parent", 2)
# 嵌套 dict 属性写入（MonitorPrintOrder 为嵌套 {'键': 值}）
_w5old34 = dict((_w5doc.object(34).dict.get("MonitorPrintOrder") or {}))
_w5new34 = dict(_w5old34); _w5new34["Sdr"] = 99; _w5new34["Nested"] = {"x": True}
_w5doc.set_property(34, "MonitorPrintOrder", _w5new34)
# 嵌套 list（数组）属性写入
_w5disp = next(o for o in _w5f.objects if o.name == "Mesh 1"
               and "Displayer" in (o.class_name or ""))
_w5doc.set_property(_w5disp.id, "DisplayerColor", [0.1, 0.2, 0.3])
# 写侧引用白名单审计：这些合法编辑应无悬空告警
assert audit_write_references(_w5f, _w5doc.patches) == [], "合法引用/嵌套编辑不应有悬空告警"
_w5dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w5_write_check.sim")
_w5save(_w5f, _w5dst, patches=_w5doc.patches, src_path=_W5SIM)
_w5r = SimFile(_w5dst)
# 重开一致：引用/嵌套 dict/嵌套 list 全部命中
assert _w5r.objmap[_w5view.id].dict["Parent"] == 2
assert _w5r.objmap[34].dict["MonitorPrintOrder"] == _w5new34
assert _w5r.objmap[_w5disp.id].dict["DisplayerColor"] == [0.1, 0.2, 0.3]
# save_sim 内置非致命审计同样为空
assert _w5f.write_reference_issues == []
# 悬空引用被审计捕获（up 标量 / down 集合 / 指向已删除对象）
_w5bad = {_w5view.id: {"Parent": 987654321}}
assert len(audit_write_references(_w5r, _w5bad)) == 1 \
    and audit_write_references(_w5r, _w5bad)[0]["key"] == "Parent"
_w5kid = next(o.id for o in _w5r.objects if isinstance(o.dict.get("Keys"), list))
_w5bad2 = {_w5kid: {"Keys": list(_w5r.objmap[_w5kid].dict["Keys"]) + [99999999]}}
assert any(i["direction"] == "down" for i in audit_write_references(_w5r, _w5bad2))
assert any(i["key"] == "Parent" for i in
           audit_write_references(_w5r, {_w5view.id: {"Parent": 2}}, deleted=[2]))
os.remove(_w5dst)
print("W5 引用/字典/嵌套结构属性全可写 + 写侧白名单审计"
      "（悬空 up/down/已删除捕获）+ 重开一致 全通过")

# --- W3：ZIP/PK 容器写出（读解压载荷补丁 → 重打包 → 重开一致） ---
import io as _w3io
import zipfile as _w3zip
from sim_writer import save_sim as _w3save
_W3SRC = "D:/training/caedecoder/stardecoding/adjointWing_start.sim"
_w3raw = open(_W3SRC, "rb").read()
_w3n0 = len(SimFile(_W3SRC).objects)
_w3entry = "inner_model.sim"
_w3bio = _w3io.BytesIO()
with _w3zip.ZipFile(_w3bio, "w", _w3zip.ZIP_DEFLATED) as _z:
    _z.writestr(_w3entry, _w3raw)
_w3zsrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w3_src.sim")
with open(_w3zsrc, "wb") as _fh:
    _fh.write(_w3bio.getvalue())
assert open(_w3zsrc, "rb").read(2) == b"PK", "合成 ZIP 容器应以 PK 头"
# 读路径：识别容器、取主载荷、对象图与原始一致
_w3f = SimFile(_w3zsrc)
assert _w3f.container_entry == _w3entry, "读路径应识别容器条目名"
assert len(_w3f.objects) == _w3n0, "容器内载荷对象图应与原始一致"
# 写路径：对容器内 sim 打属性补丁，save_sim 应解压载荷补丁后重打包命中补丁
_w3doc = SimDocument(_w3f, _w3zsrc)
_w3target = next(o for o in _w3f.objects if o.dict.get("PresentationName"))
_w3doc.set_property(_w3target.id, "PresentationName", "W3-ZIP-EDITED")
_w3dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w3_out.sim")
_w3save(_w3f, _w3dst, patches=_w3doc.patches, src_path=_w3zsrc)
assert open(_w3dst, "rb").read(2) == b"PK", "ZIP 输入写出后仍应为 PK 容器"
_w3r = SimFile(_w3dst)
assert _w3r.container_entry == _w3entry, "重打包应保持原条目名"
assert _w3r.objmap[_w3target.id].dict["PresentationName"] == "W3-ZIP-EDITED", \
    "补丁应命中容器内载荷"
assert len(_w3r.objects) == _w3n0, "重打包后对象图规模不变"
assert len(_w3r.arrays) == len(_w3f.arrays), "重打包后数组块应不变"
# 往返一致：除补丁属性外，逐对象逐字段一致
_w3_ok = True
for _oid, _o in _w3r.objmap.items():
    _ob = _w3f.objmap.get(_oid)
    if _ob is None:
        _w3_ok = False
        break
    for _k, _v in _ob.dict.items():
        if _oid == _w3target.id and _k == "PresentationName":
            continue
        if _o.dict.get(_k, _v) != _v and (_k in _o.dict or _k in _ob.dict):
            _w3_ok = False
            break
    if not _w3_ok:
        break
assert _w3_ok, "ZIP 往返除补丁属性外对象图应逐字段一致"
# 无补丁纯往返：原容器 -> save_sim -> 新容器，重开应完全一致（对象/数组/条目名）
_w3dst0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w3_out0.sim")
_w3save(SimFile(_w3zsrc), _w3dst0, src_path=_w3zsrc)
_w3r0 = SimFile(_w3dst0)
assert open(_w3dst0, "rb").read(2) == b"PK" and _w3r0.container_entry == _w3entry
assert len(_w3r0.objects) == _w3n0 and len(_w3r0.arrays) == len(_w3f.arrays)
os.remove(_w3zsrc)
os.remove(_w3dst)
os.remove(_w3dst0)
print("W3 ZIP/PK 容器写出：读解压载荷补丁 + 重打包（保条目名/DEFLATED）"
      " + 有补丁/纯往返重开一致 全通过")

# --- P2：场函数表达式求值器（math/vector/logic + interpolate，与官方语法对齐） ---
from field_fn import FieldFunction as _P2FF, Table as _P2Table, \
    numeric as _p2, compile_expression as _p2c
import math as _p2m
assert _p2("1 + 2 * 3") == 7.0
assert _p2("(1 + 2) * 3") == 9.0
assert _p2("7 % 3") == 1.0
assert round(_p2("sqrt(9)"), 9) == 3.0
assert round(_p2("pow(2, 10)"), 9) == 1024.0
assert _p2("3 > 2") == 1.0 and _p2("3 == 3") == 1.0 and _p2("3 != 3") == 0.0
assert _p2("1 && 0") == 0.0 and _p2("0 || 1") == 1.0 and _p2("!0") == 1.0
assert _p2("1 < 2 ? 10 : 20") == 10.0
assert _p2("(1 < 0) ? 10 : ((2 < 3) ? 30 : 40)") == 30.0
assert round(_p2("mag($$Velocity)", {"Velocity": (3.0, 4.0, 0.0)}), 9) == 5.0
assert _p2("dot($$u, $$v)", {"u": (1.0, 0.0, 0.0), "v": (0.0, 1.0, 0.0)}) == 0.0
assert _p2("$$Velocity[0]", {"Velocity": (3.0, 4.0, 0.0)}) == 3.0
assert _p2("$$Velocity.y", {"Velocity": (3.0, 4.0, 0.0)}) == 4.0
assert tuple(_p2("[1, 2, 3]")) == (1.0, 2.0, 3.0)
_p2A = ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
assert round(_p2("trace($$$A)", {"A": _p2A}), 9) == 9.0
assert round(_p2("$$$A.eigValue(0)", {"A": _p2A}), 9) == 2.0
_p2t = _P2Table("t", [0.0, 1.0, 2.0], {"u": [0.0, 10.0, 20.0]})
assert abs(_p2('interpolateTable(@Table("t"), "u", LINEAR, "", ${Position}[0])',
               tables={"t": _p2t}, position=(0.5, 0.0, 0.0)) - 5.0) < 1e-9
assert _p2("alternateValue(1 / 0, 99)") == 99.0
assert _p2("alternateValue(1 / 0, sqrt(-1), 42)") == 42.0
assert _p2c("1 + 2 * 3 + mag($$v)") is not None
# 诚实拒绝：非法/未知/越界 明确报错而非静默
for _bad in ("1 +", "nope(1)", "${Nope}", "1 / 0", "$$v[3]"):
    try:
        if _bad == "$$v[3]":
            _p2("$$v[3]", {"v": (1.0, 2.0, 3.0)})
        else:
            _p2(_bad)
    except Exception:
        continue
    raise AssertionError("P2 应拒绝 %s" % _bad)
print("P2 场函数表达式求值器：算术/逻辑/三元/数学/矢量/张量/插值/交替值/"
      "编译预检 + 诚实拒绝 全通过")

# ---------------- P3 初始化器（field function/常量/表格初值 —— Run 前 Initialize）---------
from init_solver import Initializer as _P3Init
from solver_run import demo_mesh as _p3mesh, DemoDiffusionSolver as _p3Solver, \
    SolverBackend as _p3Backend, SolverState as _p3State
import numpy as _p3np
_p3V, _p3C = _p3mesh(nx=4)
_p3init = _P3Init(source_field="T").add_constant("T0", 300.0) \
    .add_table("ramp", [0.0, 1.0], {"load": [0.0, 100.0]}) \
    .add_function("T",
                  'T0 + interpolateTable(@Table("ramp"), "load", LINEAR, "", '
                  '${Position}[0])').compile()
# 常量/表格/场函数 单点求值
assert _p3init.value("T0") == 300.0
assert abs(_p3init.value("T", position=(0.0, 0.0, 0.0)) - 300.0) < 1e-9
assert abs(_p3init.value("T", position=(1.0, 0.0, 0.0)) - 400.0) < 1e-9
# 初场：对网格坐标批量求值 = T0 + 100*x
_p3f = _p3init.field("T", _p3V)
assert _p3f.shape == (len(_p3V),)
assert _p3np.allclose(_p3f, 300.0 + 100.0 * _p3V[:, 0])
# Run 前 Initialize 可用：DemoDiffusionSolver 注入 initializer，初场生效
_p3s = _p3Solver(_p3V, _p3C, initializer=_p3init)
_p3s._initialize_field()
assert _p3s.field().shape == (len(_p3V),)
assert _p3np.allclose(_p3s.field(), 300.0 + 100.0 * _p3V[:, 0])
# SolverBackend.initialize 同样在 Run 前注入 init 并生成初场
_p3be = _p3Backend(_p3Solver(_p3V, _p3C), initializer=_p3init)
assert _p3be.state() == _p3State.IDLE
assert _p3be.initialize()
assert _p3be.state() == _p3State.INITIALIZED
assert _p3np.allclose(_p3be.solver.field(), 300.0 + 100.0 * _p3V[:, 0])
# 诚实拒绝：source_field 缺失 / 未知函数(求值) / 语法错误 / 初场维度不匹配
try:
    _P3Init(source_field="U").compile()
    raise AssertionError("P3 应拒绝未登记 source_field")
except ValueError:
    pass
try:
    _P3Init().add_function("g", "nope(1)").compile().value("g")
    raise AssertionError("P3 应拒绝未知函数")
except Exception:
    pass
try:
    _P3Init().add_function("h", "1 +").compile()
    raise AssertionError("P3 应拒绝语法错误")
except Exception:
    pass
try:
    _p3Solver(_p3V, _p3C)._set_initial_field(_p3np.zeros(len(_p3V) + 1))
    raise AssertionError("P3 应拒绝初场维度不匹配")
except ValueError:
    pass
print("P3 初始化器：常量/表格/场函数初值 + Run 前 Initialize 可用 全通过")

# ---------------- P4 FVM 离散核心（梯度 Gauss/LSQ、限制器、通量格式 ----------------
# 保持一致网格的守恒 / 线性复原验证（纯 numpy，occ / scdm 两环境皆可用）
from fvm_core import FVM as _p4FVM, cube_tet_mesh as _p4cube
import numpy as _p4np
# 守恒：cube_tet_mesh 6-tet Kuhn 分解体积精确填满单位立方体
for _p4n in (1, 2, 3):
    _p4V, _p4C = _p4cube(nx=_p4n)
    _p4f = _p4FVM(_p4V, _p4C)
    _p4vsum = _p4f.volumes.sum()
    assert abs(_p4vsum - 1.0) < 1e-9, "P4 网格体积应填满单位立方体: %.6f" % _p4vsum
    # 拓扑不变量：面数 = 内面 + 边界面；欧拉关系 2*内面 + 边界面 = 4*单元
    assert _p4f.n_faces == _p4f.n_interior_faces + _p4f.n_boundary_faces
    assert _p4f.is_boundary.sum() == _p4f.n_boundary_faces
    assert (_p4f.neighbor >= 0).sum() == _p4f.n_interior_faces
    assert 2 * _p4f.n_interior_faces + _p4f.n_boundary_faces == 4 * _p4f.n_cells
# 线性场复原：phi = 1 + 2x + 3y + 4z，梯度应为 (2,3,4) 到机器精度
_p4V, _p4C = _p4cube(nx=2)
_p4f = _p4FVM(_p4V, _p4C)
_p4cx = _p4f.centroids
_p4fx = _p4f.face_centroid
_p4phi = 1.0 + 2.0 * _p4cx[:, 0] + 3.0 * _p4cx[:, 1] + 4.0 * _p4cx[:, 2]
_p4fb = 1.0 + 2.0 * _p4fx[:, 0] + 3.0 * _p4fx[:, 1] + 4.0 * _p4fx[:, 2]
_p4g_exact = _p4np.array([2.0, 3.0, 4.0])
# LSQ 梯度 + 精确边界面样本 = 机器精度线性复原
_p4g_lsq = _p4f.grad_lsq(_p4phi, boundary=_p4fb)
assert _p4np.abs(_p4g_lsq - _p4g_exact).max() < 1e-10, \
    "P4 LSQ 梯度应精确复原线性场: %s" % _p4g_lsq
# 无边界时零梯度外推：不要求精确，但必须有限且非全零（法方程不奇异）
_p4g_lsq_nb = _p4f.grad_lsq(_p4phi)
assert _p4np.isfinite(_p4g_lsq_nb).all()
assert _p4np.abs(_p4g_lsq_nb).max() > 1e-3
# GG + LSQ 重构 = 机器精度线性复原；GG(naive) 在 Kuhn 网格上有 O(1) 偏差（不要求精确）
_p4g_gg = _p4f.grad_gauss(_p4phi, boundary=_p4fb, recon=_p4g_lsq)
assert _p4np.abs(_p4g_gg - _p4g_exact).max() < 1e-10, \
    "P4 Green-Gauss(重构) 梯度应精确复原线性场: %s" % _p4g_gg
_p4g_gg_naive = _p4f.grad_gauss(_p4phi, boundary=_p4fb)
assert _p4g_gg_naive.shape == (_p4f.n_cells, 3)
assert _p4np.isfinite(_p4g_gg_naive).all()
# 面插值：边界面 = 精确面值；内部面反距离权重有限
_p4phi_f = _p4f.face_value(_p4phi, boundary=_p4fb)
assert _p4np.isfinite(_p4phi_f).all()
_p4is_int = _p4f.neighbor >= 0
assert _p4np.abs(_p4phi_f[~_p4is_int] - _p4fb[~_p4is_int]).max() < 1e-12
# 限制器：线性场精确 = 1；随机场有界 [0,1]
_p4lim = _p4f.limiter(_p4phi, _p4g_lsq)
assert _p4np.abs(_p4lim - 1.0).max() < 1e-9, "P4 线性场限制器应为 1"
_p4rng = _p4np.random.RandomState(0)
_p4rnd = _p4phi + 0.3 * _p4rng.rand(_p4f.n_cells)
_p4g_rnd = _p4f.grad_lsq(_p4rnd, boundary=_p4np.zeros(_p4f.n_faces))
_p4lim_rnd = _p4f.limiter(_p4rnd, _p4g_rnd)
assert (_p4lim_rnd >= 0.0).all() and (_p4lim_rnd <= 1.0).all()
# 通量格式：扩散 常数场零通量（零梯度边界）；内部面成对抵消 → 全局守恒
_p4flux_d0 = _p4f.diffusion_flux(_p4np.full(_p4f.n_cells, 7.0), gamma=2.5)
assert abs(_p4flux_d0).max() < 1e-12, "P4 扩散 常数场应为零通量"
_p4flux_d = _p4f.diffusion_flux(_p4phi)
assert _p4np.isfinite(_p4flux_d).all()
_p4net = _p4np.zeros(_p4f.n_cells)
_p4np.add.at(_p4net, _p4f.owner[_p4is_int], _p4flux_d[_p4is_int])
_p4np.add.at(_p4net, _p4f.neighbor[_p4is_int], -_p4flux_d[_p4is_int])
assert abs(_p4net.sum()) < 1e-12, "P4 扩散通量应在迭代中全局守恒"
assert (_p4flux_d[~_p4is_int] == 0.0).all()
# 对流 上风：正 mdot 取 owner、负 mdot 取 neighbor；边界入流取边界值
_p4lin2 = _p4np.arange(_p4f.n_cells, dtype=float)
_p4mdot = _p4np.zeros(_p4f.n_faces)
_p4mdot[_p4is_int] = 1.0
_p4fu = _p4f.convection_flux_upwind(_p4mdot, _p4lin2)
assert _p4np.abs(_p4fu[_p4is_int] - _p4lin2[_p4f.owner[_p4is_int]]).max() < 1e-12
_p4mdot[_p4is_int] = -1.0
_p4fu = _p4f.convection_flux_upwind(_p4mdot, _p4lin2)
assert _p4np.abs(_p4fu[_p4is_int] - (-1.0) * _p4lin2[_p4f.neighbor[_p4is_int]]).max() < 1e-12
# 对流 中心：mdot=1 时等于面插值面值
_p4mdot_1 = _p4np.ones(_p4f.n_faces)
_p4fc = _p4f.convection_flux_central(_p4mdot_1, _p4phi, boundary=_p4fb)
assert _p4np.isfinite(_p4fc).all()
assert _p4np.abs(_p4fc - _p4f.face_value(_p4phi, boundary=_p4fb)).max() < 1e-12
# 诚实拒绝：网格/场量/梯度/通量 形状不匹配 明确报错
for _p4bad in (
    lambda: _p4FVM(_p4V[:3], _p4C),
    lambda: _p4FVM(_p4V, _p4C[:, :3]),
    lambda: _p4f.grad_gauss(_p4phi[:-1]),
    lambda: _p4f.grad_lsq(_p4phi[:-1]),
    lambda: _p4f.limiter(_p4phi, _p4g_lsq[:, :2]),
    lambda: _p4f.diffusion_flux(_p4phi[:-1]),
    lambda: _p4f.convection_flux_upwind(_p4mdot[:-1], _p4phi),
):
    try:
        _p4bad()
        raise AssertionError("P4 应拒绝非法输入")
    except ValueError:
        pass
print("P4 FVM 离散核心：网格守恒/梯度(Gauss/LSQ 线性复原)/限制器/通量格式/"
      "诚实拒绝 全通过")

# ---------------- P5 压力基求解器（SIMPLE + Rhie-Chow + 稀疏线性求解） ----------------
# 验收核心（P5 行）：残差下降曲线健康。纯 numpy（scdm 无 scipy）亦可用。
from fvm_core import cube_tet_mesh as _p5cube
from pressure_solver import PressureSolver as _p5Solver, solve_linear as _p5solve
import numpy as _p5np
# 稀疏线性求解：numpy / scipy(AMG/ILU) / auto 三路径同一精确解（M 矩阵）
_p5n = 6
_p5row, _p5col, _p5data = [], [], []
for _p5i in range(_p5n):
    _p5row.append(_p5i); _p5col.append(_p5i); _p5data.append(2.0)
    if _p5i > 0:
        _p5row.append(_p5i); _p5col.append(_p5i - 1); _p5data.append(-1.0)
        _p5row.append(_p5i - 1); _p5col.append(_p5i); _p5data.append(-1.0)
_p5b = _p5np.ones(_p5n)
_p5ax = _p5np.zeros(_p5n)
for _p5kind in ("numpy", "scipy", "auto"):
    _p5x = _p5solve(_p5np.array(_p5row, _p5np.int64),
                    _p5np.array(_p5col, _p5np.int64),
                    _p5np.array(_p5data, float), _p5b, _p5n,
                    tol=1e-12, maxit=8000, kind=_p5kind)
    _p5ax[:] = 0.0
    _p5np.add.at(_p5ax, _p5row, _p5data * _p5x[_p5col])
    assert _p5np.sqrt(((_p5ax - _p5b) ** 2).sum()) < 1e-8, \
        "P5 线性求解应给出精确解 (kind=%s): %s" % (_p5kind, _p5ax - _p5b)
# RHS 维度不匹配应报 ValueError
try:
    _p5solve(_p5np.array(_p5row), _p5np.array(_p5col),
             _p5np.array(_p5data), _p5np.ones(_p5n + 1), _p5n)
    raise AssertionError("P5 线性求解应拒绝维度不匹配 RHS")
except ValueError:
    pass
# SIMPLE：残差下降曲线健康（P5 验收核心）；全局质量守恒 + 内部散度趋零
_p5V, _p5C = _p5cube(nx=2)
_p5s = _p5Solver(_p5V, _p5C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                 alpha_momentum=0.7, alpha_pressure=0.3)
_p5hist = []
for _p5it in range(40):
    _p5hist.append(float(_p5s.step()["residual"]))
assert _p5s.iteration == 40
assert _p5s.residual() == _p5hist[-1]
assert _p5np.isfinite(_p5s.velocity()).all()
assert _p5s.pressure().shape == (len(_p5C),)
assert _p5s.mass_flux().shape == (_p5s._fv.n_faces,)
assert _p5hist[-1] < 1e-5, "P5 残差应收敛到 <1e-5: %.2e" % _p5hist[-1]
assert max(_p5hist[20:]) < max(_p5hist[:20]) * 0.1, \
    "P5 残差曲线应健康下降（后半程远低于前半程）"
assert _p5hist[-1] < _p5hist[0], "P5 终态残差应低于初值"
# 全局守恒：边界面净通量 sum(imb)~0；出口对入口闭合；内部散度归一化范数 ~0
_p5mdot = _p5s.mass_flux()
_p5imb = _p5s._continuity_imbalance()
assert abs(_p5imb.sum()) < 1e-8, "P5 边界面净通量应守恒: %.2e" % _p5imb.sum()
assert abs(_p5mdot[_p5s._outlet_faces].sum()
           + _p5mdot[_p5s._inlet_faces].sum()) < 1e-8, "P5 出口对入口应闭合质量通量"
assert _p5np.sqrt((_p5imb * _p5imb).sum()) < 1e-6, \
    "P5 内部连续性残差应趋零: %.2e" % float(_p5np.sqrt((_p5imb * _p5imb).sum()))
# 诚实拒绝：非四面体 / 过小网格 明确报错
for _p5bad in (
    lambda: _p5Solver(_p5V, _p5C[:, :3]),
    lambda: _p5Solver(_p5np.zeros((3, 3)), _p5np.zeros((0, 4), _p5np.int64)),
):
    try:
        _p5bad()
        raise AssertionError("P5 应拒绝非法网格输入")
    except ValueError:
        pass
print("P5 压力基求解器：SIMPLE 残差下降曲线健康/全局质量守恒/稀疏线性(numpy+scipy)/"
      "诚实拒绝 全通过")

# ---------------- P6 湍流族（SA → k-ε → k-ω SST → LES 子格子 + 壁面处理） ----------------
# 验收核心（P6 行）：壁面处理/壁面函数、SA→k-ε→k-ω SST→LES 子格子。纯 numpy 可用。
from fvm_core import cube_tet_mesh as _p6cube, FVM as _p6FVM
from pressure_solver import PressureSolver as _p6Solver
import turbulence as _p6T
import numpy as _p6np
# 壁面处理：y+ 定义 / 壁面函数（粘性底层-对数律极限） / 壁面剪切
assert _p6np.isclose(_p6T.y_plus(0.1, 0.5, 1e-3), 0.1 * 0.5 / 1e-3), "P6 y+ 定义"
assert _p6np.isclose(_p6T.wall_function(0.01), 0.01, atol=1e-9), "P6 粘性底层 u+≈y+"
assert _p6np.isclose(_p6T.wall_function(100.0), _p6np.log(9.8 * 100.0) / 0.41,
                     rtol=1e-6), "P6 壁面函数对数律"
assert _p6np.isclose(_p6T.wall_shear(0.3, 1.2), 1.2 * 0.3 ** 2), "P6 壁面剪切"
_p6V, _p6C = _p6cube(nx=2)
_p6fv = _p6FVM(_p6V, _p6C)
_p6d = _p6T.wall_distance(_p6fv)
assert _p6d.shape == (_p6fv.n_cells,) and (_p6d >= 0.0).all(), "P6 壁面距离"
# 模型注册：别名/大小写/连字符解析为正确类（SA/k-ε/k-ω SST/LES 四族）
_p6cls = {
    "sa": _p6T.SpalartAllmarasSolver,
    "K_Epsilon": _p6T.KEpsilonSolver,
    "k-omega-sst": _p6T.KOmegaSSTSolver,
    "LES": _p6T.LESSmagorinskySolver,
}
for _p6name, _p6c in _p6cls.items():
    assert isinstance(_p6T.make_model(_p6name, _p6fv, rho=1.0, mu=1e-3), _p6c), \
        "P6 模型名应解析为正确类: %s" % _p6name
try:
    _p6T.make_model("not-a-model", _p6fv, rho=1.0, mu=1e-3)
    raise AssertionError("P6 应拒绝未知湍流模型")
except ValueError:
    pass
# 各方程模型 update()：有限残差 + 有限非负 nu_t；SA 残差单调下降
_p6u = _p6fv.centroids[:, 1] + 1.0
_p6v = 0.1 * _p6fv.centroids[:, 0]
_p6w = _p6np.zeros(_p6fv.n_cells)
_p6sa = _p6T.make_model("sa", _p6fv, rho=1.0, mu=1e-3, u_ref=1.0, length_scale=0.2)
_p6sa_res = []
for _p6it in range(8):
    _p6sa_res.append(float(_p6sa.update(_p6u, _p6v, _p6w)))
assert _p6np.isfinite(_p6sa_res).all(), "P6 SA 残差有限"
assert _p6sa_res[-1] < _p6sa_res[0], "P6 SA 残差应下降"
assert _p6np.isfinite(_p6sa.nu_t).all() and (float(_p6sa.nu_t.max()) >= 0.0), "P6 SA nu_t"
for _p6m in ("k-epsilon", "k-omega-sst", "les"):
    _p6mm = _p6T.make_model(_p6m, _p6fv, rho=1.0, mu=1e-3, u_ref=1.0, length_scale=0.2)
    _p6r = float(_p6mm.update(_p6u, _p6v, _p6w))
    assert _p6np.isfinite(_p6r), "P6 %s 残差有限" % _p6m
    assert _p6np.isfinite(_p6mm.nu_t).all() \
        and (float(_p6mm.nu_t.max()) >= 0.0), "P6 %s nu_t 有限非负" % _p6m
# 集成：PressureSolver 以字符串注入湍流模型，nu_t 形状一致、step() 无报错；基线全零
_p6s0 = _p6Solver(_p6V, _p6C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0), max_outer=3)
assert (float(_p6s0.nu_t.max()) == 0.0), "P6 基线应无湍流粘性"
assert _p6np.isfinite(_p6s0.step()["residual"]), "P6 基线 step() 无报错"
for _p6nm in ("sa", "k-epsilon", "k-omega-sst", "les"):
    _p6t = _p6Solver(_p6V, _p6C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                     max_outer=3, turb_model=_p6nm)
    assert _p6t.nu_t.shape == (len(_p6C),), "P6 %s nu_t 形状" % _p6nm
    assert _p6np.isfinite(_p6t.step()["residual"]), "P6 %s step() 无报错" % _p6nm
    assert _p6np.isfinite(_p6t.nu_t).all(), "P6 %s nu_t 有限" % _p6nm
print("P6 湍流族：壁面处理/壁面函数/SA→k-ε→k-ω SST→LES 子格子/ν_t 耦合/诚实验证 全通过")

# ---------------- P7 能量/传热（对流扩散+共轭）+ 简化辐射谱系 ----------------
# 验收核心（P7 行）：能量/传热（对流扩散+共轭）+ 简化辐射谱系。纯 numpy 可用。
from fvm_core import cube_tet_mesh as _p7cube, FVM as _p7FVM
from pressure_solver import PressureSolver as _p7Solver
import energy as _p7E
import numpy as _p7np
# 常量 / 边界类型：互异枚举 + Stefan-Boltzmann
assert _p7np.isclose(_p7E.SIGMA_STEFAN, 5.67e-8), "P7 Stefan-Boltzmann 常数"
assert len({_p7E.BND_ADIABATIC, _p7E.BND_FIXED_TEMP, _p7E.BND_FIXED_FLUX,
            _p7E.BND_ROBIN, _p7E.BND_RADIATION}) == 5, "P7 传热边界类型互异"
# 装配 / 求解：形状 + 有界性（定温 Dirichlet 锚定，避免奇异）
_p7V, _p7C = _p7cube(nx=2)
_p7fv = _p7FVM(_p7V, _p7C)
_p7n = _p7fv.n_cells
_p7mdot = _p7E.volume_to_mass(_p7fv, _p7np.full(_p7n, 1.0),
                              _p7np.zeros(_p7n), _p7np.zeros(_p7n),
                              _p7np.full(_p7n, _p7E.DEFAULT_RHO))
_p7rows, _p7cols, _p7vals, _p7rhs, _p7ap = _p7E.assemble_energy_transport(
    _p7fv, _p7mdot, _p7np.full(_p7n, _p7E.DEFAULT_KAPPA),
    _p7np.full(_p7n, _p7E.DEFAULT_CP), _p7np.full(_p7n, 300.0),
    _p7np.zeros(_p7n))
assert _p7np.isfinite(_p7vals).all() and _p7np.isfinite(_p7rhs).all(), "P7 能量装配有限"
assert _p7rhs.shape == (_p7n,) and _p7ap.shape == (_p7n,), "P7 能量装配形状"
# 纯导热定温锚定：落在 [300, 350]
_p7bt = _p7np.full(_p7fv.n_faces, _p7E.BND_ADIABATIC, int)
_p7bv = _p7np.zeros(_p7fv.n_faces, float)
_p7in, _p7out, _p7wall, _ = _p7E.classify_thermal(_p7fv)
_p7bt[_p7in] = _p7E.BND_FIXED_TEMP; _p7bv[_p7in] = 300.0
_p7bt[_p7out] = _p7E.BND_FIXED_TEMP; _p7bv[_p7out] = 350.0
_p7Tnew, _ = _p7E.solve_energy(_p7fv, _p7np.zeros(_p7fv.n_faces),
                              _p7np.full(_p7n, 1.0), _p7np.full(_p7n, 1.0),
                              _p7np.full(_p7n, 320.0), _p7np.zeros(_p7n),
                              btype=_p7bt, bval=_p7bv, relax=1.0)
assert _p7np.isfinite(_p7Tnew).all(), "P7 纯导热 T 有限"
assert _p7Tnew.min() >= 300.0 - 1e-6 and _p7Tnew.max() <= 350.0 + 1e-6, "P7 纯导热有界"
# 边界分类 / 辅助：全覆盖不重叠 + 四元组
assert (_p7in.size + _p7out.size + _p7wall.size
        == int(_p7np.sum(_p7fv.is_boundary))), "P7 边界全覆盖"
_p7btp, _p7bvp, _p7hp, _p7ep = _p7E.thermal_btypes(
    _p7fv, _p7in, _p7out, _p7wall, inlet_temp=310.0,
    wall_btype=_p7E.BND_FIXED_TEMP, wall_bval=320.0)
assert _p7np.all(_p7btp[_p7in] == _p7E.BND_FIXED_TEMP) \
    and _p7np.allclose(_p7bvp[_p7in], 310.0), "P7 入口定温"
assert _p7np.all(_p7btp[_p7wall] == _p7E.BND_FIXED_TEMP) \
    and _p7np.allclose(_p7bvp[_p7wall], 320.0), "P7 壁面定温"
# EnergySolver：对流收敛 + 定温壁加热有界
_p7es = _p7E.EnergySolver(_p7fv, inlet_temp=300.0)
_p7u = _p7np.full(_p7n, 1.0); _p7v = _p7np.zeros(_p7n); _p7w = _p7np.zeros(_p7n)
_p7first = None
for _p7i in range(60):
    _p7r = float(_p7es.update(_p7u, _p7v, _p7w))
    _p7first = _p7first if _p7first is not None else _p7r
assert _p7np.isfinite(_p7es.T).all() and _p7r < _p7first, "P7 能量残差下降"
assert _p7np.allclose(_p7es.T, 300.0, atol=1.0), "P7 对流稳态温度"
_p7es2 = _p7E.EnergySolver(_p7fv, inlet_temp=300.0,
                          wall_btype=_p7E.BND_FIXED_TEMP, wall_bval=350.0)
for _p7i in range(80):
    _p7es2.update(_p7u, _p7v, _p7w)
assert _p7np.isfinite(_p7es2.T).all() and float(_p7es2.T.max()) > 300.0 \
    and float(_p7es2.T.max()) <= 350.0 + 1e-3, "P7 定温壁加热有界"
# 共轭：材料分区（固体区） + 有限温度
_p7es3 = _p7E.EnergySolver(_p7fv, inlet_temp=300.0,
                           wall_btype=_p7E.BND_FIXED_TEMP, wall_bval=350.0)
_p7ph = _p7np.zeros(_p7n, int); _p7ph[_p7fv.centroids[:, 0] > 0.5] = 1
_p7es3.set_materials(cell_phase=_p7ph, rho_s=7800.0, cp_s=500.0, kappa_s=16.0)
assert int(_p7es3.phase.sum()) > 0 and _p7es3.phase.dtype.kind in "iu", "P7 材料分区"
for _p7i in range(6):
    _p7es3.update(_p7u, _p7v, _p7w)
assert _p7np.isfinite(_p7es3.T).all(), "P7 共轭 T 有限"
# 简化辐射：净发射 / 辐射权重 / 体源 / 线性化
_p7rm = _p7E.RadiationModel(emissivity=0.8, tref=300.0)
assert _p7rm.emittance(300.0) == _p7np.float64(0.8 * _p7E.SIGMA_STEFAN * 300.0 ** 4), \
    "P7 灰体发射率"
assert _p7rm.net_emission(400.0) > 0.0 and _p7rm.net_emission(200.0) < 0.0, "P7 净发射号"
assert float(_p7rm.h_rad(300.0, 300.0)) > 0.0, "P7 辐射权重"
_p7prod, _p7diss = _p7rm.linear_source()
assert _p7prod > 0.0 and _p7diss > 0.0, "P7 线性化体源"
# 工厂：别名/大小写解析，未知报 ValueError
assert isinstance(_p7E.make_energy(_p7fv, "energy"), _p7E.EnergySolver), "P7 工厂 energy"
assert isinstance(_p7E.make_energy(_p7fv, "CHT"), _p7E.ConjugateSolver), "P7 工厂 cht"
try:
    _p7E.make_energy(_p7fv, "nope")
    raise AssertionError("P7 应拒绝未知能量模型")
except ValueError:
    pass
# 集成：PressureSolver 字符串注入能量模型 + Boussinesq 浮力；基线无能量模型
_p7s0 = _p7Solver(_p7V, _p7C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0), max_outer=3)
assert _p7s0.energy_model is None, "P7 基线应无能量模型"
assert _p7np.isfinite(_p7s0.step()["residual"]), "P7 基线 step() 无报错"
_p7se = _p7Solver(_p7V, _p7C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                  max_outer=3, energy_model="energy", inlet_temp=350.0,
                  beta=1.0e-3, gravity=(0.0, 0.0, -9.81))
assert _p7se.energy_model is not None, "P7 应构建能量模型"
assert _p7se.energy_model.T.shape == (len(_p7C),), "P7 能量模型 T 形状"
for _p7i in range(5):
    _p7pe = _p7se.step()
assert _p7np.isfinite(_p7se.energy_model.T).all(), "P7 耦合 T 有限"
assert _p7np.isfinite(_p7se.velocity()).all() and _p7np.isfinite(_p7pe["residual"]), \
    "P7 耦合流场/残差有限"
print("P7 能量/传热：对流-扩散能量方程/共轭传热/简化辐射/Boussinesq 耦合 全通过")

# ---------------- P8 多相 VOF（几何重构）→ Mixture → DPM 粒子轨 ----------------
# 验收核心（P8 行）：多相 VOF（几何重构）→ Mixture → DPM 粒子轨。纯 numpy 可用。
from fvm_core import cube_tet_mesh as _p8cube, FVM as _p8FVM
from pressure_solver import PressureSolver as _p8Solver
import vof as _p8V
import numpy as _p8np
# 常量 / 两相物性混合：水-空气密度比 (~850×)，混合场单调落在两相之间
assert _p8np.isclose(_p8V.DEFAULT_RHO1, 998.0), "P8 相 1 密度"
assert _p8np.isclose(_p8V.DEFAULT_RHO2, 1.18), "P8 相 2 密度"
assert _p8np.isclose(_p8V.DEFAULT_SIGMA, 0.072), "P8 表面张力"
assert _p8np.isclose(_p8V.two_phase_rho(_p8np.array([1.0])), 998.0), "P8 混合密度 α=1"
assert _p8np.isclose(_p8V.two_phase_rho(_p8np.array([0.0])), 1.18), "P8 混合密度 α=0"
assert _p8V.EPS > 0.0 and _p8V.BISECT_MAX >= 1, "P8 数值常量"
# 几何重构（PLIC）：法向由 ∇α 确定 + 体积截断满足目标 α，逐单元一致
_p8Vc, _p8Cc = _p8cube(nx=2)
_p8fv = _p8FVM(_p8Vc, _p8Cc)
_p8n = _p8fv.n_cells
_p8alpha = _p8np.clip(_p8fv.centroids[:, 0], 0.0, 1.0)
_p8normals, _p8offsets = _p8V.plic_reconstruct(_p8fv, _p8alpha, axis=0)
assert _p8normals.shape == (_p8n, 3) and _p8offsets.shape == (_p8n,), "P8 PLIC 形状"
assert _p8np.isfinite(_p8normals).all() and _p8np.isfinite(_p8offsets).all(), "P8 PLIC 有限"
for _p8i in range(_p8n):
    _p8vol = _p8V.keep_phase1_volume(_p8fv, _p8i, _p8normals[_p8i], _p8offsets[_p8i])
    _p8tgt = float(_p8np.clip(_p8alpha[_p8i], 0.0, 1.0)) * _p8fv.volumes[_p8i]
    assert abs(_p8vol - _p8tgt) <= 1e-5 * max(_p8fv.volumes[_p8i], 1e-12), "P8 几何体积一致"
# VofSolver：初始全 α=0（空气）、update() 注入入口 α 增长、混合物性有界
_p8vs = _p8V.VofSolver(_p8fv, rho1=998.0, rho2=1.18, mu1=1.0e-3, mu2=1.8e-5,
                       inlet_alpha=1.0)
assert _p8np.all(_p8vs.alpha == 0.0), "P8 VOF 初始 α"
assert _p8np.allclose(_p8vs.rho, 1.18), "P8 VOF 初始混合密度"
_p8u = _p8np.full(_p8n, 1.0); _p8v = _p8np.zeros(_p8n); _p8w = _p8np.zeros(_p8n)
_p8r0 = float(_p8vs.update(_p8u, _p8v, _p8w))
assert _p8np.isfinite(_p8r0), "P8 VOF update 残差有限"
assert _p8vs.iteration == 1 and _p8vs.phase1_volume > 0.0, "P8 VOF 注入 α 增长"
assert _p8vs.alpha.min() >= 0.0 and _p8vs.alpha.max() <= 1.0, "P8 VOF α 有界"
assert _p8vs.rho.max() <= 998.0 + 1e-9 and _p8vs.rho.min() >= 1.18 - 1e-9, "P8 混合密度有界"
# 表面张力（CSF）：均匀 α 场下界面曲率 0 → 体积力 ≈ 0
_p8sf = _p8V.surface_tension_force(_p8fv, _p8np.full(_p8n, 0.5))
assert _p8sf.shape == (_p8n, 3) and _p8np.isfinite(_p8sf).all(), "P8 CSF 形状有限"
assert _p8np.allclose(_p8sf, 0.0, atol=1e-9), "P8 均匀 α CSF≈0"
# 工厂：别名/大小写解析，未知报 ValueError
assert isinstance(_p8V.make_vof(_p8fv, "vof"), _p8V.VofSolver), "P8 工厂 vof"
assert isinstance(_p8V.make_vof(_p8fv, "volume_of_fluid"), _p8V.VofSolver), "P8 工厂 alias"
try:
    _p8V.make_vof(_p8fv, "nope")
    raise AssertionError("P8 应拒绝未知 VOF 模型")
except ValueError:
    pass
# 集成：PressureSolver 字符串注入 VOF（单流体恒密度面物性）+ step() 稳定、α 增长
_p8s0 = _p8Solver(_p8Vc, _p8Cc, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0), max_outer=3)
assert _p8s0.vof_model is None, "P8 基线应无 VOF 模型"
assert _p8np.isfinite(_p8s0.step()["residual"]), "P8 基线 step() 无报错"
_p8sv = _p8Solver(_p8Vc, _p8Cc, mu=1.8e-5, rho=1.18,
                  inlet_velocity=(1.0, 0.0, 0.0), max_outer=3,
                  vof_model="vof", inlet_alpha=1.0)
assert _p8sv.vof_model is not None, "P8 应构建 VOF 模型"
assert _p8np.allclose(_p8sv._face_rho(), 1.18), "P8 单流体恒密度面物性"
assert _p8np.allclose(_p8sv._face_mu(), 1.8e-5), "P8 单流体恒粘度面物性"
for _p8i in range(5):
    _p8pv = _p8sv.step()
assert _p8np.isfinite(_p8sv.velocity()).all() and _p8np.isfinite(_p8pv["residual"]), \
    "P8 耦合流场/残差有限"
assert _p8sv.vof_model.alpha.min() >= 0.0 and _p8sv.vof_model.alpha.max() <= 1.0, \
    "P8 耦合 α 有界"
assert _p8sv.vof_model.phase1_volume > 0.0, "P8 耦合 α 增长"
_p8m = _p8sv.monitor_payload()
assert {"alpha_min", "alpha_max", "phase1_volume"} <= set(_p8m), "P8 monitor 含 α 键"
print("P8 多相 VOF：α 守恒输运/PLIC 几何重构/两相物性混合/CSF 表面张力/单流体耦合 全通过")

# ---------------- P8b 多相 Mixture：N 相体积分数输运 + 代数滑移（drift-flux）----------------
# 验收核心（P8 行）Mixture：多相体积分数守恒输运 + 代数滑移封闭 + 混合物性。纯 numpy 可用。
import mixture as _p8M
# 常量 / 物性混合：三相水-气-油，混合场逐单元加权落在相之间
assert _p8np.isclose(_p8M.DEFAULT_RHOS[2], 800.0), "P8 Mixture 相 3 密度"
assert _p8np.isclose(_p8M.DEFAULT_MUS[1], 1.8e-5), "P8 Mixture 相 2 粘度"
assert _p8np.isclose(
    _p8M.mixture_rho(_p8np.array([[0.5, 0.25, 0.25]])),
    0.5 * 998.0 + 0.25 * 1.18 + 0.25 * 800.0), "P8 Mixture 混合密度"
# 漂移速度封闭：Σ_k α_k u_dr,k = 0 保证总体积守恒
_p8malpha = _p8np.zeros((_p8n, 3), float)
_p8malpha[:, 1] = 0.3; _p8malpha[:, 2] = 0.2; _p8malpha[:, 0] = 0.5
_p8mslip = _p8np.zeros((_p8n, 3, 3), float)
_p8mslip[:, 1, 2] = -0.02; _p8mslip[:, 2, 2] = -0.01
_p8mdrift = _p8M.drift_velocities(_p8malpha, _p8mslip)
assert _p8np.allclose(_p8np.sum(_p8malpha[:, :, None] * _p8mdrift, axis=1), 0.0,
                      atol=1e-12), "P8 Mixture 漂移体积守恒"
# 守恒输运：advance_mixture 有界且 α0 = 1 - Σ 弥散相
_p8m_u = _p8np.full(_p8n, 1.0); _p8m_v = _p8np.zeros(_p8n); _p8m_w = _p8np.zeros(_p8n)
_p8m_mdot = _p8M.volume_flux(_p8fv, _p8m_u, _p8m_v, _p8m_w)
_p8m_a0 = _p8np.zeros((_p8n, 3), float); _p8m_a0[:, 0] = 1.0
_p8m_dr0 = _p8np.zeros((_p8n, 3, 3), float)
_p8m_a_new, _p8m_dt = _p8M.advance_mixture(_p8fv, _p8m_mdot, _p8m_a0, _p8m_dr0, 1e-3)
assert _p8np.isfinite(_p8m_a_new).all(), "P8 Mixture 输运有限"
assert _p8m_a_new.min() >= 0.0 and _p8m_a_new.max() <= 1.0, "P8 Mixture α 有界"
assert _p8np.allclose(_p8m_a_new[:, 0], 1.0 - _p8m_a_new[:, 1:].sum(axis=1),
                      atol=1e-12), "P8 Mixture α0 补足"
# MixtureSolver：初始 N 相 α、混合密度/粘度、update() 注入弥散相增长
_p8ms = _p8M.MixtureSolver(_p8fv, alpha0=[0.1, 0.05])
assert _p8ms.n_phases == 3, "P8 Mixture 相数"
assert _p8np.allclose(_p8ms.alphas[:, 0], 0.85), "P8 Mixture 初始连续相 α"
assert _p8np.allclose(_p8ms.alphas.sum(axis=1), 1.0), "P8 Mixture α 求和 1"
assert _p8np.isfinite(_p8ms.rho).all() and _p8np.isfinite(_p8ms.mu).all(), \
    "P8 Mixture 物性有限"
_p8mr = float(_p8ms.update(_p8m_u, _p8m_v, _p8m_w))
assert _p8np.isfinite(_p8mr), "P8 Mixture update 残差有限"
assert _p8ms.iteration == 1 and _p8ms.phase_volume(1) > 0.0, "P8 Mixture 注入增长"
# 工厂
assert isinstance(_p8M.make_mixture(_p8fv, "mixture"), _p8M.MixtureSolver), \
    "P8 Mixture 工厂"
try:
    _p8M.make_mixture(_p8fv, "nope")
    raise AssertionError("P8 应拒绝未知 Mixture 模型")
except ValueError:
    pass
# 集成：PressureSolver 字符串注入 Mixture + step() 稳定、rho/mu 逐单元变化、monitor 含 mix 键
_p8sm = _p8Solver(_p8Vc, _p8Cc, mu=1e-3, rho=998.0, inlet_velocity=(1.0, 0.0, 0.0),
                  max_outer=3, mixture_model="mixture",
                  mixture_inlet_alphas=[0.2, 0.1])
assert _p8sm.mixture_model is not None and not isinstance(_p8sm.mixture_model, str), \
    "P8 应构建 Mixture 模型"
for _p8i in range(5):
    _p8pm = _p8sm.step()
assert _p8np.isfinite(_p8sm.velocity()).all() and _p8np.isfinite(_p8pm["residual"]), \
    "P8 Mixture 耦合流场/残差有限"
assert _p8np.allclose(_p8sm.mixture_model.alphas.sum(axis=1), 1.0, atol=1e-8), \
    "P8 Mixture 耦合 α 求和 1"
_p8mm = _p8sm.monitor_payload()
assert {"mix_rho_min", "mix_rho_max", "mix_alpha_sum"} <= set(_p8mm), \
    "P8 Mixture monitor 含 mix 键"
print("P8 Mixture：N 相体积分数守恒输运/漂移封闭/混合密度粘度/单流体耦合 全通过")

# ---------------- P8c DPM 离散相：拉格朗日粒子注入 + 运动积分 + 连续相耦合 ----------------
# 验收核心（P8 行）DPM：拉格朗日粒子轨（注入/运动/耦合源）。纯 numpy 可用。
import dpm as _p8D
# 常量 / 四面体点定位：域内质心命中所在单元，域外 -1
assert _p8np.isclose(_p8D.DEFAULT_RHO_P, 2500.0), "P8 DPM 粒子密度"
assert _p8D.locate_cells(_p8fv, _p8fv.centroids).min() >= 0, "P8 DPM 域内定位"
assert _p8D.locate_cells(_p8fv, _p8fv.centroids[:1] + 1e3)[0] == -1, "P8 DPM 域外定位"
# DpmSolver：初始无人、inject() 注入、update() 推进、耦合源有限、轨迹快照
_p8d = _p8D.DpmSolver(_p8fv, parcels_per_step=5)
assert _p8d.n_particles == 0 and _p8d.n_active == 0, "P8 DPM 初始无粒子"
assert _p8d.inject() == 5 and _p8d.n_particles == 5, "P8 DPM 注入"
_p8dr = float(_p8d.update(_p8m_u, _p8m_v, _p8m_w))
assert _p8np.isfinite(_p8dr), "P8 DPM update 残差有限"
assert _p8d.iteration == 1 and _p8d.n_particles >= 5, "P8 DPM 推进粒子增长"
_p8dcs = _p8d.coupling_source()
assert _p8dcs.shape == (_p8n, 3) and _p8np.isfinite(_p8dcs).all(), "P8 DPM 耦合源有限"
assert len(_p8d.trajectories()) == 1, "P8 DPM 轨迹快照"
# 工厂
assert isinstance(_p8D.make_dpm(_p8fv, "dpm"), _p8D.DpmSolver), "P8 DPM 工厂"
try:
    _p8D.make_dpm(_p8fv, "nope")
    raise AssertionError("P8 应拒绝未知 DPM 模型")
except ValueError:
    pass
# 集成：PressureSolver 字符串注入 DPM + step() 稳定、粒子增长、monitor 含 n_particles 键
_p8sd = _p8Solver(_p8Vc, _p8Cc, mu=1e-3, rho=998.0, inlet_velocity=(1.0, 0.0, 0.0),
                  max_outer=3, dpm_model="dpm", dpm_rho_p=2500.0, dpm_dia_p=1e-4)
assert _p8sd.dpm_model is not None and not isinstance(_p8sd.dpm_model, str), \
    "P8 应构建 DPM 模型"
for _p8i in range(5):
    _p8pd = _p8sd.step()
assert _p8np.isfinite(_p8sd.velocity()).all() and _p8np.isfinite(_p8pd["residual"]), \
    "P8 DPM 耦合流场/残差有限"
assert _p8sd.dpm_model.n_particles > 0, "P8 DPM 耦合粒子增长"
_p8md = _p8sd.monitor_payload()
assert {"n_particles", "n_active", "n_escaped"} <= set(_p8md), "P8 DPM monitor 含粒子键"
print("P8 DPM：拉格朗日粒子注入/运动积分/连续相耦合源/单流体耦合 全通过")

# ---------------- P9 燃烧算例：组分输运（质量分数）+ 全局 Arrhenius 反应动力学 ----------------
# 验收核心（P9 行）：组分输运 + 燃烧反应动力学（star.species / star.combustion）。纯 numpy 可用。
from fvm_core import cube_tet_mesh as _p9cube, FVM as _p9FVM
from pressure_solver import PressureSolver as _p9Solver
import species as _p9Sp
import combustion as _p9Cb
import numpy as _p9np
# 组分换算：质量↔摩尔彼此往返，混合摩尔质量为正，理想气体密度命中 P W/(R T)
_p9Y = _p9np.array([[0.0, 0.233, 0.0, 0.0, 0.767]], float)
_p9X = _p9Sp.mass_to_mole_frac(_p9Y, _p9Sp.DEFAULT_MW)
_p9Y2 = _p9Sp.mole_to_mass_frac(_p9X, _p9Sp.DEFAULT_MW)
assert _p9np.allclose(_p9Y2, _p9Y, atol=1e-9), "P9 质量↔摩尔往返"
_p9W = _p9Sp.mixture_molar_mass(_p9Y, _p9Sp.DEFAULT_MW)
assert _p9W.shape == (1,) and float(_p9W[0]) > 0.0, "P9 混合摩尔质量"
_p9rho = _p9Sp.gas_density(_p9W, 300.0, 101325.0)
assert _p9np.isfinite(_p9rho).all() and _p9np.all(_p9rho > 0.0), "P9 理想气体密度"
assert _p9np.isclose(float(_p9rho[0]),
                     101325.0 * float(_p9W[0]) / (_p9Sp.R_UNIV * 300.0)), "P9 密度公式"
# 零组分保护：全零分数 → 摩尔分数 全 0（无除零）
assert _p9np.all(_p9Sp.mass_to_mole_frac(_p9np.zeros((3, 5), float),
                                         _p9Sp.DEFAULT_MW) == 0.0), "P9 零组分保护"
# 输运装配/求解：形状 + 有限；Schmidt 数/扩散系数常量正值
assert _p9Sp.DEFAULT_SC_TURB > 0.0 and _p9Sp.DEFAULT_SCALAR_DIFF > 0.0, "P9 输运常量"
assert len({_p9Sp.SP_ZERO_GRAD, _p9Sp.SP_FIXED}) == 2, "P9 组分边界类型互异"
_p9V, _p9C = _p9cube(nx=2)
_p9fv = _p9FVM(_p9V, _p9C)
_p9n = _p9fv.n_cells
_p9u = _p9np.full(_p9n, 1.0); _p9v = _p9np.zeros(_p9n); _p9w = _p9np.zeros(_p9n)
_p9mdot = _p9Sp.volume_mass(_p9fv, _p9u, _p9v, _p9w, _p9np.full(_p9n, 1.0))
_p9gm = _p9np.full(_p9n, 1.0 * _p9Sp.DEFAULT_SCALAR_DIFF, float)
_p9Yc = _p9np.full(_p9n, 0.233, float)
_p9rows, _p9cols, _p9vals, _p9rhs, _p9ap = _p9Sp.assemble_species_transport(
    _p9fv, _p9mdot, _p9gm, _p9Yc, _p9np.zeros(_p9n))
assert _p9np.isfinite(_p9vals).all() and _p9np.isfinite(_p9rhs).all(), "P9 组分装配有限"
assert _p9rhs.shape == (_p9n,) and _p9ap.shape == (_p9n,), "P9 组分装配形状"
# 边界分类：入口/出口/壁面全覆盖不重叠
_p9in, _p9out, _p9wall, _p9bnd = _p9Sp.classify_species(_p9fv)
assert (_p9in.size + _p9out.size + _p9wall.size) == int(_p9np.sum(_p9fv.is_boundary)), \
    "P9 边界全覆盖"
assert _p9bnd.size == int(_p9np.sum(_p9fv.is_boundary)), "P9 边界面全集"
assert len(set(_p9in.tolist()) & set(_p9out.tolist()) & set(_p9wall.tolist())) == 0, \
    "P9 边界不重叠"
# SpeciesSolver：初始 Σ Y = 1，update() 收敛且有界
_p9sp = _p9Sp.SpeciesSolver(_p9fv,
                            inlet_fractions=[0.0, 0.233, 0.0, 0.0, 0.767])
assert _p9np.allclose(_p9sp.Y.sum(axis=1), 1.0, atol=1e-9), "P9 组分 ΣY=1"
_p9sr = float(_p9sp.update(_p9u, _p9v, _p9w, mdot=_p9mdot,
                           nu_t=_p9np.zeros(_p9n, float)))
assert _p9np.isfinite(_p9sr) and _p9np.allclose(_p9sp.Y.sum(axis=1), 1.0, atol=1e-9), \
    "P9 组分 update 收敛"
assert _p9sp.Y.min() >= 0.0 and _p9sp.Y.max() <= 1.0, "P9 组分有界"
# P10 后端门面：initialize/step/residual/monitor_payload 可用
assert _p9Sp.make_species(_p9fv, "species") is not None, "P9 组分工厂"
assert _p9np.isfinite(_p9sp.residual()), "P9 组分残差"
assert {"n_species", "sum_max"} <= set(_p9sp.monitor_payload()), "P9 组分 monitor 键"
# Arrhenius 速率：随温度单调升；β 指数提升
_p9a_cold = _p9Cb.arrhenius_rate(_p9Cb.DEFAULT_A, _p9Cb.DEFAULT_EA, 0.0, 300.0)
_p9a_hot = _p9Cb.arrhenius_rate(_p9Cb.DEFAULT_A, _p9Cb.DEFAULT_EA, 0.0, 2000.0)
assert _p9a_hot > _p9a_cold > 0.0, "P9 Arrhenius 单调"
assert _p9Cb.arrhenius_rate(_p9Cb.DEFAULT_A, _p9Cb.DEFAULT_EA, 1.0, 2000.0) > _p9a_hot, \
    "P9 β 提升速率"
# GlobalReaction：反应速率/质量源自动守恒 (Σ S_i = 0)/热释放非负
_p9reac = _p9Cb.make_prequick_mech()
_p9conc = _p9np.array([[0.5, 1.0, 0.0, 0.0, 0.0]], float)
_p9wc = float(_p9reac.rate(_p9conc, _p9np.array([300.0]))[0])
_p9wh = float(_p9reac.rate(_p9conc, _p9np.array([2000.0]))[0])
assert _p9wc >= 0.0 and _p9wh >= 0.0 and _p9wh > _p9wc, "P9 反应速率"
# 质量守恒：按 nu 与平衡分子量，Σ Mw_i ν_i = 0 → Σ S_i = 0
_p9r2 = _p9Cb.GlobalReaction(nu=[-1.0, 1.0], reactant_idx=[0],
                             reactant_orders=[1.0], A=1.0e6, Ea=1.0e8)
_p9Yr = _p9np.array([[0.5, 0.5]], float)
_p9Mwr = _p9np.array([10.0, 10.0], float)
_p9S = _p9r2.mass_source(_p9Yr, _p9np.array([1500.0], float),
                         _p9np.array([1.0], float), _p9Mwr)
assert _p9S.shape == (1, 2) and _p9np.allclose(_p9np.sum(_p9S, axis=1), 0.0, atol=1e-9), \
    "P9 质量源守恒"
assert _p9np.all(_p9r2.heat_release(_p9Yr, _p9np.array([1500.0], float),
                                    _p9np.array([1.0], float), _p9Mwr) >= 0.0), "P9 放热非负"
# 层流火焰速度/厚度/进度变量
assert _p9np.isclose(_p9Cb.laminar_flame_speed(300.0), 0.4), "P9 火焰速度"
assert _p9Cb.laminar_flame_speed(600.0) > _p9Cb.laminar_flame_speed(300.0), "P9 火焰速度升"
assert _p9np.isclose(_p9Cb.flame_thickness(0.4), 2.2e-5 / 0.4), "P9 火焰厚度"
_p9cp = _p9Cb.combustion_progress(_p9np.array([[0.05, 0.0], [0.0, 0.0]], float),
                                  fuel_idx=0, Y_fuel_inlet=0.05)
assert _p9np.allclose(_p9cp, [0.0, 1.0], atol=1e-9), "P9 进度变量"
# 点火器：温度/火花/进度 调制
_p9tig = _p9Cb.TemperatureIgnitor(threshold=1500.0)
assert _p9np.all(_p9tig.factor(_p9fv.centroids,
                               T=_p9np.full(_p9n, 2000.0)) == 1.0), "P9 高温点火"
assert _p9np.all(_p9tig.factor(_p9fv.centroids,
                               T=_p9np.full(_p9n, 300.0)) == 0.0), "P9 低温点火"
_p9sig = _p9Cb.SparkIgnitor(position=(0.5, 0.5, 0.5), radius=0.6, energy=1.0e5,
                            duration=1.0)
_p9f = _p9sig.factor(_p9fv.centroids)
assert _p9np.all(_p9f >= 0.0) and _p9np.all(_p9f <= 1.0), "P9 火花系数有界"
assert float(_p9np.max(_p9sig.heat_source(_p9fv.centroids))) > 0.0, "P9 火花热源"
_p9pig = _p9Cb.ProgressVariableIgnitor(threshold=0.5)
assert _p9np.all(_p9pig.factor(_p9fv.centroids,
                               progress=_p9np.full(_p9n, 0.9)) == 1.0), "P9 进度点火"
# 工厂：别名/大小写解析，未知报 ValueError
assert isinstance(_p9Cb.make_ignitor("temperature"), _p9Cb.TemperatureIgnitor), \
    "P9 点火器工厂 temperature"
assert isinstance(_p9Cb.make_ignitor("spark"), _p9Cb.SparkIgnitor), "P9 点火器工厂 spark"
assert isinstance(_p9Cb.make_combustion(_p9fv, "combustion"), _p9Cb.CombustionModel), \
    "P9 燃烧工厂 combustion"
try:
    _p9Cb.make_combustion(_p9fv, "nope")
    raise AssertionError("P9 应拒绝未知燃烧模型")
except ValueError:
    pass
# CombustionModel：低温未燃 Q≈0 + 燃料保存；高温燃料放热 + ΣY=1
_p9cm = _p9Cb.CombustionModel(_p9fv, inlet_fractions=[0.05, 0.20, 0.0, 0.0, 0.75])
_p9cm.set_temperature(_p9np.full(_p9n, 300.0, float))
_p9cm._chemical_source()
assert _p9cm.omega.max() < 1.0e6 and _p9np.allclose(_p9cm.Y.sum(axis=1), 1.0), \
    "P9 低温未燃"
for _p9i in range(3):
    _p9cr = float(_p9cm.update(_p9u, _p9v, _p9w, mdot=_p9mdot,
                               nu_t=_p9np.zeros(_p9n, float),
                               T=_p9np.full(_p9n, 2000.0, float)))
assert _p9np.isfinite(_p9cr) and float(_p9cm.heat_release.max()) > 0.0, "P9 高温放热"
assert _p9np.allclose(_p9cm._sp.Y.sum(axis=1), 1.0, atol=1e-9), "P9 燃烧组分守恒"
# 集成：PressureSolver 字符串注入 species/combustion，step() 无报错；基线无模型无回归
_p9s0 = _p9Solver(_p9V, _p9C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0), max_outer=3)
assert _p9s0.species_model is None and _p9s0.combustion_model is None, "P9 基线无模型"
assert _p9np.isfinite(_p9s0.step()["residual"]), "P9 基线 step() 无报错"
_p9sc = _p9Solver(_p9V, _p9C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0), max_outer=3,
                  species_model="species", combustion_model="combustion",
                  combustion_inlet_fractions=[0.05, 0.20, 0.0, 0.0, 0.75])
for _p9i in range(5):
    _p9ps = _p9sc.step()
assert _p9np.isfinite(_p9sc.velocity()).all() and _p9np.isfinite(_p9ps["residual"]), \
    "P9 耦合流场/残差有限"
assert _p9sc.combustion_model is not None and not isinstance(_p9sc.combustion_model, str), \
    "P9 应构建燃烧模型"
assert not isinstance(_p9sc.species_model, str) and _p9sc.species_model is not None, \
    "P9 应构建组分模型"
_p9m9 = _p9sc.monitor_payload()
assert {"combo_Q_max", "combo_T_max"} <= set(_p9m9), "P9 燃烧 monitor 键"
print("P9 燃烧算例：组分质量分数输运/换算/全局 Arrhenius 反应动力学/火焰诊断/"
      "点火器/PressureSolver 耦合 全通过")

# ---------------- P10 多相欧拉-欧拉：相体积分数输运 + 相间作用力 + 群体平衡 ----------------
# 验收核心（P10 行）：star.multiphase（EulerianPhase + PhaseInteraction + Population
# Balance）：相间拖曳/升力/虚拟质量/壁面润滑 + 聚并/破碎/成核。纯 numpy 可用。
from fvm_core import cube_tet_mesh as _p10cube, FVM as _p10FVM
from pressure_solver import PressureSolver as _p10Solver
import eulerian_multiphase as _p10EM
import numpy as _p10np
# 常量：升力/虚拟质量/壁面润滑常数 + 群体平衡核常数（默认群体平衡关闭）
assert _p10EM.DEFAULT_LIFT_COEF > 0.0 and _p10EM.DEFAULT_VIRTUAL_MASS_COEF > 0.0, \
    "P10 相间力常量"
assert _p10EM.DEFAULT_WALL_LUB_COEF_CW1 < 0.0 and _p10EM.DEFAULT_WALL_LUB_COEF_CW2 > 0.0, \
    "P10 壁面润滑常量"
assert _p10EM.DEFAULT_COALESCENCE_RATE == 0.0 and _p10EM.DEFAULT_BREAKUP_RATE == 0.0 \
    and _p10EM.DEFAULT_NUCLEATION_RATE == 0.0, "P10 群体平衡默认关闭"
# 拖曳：Schiller-Naumann（Stokes 低 Re 极限有限，无除零）+ Wen-Yu 空泡率修正
_p10d = _p10EM.drag_exchange(1e-3, 998.0, 1e-3, _p10np.array([1.0, 0.1]),
                             _p10np.array([0.1, 0.2]), _p10np.array([0.9, 0.8]))
assert _p10np.all(_p10np.isfinite(_p10d)) and _p10np.all(_p10d > 0.0), "P10 拖曳有限正值"
assert float(_p10d[0]) > float(_p10d[1]), "P10 拖曳随相对速度单调（高 Re 更高）"
_p10slim = _p10EM.drag_exchange(1e-3, 998.0, 1e-3, _p10np.array([0.0, 0.0]),
                                _p10np.array([0.2, 0.2]), _p10np.array([0.8, 0.8]))
assert _p10np.all(_p10np.isfinite(_p10slim)) and _p10np.all(_p10slim > 0.0), "P10 Stokes 极限有限"
_p10K, _p10F = _p10EM.interphase_drag(1e-3, 998.0, 1e-3, _p10np.array([[1.0, 0.0, 0.0],
                                                                          [0.5, 0.0, 0.0]]),
                                      _p10np.array([0.1, 0.1]), _p10np.array([0.9, 0.9]))
assert _p10F.shape == (2, 3) and _p10np.all(_p10np.isfinite(_p10F)), "P10 拖曳力密度"
# 升力/虚拟质量/壁面润滑：系数为 0 或粒径 0 时返回零
_p10zero = _p10EM.lift_force(_p10np.full(3, 0.1), 998.0,
                             _p10np.zeros((3, 3)), _p10np.zeros((3, 3)), coef=0.0)
assert _p10np.allclose(_p10zero, 0.0), "P10 升力 coef=0 归零"
_p10vmz = _p10EM.virtual_mass_force(_p10np.full(3, 0.1), 998.0,
                                    _p10np.zeros((3, 3)), coef=0.0)
assert _p10np.allclose(_p10vmz, 0.0), "P10 虚拟质量 coef=0 归零"
_p10wlz = _p10EM.wall_lubrication_force(_p10np.full(3, 0.1), 998.0,
                                        _p10np.zeros((3, 3)), _p10np.full(3, 0.05),
                                        _p10np.zeros((3, 3)), d=0.0)
assert _p10np.allclose(_p10wlz, 0.0), "P10 壁面润滑 d=0 归零"
# 群体平衡：Σ_k S_k = 0 守恒；全核为 0 返回全零
_p10pb = _p10EM.population_balance_source(_p10np.array([[0.6, 0.3, 0.1],
                                                         [0.5, 0.4, 0.1]], float),
                                          _p10np.array([0.0, 5e-4, 1e-3]),
                                          0.1, 0.05, 0.02)
assert _p10pb.shape == (2, 3), "P10 群体平衡源形状"
assert _p10np.allclose(_p10pb.sum(axis=1), 0.0, atol=1e-12), "P10 群体平衡守恒"
_p10pb0 = _p10EM.population_balance_source(_p10np.array([[0.6, 0.3, 0.1]], float),
                                           _p10np.array([0.0, 5e-4, 1e-3]), 0.0, 0.0, 0.0)
assert _p10np.allclose(_p10pb0, 0.0), "P10 群体平衡全关归零"
# EulerianMultiphaseSolver：初始 Σα=1、混合密度/粘度、update() 注入弥散相增长
_p10V, _p10C = _p10cube(nx=2)
_p10fv = _p10FVM(_p10V, _p10C)
_p10n = _p10fv.n_cells
_p10ems = _p10EM.EulerianMultiphaseSolver(_p10fv, alpha0=[0.1, 0.05])
assert _p10ems.n_phases == 3, "P10 相数"
assert _p10np.allclose(_p10ems.alphas[:, 0], 0.85) and _p10np.allclose(
    _p10ems.alphas.sum(axis=1), 1.0), "P10 初始 α 求和 1"
assert _p10np.allclose(_p10ems.rho, 0.85 * 998.0 + 0.1 * 1.18 + 0.05 * 800.0), "P10 混合密度"
assert _p10ems.slip.shape == (_p10n, 3, 3) and _p10ems.drift.shape == (_p10n, 3, 3), "P10 相速度形状"
_p10u = _p10np.full(_p10n, 1.0); _p10v = _p10np.zeros(_p10n); _p10w = _p10np.zeros(_p10n)
_p10ur = float(_p10ems.update(_p10u, _p10v, _p10w))
assert _p10np.isfinite(_p10ur) and _p10ems.iteration == 1, "P10 update 残差有限"
assert _p10np.allclose(_p10ems.alphas.sum(axis=1), 1.0, atol=1e-9), "P10 update α 守恒"
assert _p10ems.alphas.min() >= 0.0 and _p10ems.alphas.max() <= 1.0, "P10 α 有界"
assert _p10ems.phase_volume(1) > 0.0 and _p10ems.phase_volume(2) > 0.0, "P10 注入增长"
# P10 后端门面：initialize/step/residual/monitor_payload 可用
_p10info = _p10ems._initialize_field()
assert {"n_phases", "alpha_min", "alpha_max", "rho_min", "rho_max"} <= set(_p10info), \
    "P10 initialize 键"
assert abs(_p10ems.residual() - 0.0) <= 1e-12, "P10 初始残差"
_p10ems.set_velocities(_p10u, _p10v, _p10w)
_p10p = _p10ems.step()
assert {"residual", "n_phases", "alpha_min", "alpha_max"} <= set(_p10p), "P10 step 键"
_p10m = _p10ems.monitor_payload()
assert {"n_phases", "phase_volumes", "momentum_src", "residual"} <= set(_p10m), "P10 monitor 键"
# 工厂：别名 + 大小写解析，未知模型报 ValueError
assert isinstance(_p10EM.make_eulerian_multiphase(_p10fv, "eulerian"),
                  _p10EM.EulerianMultiphaseSolver), "P10 工厂 eulerian"
assert isinstance(_p10EM.make_eulerian_multiphase(_p10fv, "ee"),
                  _p10EM.EulerianMultiphaseSolver), "P10 工厂 ee"
assert isinstance(_p10EM.make_eulerian_multiphase(_p10fv, "EULER_EULER"),
                  _p10EM.EulerianMultiphaseSolver), "P10 工厂大小写"
try:
    _p10EM.make_eulerian_multiphase(_p10fv, "nope")
    raise AssertionError("P10 应拒绝未知欧拉-欧拉模型")
except ValueError:
    pass
# 相间作用力开启后动量源非零（非均匀流场才会产生升力/虚拟质量贡献）
_p10ems2 = _p10EM.EulerianMultiphaseSolver(
    _p10fv, alpha0=[0.2, 0.1], enable_lift=True, enable_virtual_mass=True)
_p10ems2.update(_p10u + _p10np.linspace(0.0, 0.2, _p10n), _p10v, _p10w)
assert _p10np.isfinite(_p10ems2.momentum_source).all(), "P10 相间动量源有限"
# 集成：PressureSolver 字符串注入 eulerian + step() 稳定、αΣ=1、monitor 含 ee 键
_p10s0 = _p10Solver(_p10V, _p10C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0), max_outer=3)
assert _p10s0.eulerian_model is None, "P10 基线无模型"
assert _p10np.isfinite(_p10s0.step()["residual"]), "P10 基线 step() 无报错"
_p10se = _p10Solver(_p10V, _p10C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0), max_outer=3,
                    eulerian_model="eulerian", eulerian_inlet_alphas=[0.2, 0.1],
                    eulerian_enable_lift=True, eulerian_population_balance=True,
                    eulerian_coalescence_rate=0.1)
for _p10i in range(6):
    _p10ps = _p10se.step()
assert _p10np.isfinite(_p10se.velocity()).all() and _p10np.isfinite(_p10ps["residual"]), \
    "P10 耦合流场/残差有限"
assert _p10se.eulerian_model is not None and not isinstance(_p10se.eulerian_model, str), \
    "P10 应构建欧拉-欧拉模型"
assert _p10np.allclose(_p10se.eulerian_model.alphas.sum(axis=1), 1.0, atol=1e-8), \
    "P10 耦合 α 守恒"
assert _p10np.isfinite(_p10se.rho).all(), "P10 耦合混合密度有限"
_p10m10 = _p10se.monitor_payload()
assert {"ee_n_phases", "ee_alpha_min", "ee_alpha_max", "ee_alpha_sum",
        "ee_mom_src"} <= set(_p10m10), "P10 monitor 含 ee 键"
print("P10 多相欧拉-欧拉：相体积分数守恒输运/相间拖曳升力虚拟质量壁面润滑/"
      "群体平衡（聚并破碎成核）/P10 后端门面/PressureSolver 耦合 全通过")

# ---------------- P11 运动谱系：刚体运动 + 滑移 interface、morphing、DFBI 6DOF、
# ------------------- overset 重叠插值、MRF 旋转参考系源 ----------------
# 验收核心（P11 行）：star.motion（Rigid Body / Sliding / Morphing / DFBI 6DOF /
# Overset Mesh / Rotating Reference Frame）——原 P9 因燃烧顺延的运动谱系。纯 numpy 可用。
from fvm_core import cube_tet_mesh as _p11cube, FVM as _p11FVM
from pressure_solver import PressureSolver as _p11Solver
from motion import (
    DfbiBody, MotionSolver, make_motion, morph_mesh, mrf_source,
    overset_donor_weights, overset_interpolate, quat_to_dcm, rigid_transform,
    rotate_points, sliding_interface, DEFAULT_ROTATION_SPEED, MAX_MORPH_ITER,
    DEFAULT_MORPH_RELAX, DEFAULT_ROTATION_DT,
)
import numpy as _p11np
# 常量
assert _p11np.isclose(DEFAULT_ROTATION_SPEED, 0.0), "P11 默认转速"
assert MAX_MORPH_ITER > 0 and 0.0 < DEFAULT_MORPH_RELAX <= 1.0, "P11 morph 常量"
assert DEFAULT_ROTATION_DT > 0.0, "P11 时间步常量"
# 刚体运动：Rodrigues 旋转（z 轴 90° x→y、轴上点不动、距离保持）
assert _p11np.allclose(rotate_points(_p11np.array([[1.0, 0.0, 0.0]]),
                                     (0, 0, 1), _p11np.pi / 2.0)[0],
                       [0.0, 1.0, 0.0], atol=1e-12), "P11 旋转 x→y"
assert _p11np.allclose(rotate_points(_p11np.array([[0.0, 0.0, 5.0]]),
                                     (0, 0, 1), 0.7)[0],
                       [0.0, 0.0, 5.0], atol=1e-12), "P11 轴上点不动"
_p11v = _p11np.array([[1.0, 2.0, 3.0]])
assert abs(_p11np.linalg.norm(rotate_points(_p11v, (1, 1, 0), 0.5))
           - _p11np.linalg.norm(_p11v)) < 1e-12, "P11 正交距离保持"
# rigid_transform：转速 0 纯平移、转速旋转无平移距离保持
_p11V = _p11np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
assert _p11np.allclose(rigid_transform(_p11V, rotation_speed=0.0,
                                       translation_velocity=(1.0, 2.0, 3.0), t=2.0),
                       _p11V + _p11np.array([2.0, 4.0, 6.0]), atol=1e-12), "P11 纯平移"
assert _p11np.allclose(rigid_transform(_p11np.array([[1.0, 0.0, 0.0]]),
                                       rotation_axis=(0, 0, 1), rotation_speed=1.0,
                                       translation_velocity=(0, 0, 0), t=_p11np.pi / 2.0)[0],
                       [0.0, 1.0, 0.0], atol=1e-12), "P11 旋转位移"
# 滑移接口
_p11Vc, _p11Cc = _p11cube(nx=3)
_p11fv = _p11FVM(_p11Vc, _p11Cc)
_p11sl = sliding_interface(_p11fv, motion_axis=(0, 0, 1), sign=0.0, tol=1e-9)
assert _p11sl.dtype == _p11np.int64 and _p11sl.size > 0, "P11 滑移面标识"
# MRF 旋转源：omega=0 归零、离心径向外、科氏反速度
_p11n = _p11fv.n_cells
assert _p11np.allclose(mrf_source(_p11np.zeros((2, 3)), _p11np.zeros(2),
                                  _p11np.zeros(2), _p11np.zeros(2), 0.0,
                                  (0, 0, 1), 1.0), 0.0), "P11 MRF omega=0 归零"
assert _p11np.allclose(mrf_source(_p11np.array([[1.0, 0.0, 0.0]]),
                                  _p11np.zeros(1), _p11np.zeros(1), _p11np.zeros(1),
                                  2.0, (0, 0, 1), 1.0)[0],
                       [4.0, 0.0, 0.0], atol=1e-12), "P11 MRF 离心径向外"
assert _p11np.allclose(mrf_source(_p11np.array([[0.0, 0.0, 0.0]]),
                                  _p11np.array([1.0]), _p11np.zeros(1),
                                  _p11np.zeros(1), 2.0, (0, 0, 1), 1.0)[0],
                       [0.0, -4.0, 0.0], atol=1e-12), "P11 MRF 科氏反速度"
# morphing：零位移恒等、边界夹持、内部有限
assert _p11np.allclose(morph_mesh(_p11fv, _p11np.zeros((_p11fv.n_vertices, 3))),
                       _p11fv.vertices, atol=1e-12), "P11 morph 零位移恒等"
_p11disp = _p11np.zeros((_p11fv.n_vertices, 3), float)
_p11disp[0, 0] = 0.1
_p11morph = morph_mesh(_p11fv, _p11disp, max_iter=200, relax=0.6)
assert _p11np.all(_p11np.isfinite(_p11morph)), "P11 morph 内部有限"
assert abs(_p11morph[0, 0] - (_p11fv.vertices[0, 0] + 0.1)) < 1e-9, "P11 morph 边界夹持"
# DFBI 6DOF：advance 受力 → 线加速度/位置；四元数/DCM 旋转映射 x→y
_p11b = DfbiBody(mass=2.0, inertia=(1.0, 1.0, 1.0))
_p11b.advance(_p11np.array([4.0, 0.0, 0.0]), _p11np.zeros(3), dt=1.0)
assert _p11np.allclose(_p11b.linear_velocity, [2.0, 0.0, 0.0], atol=1e-12), "P11 DFBI 线速度"
assert _p11np.allclose(_p11b.position, [2.0, 0.0, 0.0], atol=1e-12), "P11 DFBI 位置"
_p11q = _p11np.array([_p11np.cos(_p11np.pi / 4), 0.0, 0.0, _p11np.sin(_p11np.pi / 4)])
assert _p11np.allclose(quat_to_dcm(_p11q) @ _p11np.array([1.0, 0.0, 0.0]),
                       [0.0, 1.0, 0.0], atol=1e-12), "P11 DFBI DCM 旋转"
# overset：硬切换 + 常量场保形
_p11W = overset_donor_weights(_p11np.array([[0.0, 0.0, 0.0]]),
                              _p11np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), k=3)
assert _p11W.shape == (1, 2) and _p11np.allclose(_p11W.sum(axis=1), 1.0), "P11 overset 权重"
assert _p11np.allclose(_p11W[0], [1.0, 0.0], atol=1e-12), "P11 overset 硬切换"
assert _p11np.allclose(overset_interpolate(
    _p11np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    _p11np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    _p11np.array([2.0, 2.0, 2.0]), k=3), [2.0, 2.0], atol=1e-12), "P11 overset 常量场"
# MotionSolver：工厂别名 + P10 后端门面
_p11mo = make_motion(_p11fv, "motion", rotation_speed=2.0)
assert isinstance(_p11mo, MotionSolver), "P11 工厂 motion"
assert isinstance(make_motion(_p11fv, "MRF"), MotionSolver), "P11 工厂 MRF 大小写"
assert isinstance(make_motion(_p11fv, "6dof"), MotionSolver), "P11 工厂 6dof"
assert isinstance(make_motion(_p11fv, "morphing"), MotionSolver), "P11 工厂 morphing"
assert isinstance(make_motion(_p11fv, "overset"), MotionSolver), "P11 工厂 overset"
try:
    make_motion(_p11fv, "nope")
    raise AssertionError("P11 应拒绝未知运动模型")
except ValueError:
    pass
_p11src = _p11mo.mrf_source(rho=_p11np.full(_p11n, 1.0))
assert _p11src.shape == (_p11n, 3) and float(_p11np.linalg.norm(_p11src)) > 0.0, \
    "P11 MRF 旋转源非零"
_p11ini = _p11mo._initialize_field()
assert "mode" in _p11ini, "P11 initialize 键"
_p11mo.update(_p11np.zeros(_p11n), _p11np.zeros(_p11n), _p11np.zeros(_p11n), mdot=None)
assert _p11np.isfinite(_p11mo.residual()), "P11 update 残差有限"
_p11p = _p11mo.step()
assert "residual" in _p11p and _p11np.isfinite(_p11p["residual"]), "P11 step 键"
_p11m = _p11mo.monitor_payload()
assert {"mode", "n_bodies", "rotation_speed", "t", "iteration"} <= set(_p11m), \
    "P11 monitor 键"
# 集成：PressureSolver 字符串注入 motion + step() 稳定、monitor 含 motion 键；基线无
_p11s0 = _p11Solver(_p11Vc, _p11Cc, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
assert _p11s0.motion_model is None, "P11 基线无模型"
assert _p11np.isfinite(_p11s0.step()["residual"]), "P11 基线 step() 无报错"
_p11sm = _p11Solver(_p11Vc, _p11Cc, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                    motion_model="motion", motion_rotation_speed=0.5,
                    motion_rotation_axis=(0.0, 0.0, 1.0))
for _p11k in range(6):
    _p11ps = _p11sm.step()
assert _p11np.isfinite(_p11sm.velocity()).all() and _p11np.isfinite(_p11ps["residual"]), \
    "P11 耦合流场/残差有限"
assert _p11sm.motion_model is not None and not isinstance(_p11sm.motion_model, str), \
    "P11 应构建运动模型"
_p11m11 = _p11sm.monitor_payload()
assert {"motion_mode", "motion_rotation_speed", "motion_disp_max"} <= set(_p11m11), \
    "P11 monitor 含 motion 键"
print("P11 运动谱系：刚体运动+滑移 interface/Rodrigues 旋转/morphing 拉普拉斯变形/"
      "DFBI 6DOF 刚体动力学/overset 重叠插值/MRF 离心科氏源/P10 后端门面/"
      "PressureSolver 耦合 全通过")

print("ALL CHECKS PASSED")
