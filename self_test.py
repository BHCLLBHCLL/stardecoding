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

print("ALL CHECKS PASSED")
