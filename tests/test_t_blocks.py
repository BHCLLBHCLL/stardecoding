# -*- coding: utf-8 -*-
"""R 波 R3：二进制状态表 T 载荷文法（A/B 几何记录 + 覆盖率 + 坐标真值校验）。

覆盖：
  常量：T_GEOM_A_SIZE/T_GEOM_B_SIZE/容器标记字节模式/标记置信度表
  扫描：A+B 成对记录、截断不误判、字节级重同步、B-only 记录、无顶点表时 verified=None
  不变量：v0=a+3 / v1=a+4 / v2=a-4 与 count∈{3,4} 的 conform 判定；B 首字段链接 A.v0
  语料：manifold_start（15 几何记录 / 15 命中 Float8 顶点表 / 14 索引不变量 /
        133 容器开 + 196 容器闭）、airfoil 与 vibratingPipe（0 几何记录，诚实零）
  ASCII：adjointWing（T 载荷走 G1 元素流口径）
  报告：t_block_report 字段与"无几何则不报命中率"的诚实口径

验收核心（R3 行）：T 载荷几何记录可解码，未解字节如实计数。
"""
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from sim_parser import (  # noqa: E402
    T_BINARY_MARKERS, T_CONTAINER_CLOSE, T_CONTAINER_MAX_COUNT, T_CONTAINER_OPEN,
    T_CONTAINER_SIZE, T_GEOM_A_SIZE, T_GEOM_B_SIZE, SimFile, _t_scan_binary,
    decode_t_blocks, t_block_report,
)

CORPUS = r"D:/training/starccm/startutorialsdata"
MANIFOLD = os.path.join(CORPUS, "solidStress", "data", "manifold_start.sim")
AIRFOIL = os.path.join(CORPUS, "optimate", "data", "airfoil.sim")
VIBPIPE = os.path.join(CORPUS, "solidStress", "data", "vibratingPipe_start.sim")
ADJWING = os.path.join(CORPUS, "adjoint", "data", "adjointWing_start.sim")

needs_manifold = pytest.mark.skipif(not os.path.isfile(MANIFOLD),
                                    reason="语料 manifold_start.sim 缺失")
needs_binary = pytest.mark.skipif(
    not (os.path.isfile(AIRFOIL) and os.path.isfile(VIBPIPE)),
    reason="二进制语料缺失")
needs_adjwing = pytest.mark.skipif(not os.path.isfile(ADJWING),
                                   reason="语料 adjointWing_start.sim 缺失")


# ---------------------------------------------------------------- 构造工具
def a_record(count=4, base=100, ref=5000, vids=None, xyz=(1.0, 2.0, 3.0)):
    if vids is None:
        vids = (base + 3, base + 4, base - 4)
    return struct.pack(">9H3d", count, 29, base, 0, ref, 1,
                       vids[0], vids[1], vids[2], xyz[0], xyz[1], xyz[2])


def b_record(v0=103, ref=6000, val=0.5):
    return struct.pack(">9Hd", 18, v0, 0, ref, v0 + 2, v0 + 3, v0 - 4, v0 + 4, v0 + 1, val)


# ---------------------------------------------------------------- 常量
def test_constants():
    assert T_GEOM_A_SIZE == 42 and T_GEOM_B_SIZE == 26
    assert T_BINARY_MARKERS[18] == ("geom-companion", "confirmed")
    assert T_BINARY_MARKERS[29] == ("geometry-element", "confirmed")
    assert T_BINARY_MARKERS[81][0] == "container-open"
    assert T_BINARY_MARKERS[82][0] == "container-close"
    assert T_CONTAINER_OPEN == b"\x00\x51\x00\x00\x00\x01"
    assert T_CONTAINER_CLOSE == b"\x00\x52\x00\x00\x00\x01"
    assert T_CONTAINER_SIZE == 8 and T_CONTAINER_MAX_COUNT == 64
    assert T_BINARY_MARKERS[29][1] == "confirmed"
    assert len(a_record()) == T_GEOM_A_SIZE
    assert len(b_record()) == T_GEOM_B_SIZE


# ---------------------------------------------------------------- 扫描
def test_scan_ab_pair_verified():
    xyz = (0.25, -1.5, 3.75)
    blob = a_record(count=4, base=100, ref=5000, xyz=xyz) + b_record(v0=103)
    vset = {tuple(round(v, 9) for v in xyz)}
    recs, attr = _t_scan_binary(blob, vset)
    assert attr == T_GEOM_A_SIZE + T_GEOM_B_SIZE
    assert len(recs) == 1
    r = recs[0]
    assert r["kind"] == "geom-A" and r["count"] == 4 and r["base"] == 100
    assert r["ref"] == 5000 and r["vids"] == [103, 104, 96]
    assert r["conform"] is True and r["verified"] is True
    assert r["b"]["ints"][0] == 103 and r["b"]["links_a"] is True
    assert abs(r["b"]["value"] - 0.5) < 1e-12
    assert [round(v, 9) for v in r["xyz"]] == [round(v, 9) for v in xyz]


def test_scan_without_vertex_set_is_unverified():
    recs, attr = _t_scan_binary(a_record() + b_record(), None)
    assert attr == 68 and recs[0]["verified"] is None


def test_scan_truncated_a_is_not_false_positive():
    recs, attr = _t_scan_binary(a_record()[:30], set())
    assert recs == [] and attr == 0


def test_scan_resyncs_after_noise():
    blob = b"\x01\x02\x03" + a_record(base=7) + b_record(v0=10)
    recs, attr = _t_scan_binary(blob, set())
    assert len(recs) == 1 and recs[0]["base"] == 7
    assert recs[0]["off"] == 3 and attr == 68


def test_scan_container_elements():
    blob = struct.pack(">HI H", 81, 1, 432) + struct.pack(">HI H", 82, 3, 433)
    assert len(blob) == 2 * T_CONTAINER_SIZE
    recs, attr = _t_scan_binary(blob, set())
    assert attr == 2 * T_CONTAINER_SIZE
    assert [r["kind"] for r in recs] == ["container-open", "container-close"]
    assert recs[0]["count"] == 1 and recs[0]["id"] == 432
    assert recs[1]["count"] == 3 and recs[1]["id"] == 433
    assert recs[0]["resolved"] is None      # 无 objmap 时不假装解析


def test_scan_container_count_out_of_range_rejected():
    blob = struct.pack(">HI H", 81, 10 ** 6, 432)
    recs, attr = _t_scan_binary(blob, set())
    assert recs == [] and attr == 0          # 计数越界 → 不当容器元素吞掉字节


def test_scan_container_id_resolves_against_objmap():
    class _O:
        class_name = "star.common.NameManager"

    blob = struct.pack(">HI H", 81, 1, 432)
    recs, _ = _t_scan_binary(blob, set(), {432: _O()})
    assert recs[0]["resolved"] == "star.common.NameManager"


def test_scan_b_only_record():
    blob = b_record(v0=42)
    recs, attr = _t_scan_binary(blob, set())
    assert len(recs) == 1 and recs[0]["kind"] == "geom-B"
    assert recs[0]["ints"][0] == 42 and attr == T_GEOM_B_SIZE


def test_scan_variant_vids_flagged_nonconforming():
    blob = a_record(base=100, vids=(1, 2, 3)) + b_record(v0=42)
    recs, _ = _t_scan_binary(blob, set())
    assert recs[0]["conform"] is False
    assert recs[0]["vids"] == [1, 2, 3]
    assert recs[0]["b"]["links_a"] is False


def test_scan_count_outside_3_4_flagged():
    blob = a_record(count=5) + b_record()
    recs, _ = _t_scan_binary(blob, set())
    assert recs[0]["count"] == 5 and recs[0]["conform"] is False


# ---------------------------------------------------------------- 语料
@needs_manifold
def test_corpus_manifold_geometry_records():
    rep = t_block_report(SimFile(MANIFOLD))
    assert rep["ok"] and rep["mode"] == "binary"
    assert rep["n_t_records"] == 1264 and rep["n_with_payload"] == 1224
    assert rep["bytes"] == 24469
    assert rep["n_geometry"] == 15
    assert rep["n_verified"] == 15 and rep["verify_rate_pct"] == 100.0
    assert rep["n_conform"] == 14 and rep["conform_rate_pct"] == 93.3
    assert 10 < rep["coverage_pct"] < 25         # 未解字节仍占多数，如实计数
    att = rep["attribution"]
    assert sum(att.values()) == rep["bytes"]
    assert att["geom-A+B"] == 1020 and att["geom-B"] == 208 and att["container"] == 2648
    assert rep["markers"]["geometry-element"] == 15
    assert rep["markers"]["geom-companion"] == 23
    assert rep["markers"]["container-open"] == 159
    assert rep["markers"]["container-close"] == 172
    assert rep["container_ids"] == 331 and rep["container_ids_resolved"] == 155
    assert rep["residue_top_u16"]                      # 残差画像非空（事实统计）


@needs_binary
def test_corpus_binary_files_honest_zero():
    for path, nbytes in ((AIRFOIL, 2250), (VIBPIPE, 613)):
        rep = t_block_report(SimFile(path))
        assert rep["ok"] and rep["mode"] == "binary"
        assert rep["bytes"] == nbytes
        assert rep["n_geometry"] == 0
        assert rep["verify_rate_pct"] is None    # 无几何 → 不报命中率（诚实）
    assert t_block_report(SimFile(AIRFOIL))["coverage_pct"] < 10.0    # 仅容器元素
    assert t_block_report(SimFile(VIBPIPE))["coverage_pct"] == 0.0    # 完全未解


@needs_adjwing
def test_corpus_ascii_t_payload_elements():
    rep = t_block_report(SimFile(ADJWING))
    assert rep["ok"] and rep["mode"] == "ascii"
    assert rep["n_t_records"] == 2 and rep["n_with_payload"] == 0
    assert rep["bytes"] == 66 and rep["coverage_pct"] == 100.0


@needs_manifold
def test_decode_t_blocks_records_view():
    dec = decode_t_blocks(SimFile(MANIFOLD), max_records=2)
    assert dec["ok"] and len(dec["records"]) <= 2
    for rec in dec["records"]:
        assert set(rec) >= {"token_index", "n_bytes", "attributed", "geometry", "markers"}
        assert rec["attributed"] <= rec["n_bytes"]
        for g in rec["geometry"]:
            assert g["kind"] in ("geom-A", "geom-B")
            if g["kind"] == "geom-A":
                assert len(g["vids"]) == 3 and len(g["xyz"]) == 3


@needs_manifold
# ---------------------------------------------------------------- 流模型（残差结构分解）
@needs_manifold
def test_stream_report_manifold_structure_and_semantics():
    from sim_parser import t_stream_report
    st = t_stream_report(SimFile(MANIFOLD))
    assert st["ok"] and st["bytes"] == 24469
    assert 15.0 < st["attributed_pct"] < 22.0          # 流模型下元素归属高于逐记录
    assert 30.0 < st["double_pct"] < 50.0
    assert 35.0 < st["u16_pct"] < 50.0
    assert st["unknown_bytes"] < 64 and st["structural_pct"] > 99.5
    assert st["double_hit_pct"] > 15.0                 # 命中对象图数值（随机基线≈0）
    assert st["u16_hit_pct"] > st["u16_chance_pct"]    # 命中对象 id 高于随机基线
    assert st["n_gaps"] > 0 and st["n_elements"] > 0


@needs_binary
def test_stream_report_binary_files_structure():
    from sim_parser import t_stream_report
    for path, nbytes in ((AIRFOIL, 2250), (VIBPIPE, 613)):
        st = t_stream_report(SimFile(path))
        assert st["ok"] and st["bytes"] == nbytes
        assert st["structural_pct"] > 99.0
        assert st["unknown_bytes"] < 16
        assert st["u16_hit_pct"] > st["u16_chance_pct"]


@needs_adjwing
def test_stream_report_rejects_ascii():
    from sim_parser import t_stream_report
    st = t_stream_report(SimFile(ADJWING))
    assert st["ok"] is False and "非二进制" in st["reason"]


def test_report_shape_and_honesty():
    rep = t_block_report(SimFile(MANIFOLD))
    for key in ("ok", "mode", "n_t_records", "bytes", "attributed_bytes",
                "coverage_pct", "n_geometry", "n_verified", "n_conform",
                "verify_rate_pct", "conform_rate_pct", "markers"):
        assert key in rep
    assert rep["attributed_bytes"] < rep["bytes"]
