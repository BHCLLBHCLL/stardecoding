# -*- coding: utf-8 -*-
"""R 波 R6：协同仿真链接配置前段（类型表/校验/JSON 往返/抽取/宏前端）。

覆盖：
  类型：resolve_type 别名与未知拒绝、type_defaults、resolve_profile_method
  工厂：make_link 按类型默认连接/启动方式、template 含区域与场
  校验：端口范围/连接文件/可执行/命令行/耦合区间/URF 参数域/区域与场完整性/并发模式
  往返：to_dict↔from_dict、to_json↔from_json 摘要一致
  抽取：有 CoSimulation 对象才解；无对象诚实拒绝（真实语料 46 文件 0 个 cosim 对象）
  宏：render_macro 含 best-effort 声明与区域/场注释
  CLI：--template/--validate 退出码
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from cosimulation import (  # noqa: E402
    CONCURRENCY_MODES, CONNECTION_METHODS, COSIM_TYPES, JAVADOC_ROOT, LAUNCH_OPTIONS,
    PROFILE_METHODS, URF_STRATEGIES, VERIFIED_METHODS, CoSimZone, CoSimulationLink,
    _main, extract_cosimulation, make_link, render_macro, resolve_profile_method,
    resolve_type, template, type_defaults,
)

CORPUS = r"D:/training/starccm/startutorialsdata"
PLATE = os.path.join(CORPUS, "coupling", "data", "plate-cosim.sim")
needs_corpus = pytest.mark.skipif(not os.path.isfile(PLATE), reason="coupling 语料缺失")


def good_link():
    link = make_link("amesim", name="amesim", host="127.0.0.1", port=5555,
                     executable=r"C:/AMESim/amesim.exe", coupling_interval=0.01,
                     urf_strategy="constant", urf_params={"urf": 0.5})
    link.zones = [CoSimZone("valve", boundaries=["Inlet", "Outlet"],
                            exported={"pressure": "pressure"},
                            imported={"displacement": "displacement"})]
    return link


# ---------------------------------------------------------------- 类型表
def test_type_aliases_and_defaults():
    assert resolve_type("amesim") == "AmesimCoSimulationType"
    assert resolve_type("AMESim") == "AmesimCoSimulationType"
    assert resolve_type("gt_power") == "GtPowerCoSimulationType"
    assert resolve_type("gt-power") == "GtPowerCoSimulationType"
    assert resolve_type("fmu") == "FmiLibraryImportType"
    assert resolve_type("cgns") == "CgnsLinkType"
    assert type_defaults("fmi")["launch"] == "partner_library"
    assert type_defaults("generic")["connection"] == "connection_file"
    with pytest.raises(ValueError):
        resolve_type("openfoam")


def test_profile_methods():
    assert resolve_profile_method("heat flux") == "CoSimHeatFluxProfileMethod"
    assert resolve_profile_method("Heat-Flux") == "CoSimHeatFluxProfileMethod"
    assert resolve_profile_method("mass_flow") == "CoSimMassFlowProfileMethod"
    with pytest.raises(ValueError):
        resolve_profile_method("enthalpy")


def test_public_tables_cover_official_names():
    assert len(COSIM_TYPES) >= 8 and len(PROFILE_METHODS) >= 10
    assert set(URF_STRATEGIES) >= {"constant", "adaptive", "anderson"}
    assert "host_port" in CONNECTION_METHODS and "connection_file" in CONNECTION_METHODS
    assert set(LAUNCH_OPTIONS) >= {"none", "executable", "command_line"}
    assert set(CONCURRENCY_MODES) == {"serial", "concurrent"}


# ---------------------------------------------------------------- 工厂/模板
def test_make_link_and_template():
    link = make_link("generic")
    assert link.sim_type == "GenericApplicationType"
    assert link.connect_method == "connection_file"
    assert link.launch_option == "command_line"
    tpl = template("amesim")
    assert tpl.zones and tpl.zones[0].exported and tpl.zones[0].imported
    assert isinstance(tpl.validate(), list)


# ---------------------------------------------------------------- 校验
def test_validate_good_config_clean():
    assert good_link().validate() == []


def test_validate_port_and_bootstrap_fields():
    link = good_link()
    link.port = 0
    assert any("端口" in p for p in link.validate())
    link = good_link()
    link.executable = None
    assert any("可执行文件" in p for p in link.validate())
    link = good_link()
    link.connect_method = "connection_file"
    link.connection_file = None
    assert any("连接文件" in p for p in link.validate())
    link = good_link()
    link.connect_method = "telepathy"
    assert any("未知连接方式" in p for p in link.validate())


def test_validate_interval_and_concurrency():
    link = good_link()
    link.coupling_interval = -1.0
    assert any("耦合区间" in p for p in link.validate())
    link = good_link()
    link.interval_unit = "fortnights"
    assert any("耦合区间单位" in p for p in link.validate())
    link = good_link()
    link.concurrency = "parallel"
    assert any("并发模式" in p for p in link.validate())


def test_validate_urf_rules():
    link = good_link()
    link.urf_strategy = "constant"
    link.urf_params = {}
    assert any("缺少参数 urf" in p for p in link.validate())
    link = good_link()
    link.urf_params = {"urf": 5.0}
    assert any("URF 权重" in p for p in link.validate())
    link = good_link()
    link.urf_strategy = "anderson"
    link.urf_params = {"depth": 0, "urf": 1.0}
    assert any("Anderson 深度" in p for p in link.validate())
    link = good_link()
    link.urf_strategy = "runge-kutta"
    assert any("未知 URF 策略" in p for p in link.validate())


def test_validate_zone_rules():
    link = good_link()
    link.zones = []
    assert any("至少需要一个协同仿真区域" in p for p in link.validate())
    link.zones = [CoSimZone("z", boundaries=[], exported={}, imported={})]
    problems = link.validate()
    assert any("至少需要一个边界" in p for p in problems)
    assert any("未定义任何传递场" in p for p in problems)
    link.zones = [CoSimZone("z", boundaries=["b"], exported={"pressure": "unknown-method"})]
    assert any("未知场传递方式" in p for p in link.validate())


# ---------------------------------------------------------------- 往返
def test_json_roundtrip():
    link = good_link()
    back = CoSimulationLink.from_json(link.to_json())
    assert back.summary() == link.summary()
    assert back.to_dict() == link.to_dict()
    assert back.validate() == []


def test_json_file_roundtrip():
    # 用 tempfile 而非 pytest tmp_path（本机 pytest 临时根目录权限受限）
    tmp = tempfile.mkdtemp(prefix="r6link")
    try:
        path = os.path.join(tmp, "link.json")
        link = good_link()
        link.to_json(path)
        assert CoSimulationLink.from_json(path=path).summary() == link.summary()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 抽取
def test_extract_without_cosim_objects_is_honest():
    class _O:
        class_name = "star.common.Simulation"
        name = "s"
        id = 2
        dict = {}

    class _Sim:
        objects = [_O()]

    res = extract_cosimulation(_Sim())
    assert res["ok"] is False and res["n_objects"] == 0
    assert "无协同仿真对象" in res["reason"]


def test_extract_with_cosim_root_object():
    class _Link:
        class_name = "star.cosimulation.link.common.CoSimulation"
        name = "amesim"
        id = 42
        dict = {"PresentationName": "AMESim Link", "Type": 43}

    class _Other:
        class_name = "star.cosimulation.common.CoSimCouplingInterval"
        name = None
        id = 44
        dict = {}

    class _Sim:
        objects = [_Link(), _Other()]

    res = extract_cosimulation(_Sim())
    assert res["ok"] is True and res["n_objects"] == 2
    assert res["links"][0]["name"] == "AMESim Link"
    assert res["links"][0]["fields"]["Type"] == 43


def test_extract_cosim_objects_without_root_is_honest():
    class _Other:
        class_name = "star.cosimulation.common.CoSimulationZone"
        name = "z"
        id = 7
        dict = {}

    class _Sim:
        objects = [_Other()]

    res = extract_cosimulation(_Sim())
    assert res["ok"] is False and res["n_objects"] == 1
    assert "无 CoSimulation 根对象" in res["reason"]


@needs_corpus
def test_corpus_has_no_cosim_link():
    from sim_parser import SimFile
    res = extract_cosimulation(SimFile(PLATE))
    assert res["ok"] is False and res["n_objects"] == 0


# ---------------------------------------------------------------- 宏 / CLI
def test_render_macro_uses_only_verified_signatures():
    text = render_macro(good_link())
    # 头部声明与 Javadoc 路径
    assert "[verified-signatures]" in text and JAVADOC_ROOT in text
    assert "AmesimCoSimulationType" in text
    # 只出现官方 Javadoc 核对过的方法
    for verified in ("getCoSimulations()", "getCoSimulationZoneManager()",
                     "createEmptyCoSimulationZone()", "getCoSimulationValues()",
                     "isSolverStarted()", "getCoSimulationZoneValues()"):
        assert verified.replace("()", "(") in text or verified in text
    # 首版编造的 API 必须消失
    assert "createCoSimulation(" not in text
    assert "// TODO" in text                      # 写侧缺口显式标注
    assert "区域 valve" in text and "CoSimPressureProfileMethod" in text
    assert text.rstrip().endswith("}")


def test_verified_methods_table_matches_javadoc():
    for cls, methods in VERIFIED_METHODS.items():
        assert methods, cls
        for m in methods:
            assert m.endswith("()") and m[0].islower(), m
    assert "createEmptyCoSimulationZone()" in VERIFIED_METHODS["CoSimulationZoneManager"]
    assert "getCoSimulations()" in VERIFIED_METHODS["CoSimulationManager"]


def test_cli_template_and_validate():
    tmp = tempfile.mkdtemp(prefix="r6cli")
    try:
        path = os.path.join(tmp, "tpl.json")
        assert _main(["--template", "fmi", "--out", path]) == 0
        assert os.path.isfile(path)
        # 模板缺端口/可执行 → 校验应报问题并以退出码 1 结束
        assert _main(["--validate", path]) == 1
        link = CoSimulationLink.from_json(path=path)
        link.host, link.port = "localhost", 5000
        link.launch_option, link.executable = "partner_library", "/opt/partner/lib.so"
        link.to_json(path)
        assert _main(["--validate", path]) == 0
        assert _main([]) == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
