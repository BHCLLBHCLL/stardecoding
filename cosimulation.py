# -*- coding: utf-8 -*-
"""R 波 R6：协同仿真链接配置前段（对齐 star.cosimulation.link.common / .common 官方类表）。

语料事实（46 个 .sim：21 教程 + 25 openfoam benchmark）：**star.cosimulation.* 对象 0 个** ——
coupling 教程的 .sim 是**耦合前状态**（Simulation 名 "plate-cosim_start" / "amesimNewstarting"），
链接是运行期经 CoSim API 建立，不落盘在 .sim 里。故本模块交付**配置前段**：
  ① 链接配置模型（主机/端口、连接文件、可执行文件与命令行、耦合区间、URF 策略、区域与场映射、
     并发模式、时间步调整），字段名与取值域依官方类表；
  ② 校验（端口范围、连接方式必填项、区域/场映射完整性、URF 参数域）→ 问题清单（不抛异常硬崩）；
  ③ JSON 往返（前端配置可存可读）；
  ④ 从 .sim 对象图抽取（**有对象才解，无对象诚实拒绝**）；
  ⑤ Java 宏前端渲染（类名依官方类表；方法签名未逐条核对 → 标注 best-effort，需许可环境核验）。

诚实边界：不解协议、不建连接、不依赖三方求解器（计划 §1 维度 17 的 "仅做配置解析前段"）。
"""
import json
import os
import sys

# ------------------------------------------------------------------ 类型表
# 依 work_pkgs/star__cosimulation__link__common.txt 的官方类清单
COSIM_TYPES = {
    "amesim": ("AmesimCoSimulationType", {"launch": "executable", "connection": "host_port"}),
    "abaqus": ("AbaqusCoSimulationType", {"launch": "executable", "connection": "host_port"}),
    "gtpower": ("GtPowerCoSimulationType", {"launch": "executable", "connection": "host_port"}),
    "gt-power": ("GtPowerCoSimulationType", {"launch": "executable", "connection": "host_port"}),
    "fmi": ("FmiLibraryImportType", {"launch": "partner_library",
                                     "connection": "host_port"}),
    "fmu": ("FmiLibraryImportType", {"launch": "partner_library",
                                     "connection": "host_port"}),
    "generic": ("GenericApplicationType", {"launch": "command_line",
                                           "connection": "connection_file"}),
    "cgns": ("CgnsLinkType", {"launch": "none", "connection": "connection_file"}),
}

# 依 star.cosimulation.common 的 *ProfileMethod 类清单
PROFILE_METHODS = {
    "pressure": "CoSimPressureProfileMethod",
    "traction": "CoSimTractionProfileMethod",
    "force": "CoSimForceProfileMethod",
    "temperature": "CoSimTemperatureProfileMethod",
    "total_temperature": "CoSimTotalTemperatureProfileMethod",
    "heat_flux": "CoSimHeatFluxProfileMethod",
    "heat_transfer_coefficient": "CoSimHeatTransferCoefficientProfileMethod",
    "mass_flow": "CoSimMassFlowProfileMethod",
    "mass_fraction": "CoSimMassFractionProfileMethod",
    "passive_scalar": "CoSimPassiveScalarProfileMethod",
    "displacement": "CoSimFieldDisplacementProfileMethod",
    "density": "CoSimDensityProfileMethod",
}

URF_STRATEGIES = {
    "constant": ("CoSimConstantUrf", {"urf": float}),
    "constant-expert": ("CoSimConstantUrfExpert", {"urf": float}),
    "adaptive": ("CoSimAdaptiveUrf", {"urf_min": float, "urf_max": float}),
    "adaptive-expert": ("CoSimAdaptiveUrfExpert", {"urf_min": float, "urf_max": float}),
    "anderson": ("CoSimAndersonUrf", {"depth": int, "urf": float}),
    "imported-field": ("CoSimImportedFieldUrfOption", {}),
}

CONNECTION_METHODS = ("host_port", "assigned_host_port", "connection_file", "command_line")
LAUNCH_OPTIONS = ("none", "executable", "command_line", "partner_library")
CONCURRENCY_MODES = ("serial", "concurrent")


def resolve_type(name):
    """协同仿真类型别名 → 官方类名（大小写/连字符/下划线不敏感）。"""
    if name is None:
        raise ValueError("协同仿真类型不能为空")
    key = str(name).strip().lower().replace("_", "-")
    if key in COSIM_TYPES:
        return COSIM_TYPES[key][0]
    raise ValueError("未知协同仿真类型: %r（可选 %s）" % (name, ", ".join(sorted(COSIM_TYPES))))


def type_defaults(name):
    key = str(name).strip().lower().replace("_", "-")
    return dict(COSIM_TYPES.get(key, (None, {}))[1])


def resolve_profile_method(name):
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key in PROFILE_METHODS:
        return PROFILE_METHODS[key]
    raise ValueError("未知场传递方式: %r（可选 %s）" % (name, ", ".join(sorted(PROFILE_METHODS))))


# ------------------------------------------------------------------ 配置部件
class CoSimZone(object):
    """协同仿真区域：区域/边界 ↔ 外场，含导出/导入场与传递方式。"""

    def __init__(self, name, zone_type="boundary", region=None, boundaries=None,
                 exported=None, imported=None):
        self.name = name
        self.zone_type = zone_type
        self.region = region
        self.boundaries = list(boundaries or [])
        self.exported = dict(exported or {})      # 场名 → profile method 键
        self.imported = dict(imported or {})

    def field_pairs(self):
        return sorted(set(self.exported) | set(self.imported))

    def validate(self, prefix="zone"):
        problems = []
        if not self.name:
            problems.append("%s: 区域名不能为空" % prefix)
        if self.zone_type not in ("boundary", "region", "shell"):
            problems.append("%s(%s): 未知区域类型 %r" % (prefix, self.name, self.zone_type))
        if not self.boundaries and self.zone_type == "boundary":
            problems.append("%s(%s): 边界型区域至少需要一个边界" % (prefix, self.name))
        for kind, table in (("exported", self.exported), ("imported", self.imported)):
            for field_name, method in table.items():
                try:
                    resolve_profile_method(method)
                except ValueError as exc:
                    problems.append("%s(%s): %s 场 %s → %s"
                                    % (prefix, self.name, kind, field_name, exc))
        if not self.field_pairs():
            problems.append("%s(%s): 未定义任何传递场（导出/导入均为空）" % (prefix, self.name))
        return problems

    def to_dict(self):
        return {"name": self.name, "zone_type": self.zone_type, "region": self.region,
                "boundaries": list(self.boundaries),
                "exported": dict(self.exported), "imported": dict(self.imported)}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("name"), d.get("zone_type", "boundary"), d.get("region"),
                   d.get("boundaries"), d.get("exported"), d.get("imported"))

    def summary(self):
        return {"name": self.name, "type": self.zone_type, "region": self.region,
                "n_boundaries": len(self.boundaries),
                "exported": dict(self.exported), "imported": dict(self.imported)}


class CoSimulationLink(object):
    """协同仿真链接（一个外部伙伴）。字段依官方 star.cosimulation.link.common。"""

    def __init__(self, name, sim_type, partner=None, connect_method="host_port",
                 host="localhost", port=0, connection_file=None, executable=None,
                 command_line=None, launch_option=None, coupling_interval=0.0,
                 interval_unit="seconds", urf_strategy="constant", urf_params=None,
                 concurrency="serial", time_step_adjust=True, zones=None):
        self.name = name
        self.sim_type = resolve_type(sim_type)
        self.type_key = str(sim_type).strip().lower().replace("_", "-")
        self.partner = partner
        self.connect_method = connect_method
        self.host = host
        self.port = int(port or 0)
        self.connection_file = connection_file
        self.executable = executable
        self.command_line = command_line
        self.launch_option = launch_option or type_defaults(sim_type).get("launch", "executable")
        self.coupling_interval = float(coupling_interval or 0.0)
        self.interval_unit = interval_unit
        self.urf_strategy = urf_strategy
        self.urf_params = dict(urf_params or {})
        self.concurrency = concurrency
        self.time_step_adjust = bool(time_step_adjust)
        self.zones = list(zones or [])

    # ---------------------------------------------------------- 校验
    def validate(self):
        p = []
        if not self.name:
            p.append("链接名不能为空")
        if self.connect_method not in CONNECTION_METHODS:
            p.append("未知连接方式 %r（可选 %s）" % (self.connect_method,
                                                ", ".join(CONNECTION_METHODS)))
        if self.connect_method in ("host_port", "assigned_host_port"):
            if not self.host:
                p.append("连接方式 %s 需要主机名" % self.connect_method)
            if not (1 <= self.port <= 65535):
                p.append("端口须在 1..65535（当前 %r）" % self.port)
        if self.connect_method == "connection_file":
            if not self.connection_file:
                p.append("连接方式 connection_file 需要连接文件路径")
            elif os.path.splitext(str(self.connection_file))[1].lower() not in (".cosim", ".ccm", ".xml", ".json"):
                p.append("连接文件扩展名可疑: %s" % self.connection_file)
        if self.launch_option not in LAUNCH_OPTIONS:
            p.append("未知启动方式 %r（可选 %s）" % (self.launch_option, ", ".join(LAUNCH_OPTIONS)))
        if self.launch_option == "executable" and not self.executable:
            p.append("启动方式 executable 需要可执行文件路径")
        if self.launch_option == "command_line" and not self.command_line:
            p.append("启动方式 command_line 需要命令行")
        if self.coupling_interval < 0:
            p.append("耦合区间不能为负（当前 %r）" % self.coupling_interval)
        if self.interval_unit not in ("seconds", "iterations", "time_step"):
            p.append("未知耦合区间单位 %r" % self.interval_unit)
        if self.urf_strategy not in URF_STRATEGIES:
            p.append("未知 URF 策略 %r（可选 %s）" % (self.urf_strategy,
                                                ", ".join(sorted(URF_STRATEGIES))))
        else:
            spec = URF_STRATEGIES[self.urf_strategy][1]
            for key, caster in spec.items():
                if key not in self.urf_params:
                    p.append("URF 策略 %s 缺少参数 %s" % (self.urf_strategy, key))
                    continue
                try:
                    val = caster(self.urf_params[key])
                except (TypeError, ValueError):
                    p.append("URF 参数 %s 类型应为 %s" % (key, caster.__name__))
                    continue
                if key == "urf" and not (0.0 < val < 2.0):
                    p.append("URF 权重应在 (0,2)（当前 %r）" % val)
                if key == "depth" and val < 1:
                    p.append("Anderson 深度应 ≥1（当前 %r）" % val)
        if self.concurrency not in CONCURRENCY_MODES:
            p.append("未知并发模式 %r（可选 %s）" % (self.concurrency,
                                                ", ".join(CONCURRENCY_MODES)))
        if not self.zones:
            p.append("至少需要一个协同仿真区域")
        for i, z in enumerate(self.zones):
            p.extend(z.validate("zone[%d]" % i))
        return p

    # ---------------------------------------------------------- 往返
    def to_dict(self):
        return {"name": self.name, "sim_type": self.sim_type, "type_key": self.type_key,
                "partner": self.partner, "connect_method": self.connect_method,
                "host": self.host, "port": self.port,
                "connection_file": self.connection_file, "executable": self.executable,
                "command_line": self.command_line, "launch_option": self.launch_option,
                "coupling_interval": self.coupling_interval,
                "interval_unit": self.interval_unit, "urf_strategy": self.urf_strategy,
                "urf_params": dict(self.urf_params), "concurrency": self.concurrency,
                "time_step_adjust": self.time_step_adjust,
                "zones": [z.to_dict() for z in self.zones]}

    @classmethod
    def from_dict(cls, d):
        link = cls(d.get("name"), d.get("type_key") or d.get("sim_type"),
                   partner=d.get("partner"), connect_method=d.get("connect_method",
                                                                  "host_port"),
                   host=d.get("host", "localhost"), port=d.get("port", 0),
                   connection_file=d.get("connection_file"),
                   executable=d.get("executable"), command_line=d.get("command_line"),
                   launch_option=d.get("launch_option"),
                   coupling_interval=d.get("coupling_interval", 0.0),
                   interval_unit=d.get("interval_unit", "seconds"),
                   urf_strategy=d.get("urf_strategy", "constant"),
                   urf_params=d.get("urf_params"), concurrency=d.get("concurrency", "serial"),
                   time_step_adjust=d.get("time_step_adjust", True),
                   zones=[CoSimZone.from_dict(z) for z in d.get("zones") or []])
        return link

    def to_json(self, path=None):
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=1, sort_keys=True)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        return text

    @classmethod
    def from_json(cls, text=None, path=None):
        if path:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        return cls.from_dict(json.loads(text))

    def summary(self):
        return {"name": self.name, "type": self.sim_type, "partner": self.partner,
                "connect": {"method": self.connect_method, "host": self.host,
                            "port": self.port, "connection_file": self.connection_file},
                "launch": {"option": self.launch_option, "executable": self.executable,
                           "command_line": self.command_line},
                "coupling_interval": self.coupling_interval,
                "interval_unit": self.interval_unit,
                "urf": {"strategy": self.urf_strategy, "params": dict(self.urf_params)},
                "concurrency": self.concurrency,
                "zones": [z.summary() for z in self.zones],
                "n_problems": len(self.validate())}


# ------------------------------------------------------------------ 工厂 / 模板
def make_link(sim_type, name=None, **kwargs):
    """按类型默认值建链（未给的连接/启动方式取该类默认）。"""
    defaults = type_defaults(sim_type)
    kwargs.setdefault("connect_method", defaults.get("connection", "host_port"))
    kwargs.setdefault("launch_option", defaults.get("launch", "executable"))
    link = CoSimulationLink(name or ("%s_link" % str(sim_type).lower()), sim_type, **kwargs)
    return link


def template(sim_type="generic"):
    """可直接编辑的配置模板（含一个区域与一组常见场）。"""
    link = make_link(sim_type)
    link.zones = [CoSimZone("zone1", boundaries=["boundary1"],
                            exported={"pressure": "pressure", "traction": "traction"},
                            imported={"displacement": "displacement"})]
    link.urf_params = {"urf": 0.5} if link.urf_strategy == "constant" else {}
    return link


# ------------------------------------------------------------------ .sim 抽取
def extract_cosimulation(sim):
    """从 .sim 对象图抽取协同仿真链接配置（**有对象才解，无对象诚实拒绝**）。

    返回 {"ok","links","n_objects","reason"}；links=[link.summary() + 原始字段]。
    """
    objs = [o for o in getattr(sim, "objects", [])
            if (o.class_name or "").startswith("star.cosimulation")]
    if not objs:
        return {"ok": False, "links": [], "n_objects": 0,
                "reason": "语料无协同仿真对象（star.cosimulation.*）——链接为运行期配置"}
    links = []
    for o in objs:
        if not (o.class_name or "").endswith("CoSimulation"):
            continue
        d = getattr(o, "dict", {}) or {}
        name = d.get("PresentationName") or getattr(o, "name", None) or ("link%d" % o.id)
        links.append({"name": name, "class": o.class_name, "id": o.id,
                      "fields": {k: v for k, v in sorted(d.items())}})
    if not links:
        return {"ok": False, "links": [], "n_objects": len(objs),
                "reason": "有 star.cosimulation.* 对象但无 CoSimulation 根对象（%d 个其他对象）"
                          % len(objs)}
    return {"ok": True, "links": links, "n_objects": len(objs), "reason": ""}


# ------------------------------------------------------------------ 宏前端
def render_macro(link, class_name="SetupCoSimulation"):
    """配置 → Java 宏文本（B 路线前端）。

    **best-effort**：类名取自官方 star.cosimulation.link.common / .common 类清单
    （work_pkgs/*.txt），方法签名未逐条核对官方 Javadoc，需在许可环境核验后再执行。
    """
    lines = [
        "// [best-effort] 由 stardecoding cosimulation.py 生成（R6 前端配置 → 宏）",
        "// 类名依据官方 star.cosimulation.link.common / star.cosimulation.common 类清单；",
        "// 方法签名未逐条核对，须在带 license 的 STAR-CCM+ 环境核验后再运行。",
        "import star.common.*;",
        "import star.cosimulation.link.common.*;",
        "import star.cosimulation.common.*;",
        "",
        "public class %s extends StarMacro {" % class_name,
        "    public void execute() {",
        "        Simulation sim = getActiveSimulation();",
        "        CoSimulationManager mgr = sim.get(CoSimulationManager.class);",
        '        CoSimulation link = mgr.createCoSimulation("%s");' % link.name,
        "        // 类型：%s" % link.sim_type,
        "        // 连接：%s host=%s port=%d file=%s"
        % (link.connect_method, link.host, link.port, link.connection_file),
        "        // 启动：%s exe=%s cmd=%s"
        % (link.launch_option, link.executable, link.command_line),
        "        // 耦合区间：%g %s；URF：%s %s；并发：%s"
        % (link.coupling_interval, link.interval_unit, link.urf_strategy,
           link.urf_params, link.concurrency),
    ]
    for z in link.zones:
        lines.append("        // 区域 %s（%s）边界=%s" % (z.name, z.zone_type, z.boundaries))
        for fn, method in sorted(z.exported.items()):
            lines.append("        //   导出场 %s → %s" % (fn, resolve_profile_method(method)))
        for fn, method in sorted(z.imported.items()):
            lines.append("        //   导入场 %s → %s" % (fn, resolve_profile_method(method)))
    lines += ['        sim.saveState(sim.getPresentationName() + "_cosim.sim");',
              "    }", "}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI
def _main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    import argparse
    ap = argparse.ArgumentParser(description="R6 协同仿真链接配置前段（模板/校验/往返/宏）")
    ap.add_argument("--template", metavar="TYPE", default=None,
                    help="输出 %s 之一的配置模板 JSON" % "/".join(sorted(COSIM_TYPES)))
    ap.add_argument("--validate", metavar="JSON", default=None, help="校验配置文件")
    ap.add_argument("--macro", metavar="JSON", default=None, help="配置 → Java 宏（best-effort）")
    ap.add_argument("--out", metavar="PATH", default=None, help="写出路径（模板/宏）")
    args = ap.parse_args(argv)
    if args.template:
        link = template(args.template)
        text = link.to_json(args.out)
        if not args.out:
            print(text)
        print("模板: %s（校验问题 %d 条）" % (link.sim_type, len(link.validate())))
        return 0
    if args.validate:
        link = CoSimulationLink.from_json(path=args.validate)
        problems = link.validate()
        print("配置: %s  类型: %s  区域: %d" % (link.name, link.sim_type, len(link.zones)))
        print("问题:", "无" if not problems else "")
        for p in problems:
            print("  -", p)
        return 1 if problems else 0
    if args.macro:
        link = CoSimulationLink.from_json(path=args.macro)
        text = render_macro(link)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            print("宏已写出:", args.out)
        else:
            print(text)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
