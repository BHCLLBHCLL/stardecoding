# -*- coding: utf-8 -*-
"""STAR-CCM+ 宏桥：只在工作副本上 -batch，绝不改教程原件。

P10：B 路线扩展 —— 在 `GENERATE_MESH_JAVA`（网格）之外，补充运行闭环的
Solve / Initialize / Step 宏模板，并提供**运行日志逐行回流**：`run_star_macro_stream`
用后台线程读 stdout，逐行回调 `on_line`，GUI 把每行喂给 `MessageWindow.log` 即实时输出窗。
`run_star_macro_on_copy` 保持旧签名（返回汇集的日志字符串），向后兼容网格宏调用。
"""

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time

GENERATE_MESH_JAVA = r"""
import star.common.StarMacro;
import star.meshing.MeshPipelineController;
import star.common.Simulation;

public class generate_mesh_copy extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    try {
      MeshPipelineController mpc = sim.get(MeshPipelineController.class);
      mpc.generateVolumeMesh();
    } catch (Exception ex) {
      sim.println("generate_mesh_copy: " + ex);
    }
    sim.saveState(resolvePath("out.sim"));
  }
}
"""

SOLVE_MACRO_JAVA = r"""
import star.common.StarMacro;
import star.common.Simulation;
import star.common.Solver;

public class solve_copy extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    try {
      Solver solver = sim.get(Solver.class);
      solver.run();
      sim.println("solve_copy: done");
    } catch (Exception ex) {
      sim.println("solve_copy: " + ex);
    }
    sim.saveState(resolvePath("out.sim"));
  }
}
"""

INITIALIZE_MACRO_JAVA = r"""
import star.common.StarMacro;
import star.common.Simulation;
import star.common.Solver;

public class initialize_copy extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    try {
      Solver solver = sim.get(Solver.class);
      solver.initialize();
      sim.println("initialize_copy: done");
    } catch (Exception ex) {
      sim.println("initialize_copy: " + ex);
    }
    sim.saveState(resolvePath("out.sim"));
  }
}
"""

STEP_MACRO_JAVA = r"""
import star.common.StarMacro;
import star.common.Simulation;
import star.common.Solver;

public class step_copy extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    try {
      Solver solver = sim.get(Solver.class);
      solver.run();
      sim.println("step_copy: one_pass done");
    } catch (Exception ex) {
      sim.println("step_copy: " + ex);
    }
    sim.saveState(resolvePath("out.sim"));
  }
}
"""


def write_macro(dirpath, basename, template):
    """把 Java 宏模板写入目录，返回路径（ASCII 安全）。"""
    path = os.path.join(dirpath, basename)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(template.strip() + "\n")
    return path


def write_generate_mesh_macro(dirpath):
    return write_macro(dirpath, "generate_mesh_copy.java", GENERATE_MESH_JAVA)


def write_solve_macro(dirpath):
    return write_macro(dirpath, "solve_copy.java", SOLVE_MACRO_JAVA)


def write_initialize_macro(dirpath):
    return write_macro(dirpath, "initialize_copy.java", INITIALIZE_MACRO_JAVA)


def write_step_macro(dirpath):
    return write_macro(dirpath, "step_copy.java", STEP_MACRO_JAVA)


# ---------------------------------------------------------------------------
# A1：参数化宏渲染 + 操作→Java 映射表（录制器共用）
# ---------------------------------------------------------------------------
# 说明：模板里固定类名/固定 out.sim 的问题在此解决——`render_macro` 把
# class_name / body / save_name / imports 全部参数化注入，录制器据此产出宏。
DEFAULT_MACRO_IMPORTS = ("star.common.StarMacro", "star.common.Simulation")

# 操作注册表：名称 -> {category, imports, body}
#   category ∈ meshing / physics / solver / post
#   body 为 Java 语句模板列表，使用 %(name)s 占位符（缺失键由 _format_body 补默认）。
# 这些条目全部对应可验证的真实 STAR-CCM+ 宏 API（与既有模板同源）。
OPERATIONS = {
    # -- 网格 ---------------------------------------------------------------
    "generate_surface_mesh": {
        "category": "meshing",
        "imports": ("star.meshing.MeshPipelineController",),
        "body": (
            "MeshPipelineController mpc = sim.get(MeshPipelineController.class);",
            "mpc.generateSurfaceMesh();",
        ),
    },
    "generate_volume_mesh": {
        "category": "meshing",
        "imports": ("star.meshing.MeshPipelineController",),
        "body": (
            "MeshPipelineController mpc = sim.get(MeshPipelineController.class);",
            "mpc.generateVolumeMesh();",
        ),
    },
    "clear_mesh": {
        "category": "meshing",
        "imports": ("star.meshing.MeshPipelineController",),
        "body": (
            "MeshPipelineController mpc = sim.get(MeshPipelineController.class);",
            "mpc.clearGeneratedMeshes();",
        ),
    },
    # -- 物理模型 -----------------------------------------------------------
    "enable_model": {
        "category": "physics",
        "imports": ("star.common.PhysicsContinuum",),
        "imports_fmt": ("%(model_class)s",),
        "body": (
            'PhysicsContinuum pc = sim.getContinuumManager().getContinuum("%(continuum)s");',
            "pc.enable(%(model_class)s.class);",
        ),
    },
    # -- 求解 ---------------------------------------------------------------
    "initialize": {
        "category": "solver",
        "imports": ("star.common.Solver",),
        "body": (
            "Solver solver = sim.get(Solver.class);",
            "solver.initialize();",
        ),
    },
    "run": {
        "category": "solver",
        "imports": ("star.common.Solver",),
        "body": (
            "Solver solver = sim.get(Solver.class);",
            "solver.run();",
        ),
    },
    "step": {
        "category": "solver",
        "imports": ("star.common.Solver",),
        "body": (
            "Solver solver = sim.get(Solver.class);",
            "solver.step();",
        ),
    },
    "stop": {
        "category": "solver",
        "imports": ("star.common.Solver",),
        "body": (
            "Solver solver = sim.get(Solver.class);",
            "solver.stop();",
        ),
    },
    "clear_solution": {
        "category": "solver",
        "imports": ("star.common.Solver",),
        "body": (
            "Solver solver = sim.get(Solver.class);",
            "solver.clearSolution();",
        ),
    },
    # -- 后处理 -------------------------------------------------------------
    "export_scene_hardcopy": {
        "category": "post",
        "imports": ("star.vis.Scene",),
        "body": (
            'Scene scene = sim.getSceneManager().getScene("%(scene)s");',
            'scene.printAndWait(resolvePath("%(file)s"), 1, %(width)d, %(height)d);',
        ),
    },
    "export_plot": {
        "category": "post",
        "imports": ("star.common.Cartesian2DPlot",),
        "body": (
            'Cartesian2DPlot plot = sim.getPlotManager().getPlot("%(plot)s");',
            'plot.export(resolvePath("%(file)s"), "%(sep)s");',
        ),
    },
}

# 各操作占位符默认值（渲染时缺失键自动补默认，避免 KeyError）。
OPERATION_DEFAULTS = {
    "continuum": "Physics 1",
    "model_class": "star.flow.SegregatedFlowModel",
    "scene": "Scene 1",
    "plot": "Plot 1",
    "file": "export.png",
    "width": 1280,
    "height": 720,
    "sep": ",",
}


def operation_names(category=None):
    """操作名列表（可按 4 大类过滤：meshing/physics/solver/post）。"""
    return sorted(n for n, spec in OPERATIONS.items()
                  if category is None or spec.get("category") == category)


def _format_body(body, args):
    """把 %(name)s 模板列表按 args+默认值渲染为 Java 语句列表。"""
    merged = dict(OPERATION_DEFAULTS)
    for k, v in (args or {}).items():
        merged[k] = v
    return [line % merged for line in body]


def render_operations(ops, class_name="recorded_macro", save_name="out.sim",
                      do_save=True):
    """把操作序列渲染为可直接回放的 Java 宏源码。

    ops 为 (name, args_dict) 或 {name, args} 两类；未注册的操作渲染为
    `// [unmapped operation] ...` 注释（诚实降级，不编造 API）。
    返回 (java_source, import_list, unmapped_names) 三元组。
    """
    imports = list(DEFAULT_MACRO_IMPORTS)
    body = []
    unmapped = []
    for item in ops:
        if isinstance(item, dict):
            name = item.get("name")
            args = item.get("args") or {}
        else:
            name, args = item
            args = args or {}
        spec = OPERATIONS.get(name)
        if spec is None:
            unmapped.append(name)
            body.append("// [unmapped operation] %s %r" % (name, args))
            continue
        for imp in spec.get("imports", ()):
            if imp not in imports:
                imports.append(imp)
        for tmpl in spec.get("imports_fmt", ()):
            imp = tmpl % dict(OPERATION_DEFAULTS, **(args or {}))
            if imp not in imports:
                imports.append(imp)
        body.extend(_format_body(spec.get("body", ()), args))
    if not body:
        body.append("// (no recorded operations)")
    src = render_macro(class_name, body, save_name, imports, do_save)
    return src, imports, unmapped


def render_macro(class_name, body, save_name="out.sim", imports=None, do_save=True):
    """A1 参数化宏渲染：类名 / 语句体 / 保存文件名 / import 全量注入。

    旧模板（类名绑文件名、固定 out.sim）由本函数统一参数化。
    """
    if imports is None:
        imports = list(DEFAULT_MACRO_IMPORTS)
    if isinstance(body, str):
        body_lines = body.strip("\n").split("\n")
    else:
        body_lines = [str(x) for x in body]
    out = ["import %s;" % imp for imp in imports]
    out.append("")
    out.append("public class %s extends StarMacro {" % class_name)
    out.append("  public void execute() {")
    out.append("    Simulation sim = getActiveSimulation();")
    for line in body_lines:
        out.append(("    " + line).rstrip())
    if do_save and save_name:
        out.append('    sim.saveState(resolvePath("%s"));' % save_name)
    out.append("  }")
    out.append("}")
    return "\n".join(out) + "\n"


def write_operation_macro(dirpath, ops, basename=None, class_name=None,
                          save_name="out.sim", do_save=True):
    """渲染操作宏并写盘，返回宏文件路径。"""
    if class_name is None:
        class_name = "recorded_macro"
    if basename is None:
        basename = class_name + ".java"
    src, _imports, _unmapped = render_operations(ops, class_name, save_name, do_save)
    return write_macro(dirpath, basename, src)


def _drain(stdout, sink):
    """后台线程：把 stdout 逐行投递到 sink（以 None 结尾）。"""
    try:
        for raw in stdout:
            sink.put(raw.rstrip("\r\n"))
    finally:
        sink.put(None)


def run_star_macro_stream(exe, src_sim, macro_java, on_line=None, timeout=600):
    """复制 .sim 到临时目录再跑宏，**逐行回流日志**。

    返回 (workdir, returncode, lines)；`on_line(line)` 每行回调一次（GUI 输出窗用）。
    超时会 kill 子进程并抛 TimeoutError；调用失败清理临时目录。
    """
    if not exe or not os.path.isfile(exe):
        raise IOError("没有 STAR-CCM+ 可执行文件")
    src_sim = os.path.abspath(src_sim)
    work = tempfile.mkdtemp(prefix="star_macro_")
    dest = os.path.join(work, "work.sim")
    shutil.copy2(src_sim, dest)
    macro_name = os.path.basename(macro_java)
    shutil.copy2(macro_java, os.path.join(work, macro_name))
    cmd = [exe, "-batch", macro_name, dest]
    try:
        proc = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    sink = queue.Queue()
    reader = threading.Thread(target=_drain, args=(proc.stdout, sink), daemon=True)
    reader.start()
    lines = []
    deadline = time.time() + timeout
    try:
        while True:
            try:
                line = sink.get(timeout=0.05)
            except queue.Empty:
                if time.time() > deadline and proc.poll() is None:
                    proc.kill()
                    reader.join(timeout=2)
                    raise TimeoutError("STAR macro 超时 %ss" % timeout)
                continue
            if line is None:
                break
            lines.append(line)
            if on_line is not None:
                on_line(line)
        code = proc.wait()
        reader.join(timeout=5)
        return work, code, lines
    except Exception:
        if proc.poll() is None:
            proc.kill()
            reader.join(timeout=2)
        shutil.rmtree(work, ignore_errors=True)
        raise


def run_star_macro_on_copy(exe, src_sim, macro_java, timeout=600):
    """旧接口：日志以字符串汇总返回（对网格宏兼容）。"""
    work, code, lines = run_star_macro_stream(exe, src_sim, macro_java, None, timeout)
    return work, code, "\n".join(lines)
