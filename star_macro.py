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
