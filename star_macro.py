# -*- coding: utf-8 -*-
"""STAR-CCM+ 宏桥：只在工作副本上 -batch，绝不改教程原件。"""

import os
import shutil
import subprocess
import tempfile

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


def write_generate_mesh_macro(dirpath):
    path = os.path.join(dirpath, "generate_mesh_copy.java")
    with open(path, "w", encoding="ascii") as fh:
        fh.write(GENERATE_MESH_JAVA.strip() + "\n")
    return path


def run_star_macro_on_copy(exe, src_sim, macro_java, timeout=600):
    """复制 .sim 到临时目录再跑宏。返回 (workdir, returncode, log)."""
    if not exe or not os.path.isfile(exe):
        raise IOError("没有 STAR-CCM+ 可执行文件")
    src_sim = os.path.abspath(src_sim)
    work = tempfile.mkdtemp(prefix="star_macro_")
    dest = os.path.join(work, "work.sim")
    shutil.copy2(src_sim, dest)
    shutil.copy2(macro_java, os.path.join(work, os.path.basename(macro_java)))
    cmd = [exe, "-batch", os.path.basename(macro_java), dest]
    try:
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                           timeout=timeout)
        log = (p.stdout or "") + (p.stderr or "")
        return work, p.returncode, log
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
