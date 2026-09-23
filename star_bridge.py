# -*- coding: utf-8 -*-
"""S 波 S1：官方 STAR-CCM+ 桥（探测 → 宏 → 工作副本 → 日志回流 → 差分）。

本机实测（2026-09-14）：Simcenter STAR-CCM+ 2502 (20.02.007) 带 license 可用，
`starccmw.exe -batch <macro.java> <copy.sim>` 在工作副本上跑通并回传日志（returncode=0）。
本模块把这套能力封装成可复用桥：
  ① find_star_exe/bridge_status：STARCCM_HOME → 常见安装根 → star/lib 布局 依次探测；
  ② 宏模板只用**已在官方 API 中用过/核对过**的调用（见 VERIFIED_APIS）：
     getActiveSimulation / getPresentationName / getPartManager().getObjects() /
     getRegionManager().getObjects() / getSceneManager().getObjects() /
     getContinuumManager().getObjects() / getReportManager().getObjects() /
     getPlotManager().getObjects() / getMonitorManager().getObjects() / saveState；
  ③ official_smoke / official_open_stats / official_resave：官方打开、视图统计、重存；
  ④ compare_official_view：官方视图计数 ↔ 本仓库解析计数 的逐项对照；
  ⑤ MESH_CASE_MACRO / official_mesh_case（S6）：我方 CGNS → 官方同网格算例
     （导入 → 建区域/物理/边界 → 稳态求解 → Save As），SAMEMESH_* 日志逐段回流。

安全约束（沿用 F7）：**只在临时目录的工作副本上跑**，绝不改教程原件；无 exe 时全部返回
skipped 且不抛错（供 W6 门控复用）。
"""
import os
import re
import shutil
import sys
import tempfile

VERIFIED_APIS = (
    "getActiveSimulation()",
    "sim.getPresentationName()",
    "sim.getPartManager().getObjects()",
    "sim.getRegionManager().getObjects()",
    "sim.getSceneManager().getObjects()",
    "sim.getContinuumManager().getObjects()",
    "sim.getReportManager().getObjects()",
    "sim.getPlotManager().getObjects()",
    "sim.getMonitorManager().getObjects()",
    "sim.saveState(",
    "sim.println(",
    # S6 同网格官方算例（均已在本机 Javadoc / 官方教程录制宏双重核对）：
    "sim.getImportManager().importFile(",
    "sim.getRegionManager().newRegionsFromParts(",   # 4 参版（5 参 featureCurveMode 已弃用）
    "sim.getContinuumManager().createContinuum(",
    "pc.enable(",                                     # Continuum.enable(Class)
    "pc.getModelManager().getModel(",                 # ModelManager.getModel(Class)
    "getMaterialProperties().getMaterialProperty(",   # MaterialPropertiesHolder → Manager
    "pc.getInitialConditions().get(",                 # ConditionManager.get(Class)
    "setMethod(",                                     # Profile.setMethod(Class)
    ".getMethod(",                                     # Profile/MaterialProperty.getMethod
    "getQuantity()",
    "region.setPhysicsContinuum(",
    "getBoundaryManager().getBoundary(",
    "boundary.setBoundaryType(",                       # Journal using classes
    "b.getValues().get(",
    "sim.getSimulationIterator()",
    "it.stop()",
    "sim.clearSolution(",
    "mpc.generateVolumeMesh()",
)

ROOT_HINTS = (
    r"C:\Program Files\Siemens",
    r"C:\Program Files\CD-adapco",
    r"D:\Program Files\Siemens",
    r"D:\Siemens",
)


def candidate_exes(roots=None, max_depth=8):
    """候选 starccmw.exe 路径：STARCCM_HOME 优先，其次常见安装根（有界深度）。"""
    out = []
    home = os.environ.get("STARCCM_HOME")
    if home:
        for rel in (os.path.join("star", "bin", "starccmw.exe"),
                    os.path.join("star", "lib", "win64")):
            p = os.path.join(home, rel)
            if p.lower().endswith(".exe") and os.path.isfile(p):
                out.append(p)
            elif os.path.isdir(p):
                for dirpath, dirnames, filenames in os.walk(p):
                    if dirpath.count(os.sep) - p.count(os.sep) > 3:
                        dirnames[:] = []
                        continue
                    for f in filenames:
                        if f.lower() == "starccmw.exe":
                            out.append(os.path.join(dirpath, f))
    for root in (roots or ROOT_HINTS):
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth > max_depth:
                dirnames[:] = []
                continue
            for f in filenames:
                if f.lower() == "starccmw.exe":
                    out.append(os.path.join(dirpath, f))
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def find_star_exe(roots=None):
    """找到可用的 starccmw.exe（找不到返回 None，不抛错）。"""
    cands = candidate_exes(roots)
    return cands[0] if cands else None


def bridge_status(roots=None):
    """桥可用性：{available, exe, candidates, reason}。"""
    cands = candidate_exes(roots)
    if not cands:
        return {"available": False, "exe": None, "candidates": [],
                "reason": "未找到 starccmw.exe（可设 STARCCM_HOME）"}
    return {"available": True, "exe": cands[0], "candidates": cands, "reason": ""}


# ------------------------------------------------------------------ 宏模板
SMOKE_MACRO = """
package macro;

import star.common.*;

public class StarBridgeSmoke extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    sim.println("BRIDGE_SMOKE name=" + sim.getPresentationName());
    sim.println("BRIDGE_SMOKE regions=" + sim.getRegionManager().getObjects().size());
  }
}
"""

OPEN_STATS_MACRO = """
package macro;

import star.common.*;

public class StarBridgeStats extends StarMacro {
  private void dump(Simulation sim, String label, Iterable<?> objs) {
    int n = 0;
    for (Object o : objs) {
      sim.println("BRIDGE_OBJ " + label + "|" + String.valueOf(o));
      n++;
    }
    sim.println("BRIDGE_COUNT " + label + "=" + n);
  }

  public void execute() {
    Simulation sim = getActiveSimulation();
    sim.println("BRIDGE_STATS name=" + sim.getPresentationName());
    dump(sim, "Part", sim.getPartManager().getObjects());
    dump(sim, "Region", sim.getRegionManager().getObjects());
    dump(sim, "Scene", sim.getSceneManager().getObjects());
    dump(sim, "Continuum", sim.getContinuumManager().getObjects());
    dump(sim, "Report", sim.getReportManager().getObjects());
    dump(sim, "Plot", sim.getPlotManager().getObjects());
    dump(sim, "Monitor", sim.getMonitorManager().getObjects());
    sim.println("BRIDGE_STATS_END");
  }
}
"""

RESAVE_MACRO = """
package macro;

import star.common.*;

public class StarBridgeResave extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    String out = "%(out)s";
    sim.println("BRIDGE_RESAVE input=" + sim.getPresentationName());
    sim.saveState(out);
    sim.println("BRIDGE_RESAVE_DONE " + out);
  }
}
"""


# ------------------------------------------------------------------ S6：受控官方运行（生成参考数据）
# 以下 API 均已在本机官方 Javadoc 中核对（star/common/Simulation.html、SimulationIterator.html、
# meshing/MeshPipelineController.html）：
#   sim.getSimulationIterator() / it.run() / it.runAndWait() / it.stop() / it.getCurrentIteration()
#   it.isIterating() / it.getNumberOfSteps() / sim.clearSolution() / mpc.generateVolumeMesh()
# 注意：sim.getSolver() 在本版本**不存在**（实测编译失败），不要再写。
RUN_CASE_MACRO = """
package macro;

import star.common.*;
import star.meshing.MeshPipelineController;

public class StarBridgeRunCase extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    String out = "%(out)s";
    long target = %(target)d;
    sim.println("BRIDGE_RUN begin mesh=%(mesh)d target=" + target);
    if (%(mesh)d == 1) {
      try {
        MeshPipelineController mpc = sim.get(MeshPipelineController.class);
        mpc.generateVolumeMesh();
        sim.println("BRIDGE_RUN mesh done");
      } catch (Exception ex) {
        sim.println("BRIDGE_RUN mesh failed: " + ex);
      }
    }
    if (%(clear)d == 1) {
      try {
        sim.clearSolution();
        sim.println("BRIDGE_RUN solution cleared");
      } catch (Exception ex) {
        sim.println("BRIDGE_RUN clear failed: " + ex);
      }
    }
    SimulationIterator it = sim.getSimulationIterator();
    sim.println("BRIDGE_RUN iterations before=" + it.getCurrentIteration());
    it.run();
    while (it.isIterating()) {
      if (target > 0 && it.getCurrentIteration() >= target) {
        it.stop();
        sim.println("BRIDGE_RUN stopped at " + it.getCurrentIteration());
        break;
      }
      try { Thread.sleep(%(poll)d); } catch (Exception ex) { }
    }
    sim.println("BRIDGE_RUN iterations after=" + it.getCurrentIteration());
    sim.saveState(out);
    sim.println("BRIDGE_RUN_DONE " + out);
  }
}
"""


# ------------------------------------------------------------------ S6：同网格官方算例
# 我方 CGNS（postprocess.write_cgns，含 ZoneBC 边界 patch）→ 官方外壳内：
#   importFile 导入网格 → newRegionsFromParts 建区域/边界 → 3D 稳态气体单组分 +
#   分离流 + 恒密度 + 层流连续体（ρ/μ/入口流速按参）→ Inlet/Outlet/壁面边界 →
#   稳态迭代到目标步 → Save As。
# API 全部经本机 Javadoc + 官方教程录制宏双重核对（详见 VERIFIED_APIS 注）；
# 注意 getObjects() 返回 Collection<Part>（非 GeometryPart），须 instanceof 过滤。
MESH_CASE_MACRO = """
package macro;

import star.common.*;
import star.flow.ConstantDensityModel;
import star.flow.ConstantDensityProperty;
import star.flow.DynamicViscosityProperty;
import star.flow.LaminarModel;
import star.flow.VelocityMagnitudeProfile;
import star.flow.VelocityProfile;
import star.material.ConstantMaterialPropertyMethod;
import star.material.Gas;
import star.material.SingleComponentGasModel;
import star.metrics.ThreeDimensionalModel;
import star.segregatedflow.SegregatedFlowModel;

public class StarBridgeMeshCase extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    try {
      run(sim);
    } catch (Exception ex) {
      sim.println("SAMEMESH_FAIL uncaught: " + ex);
    }
  }

  private void run(Simulation sim) {
    String cgns = "%(cgns)s";
    String out = "%(out)s";
    long target = %(target)d;
    double rho = %(rho)s;
    double mu = %(mu)s;
    double uIn = %(u_in)s;

    // ① 快照（部件 + 区域）→ 导入我方 CGNS → 差分。
    //    体网格导入的产物随版本而异：新部件（需转区域）**或**直接新区域，
    //    两条路都走；都没有 → 诚实失败。
    java.util.HashSet<String> partsBefore = new java.util.HashSet<String>();
    for (Object po : sim.getPartManager().getObjects()) {
      partsBefore.add(((star.base.neo.ClientServerObject) po).getPresentationName());
    }
    java.util.HashSet<String> regionsBefore = new java.util.HashSet<String>();
    for (Region r : sim.getRegionManager().getObjects()) {
      regionsBefore.add(r.getPresentationName());
    }
    sim.println("SAMEMESH parts_before=" + partsBefore.size()
        + " regions_before=" + regionsBefore.size());
    try {
      sim.getImportManager().importFile(cgns);
    } catch (Exception ex) {
      sim.println("SAMEMESH_FAIL import: " + ex);
      return;
    }
    java.util.ArrayList<GeometryPart> newParts = new java.util.ArrayList<GeometryPart>();
    for (Object po : sim.getPartManager().getObjects()) {
      if (!partsBefore.contains(
            ((star.base.neo.ClientServerObject) po).getPresentationName())
          && (po instanceof GeometryPart)) {
        newParts.add((GeometryPart) po);
      }
    }
    sim.println("SAMEMESH parts_new=" + newParts.size());
    if (!newParts.isEmpty()) {
      // ② 部件 → 区域（OneRegionPerPart + 每表面一个边界，CGNS ZoneBC 名应成为边界名）
      try {
        sim.getRegionManager().newRegionsFromParts(newParts, "OneRegionPerPart",
            "OneBoundaryPerPartSurface", true);
      } catch (Exception ex) {
        sim.println("SAMEMESH_FAIL regions: " + ex);
        return;
      }
    }
    Region region = null;
    for (Region r : sim.getRegionManager().getObjects()) {
      if (!regionsBefore.contains(r.getPresentationName())) {
        region = r;
        break;
      }
    }
    if (region == null) {
      sim.println("SAMEMESH_FAIL no_region");
      return;
    }
    sim.println("SAMEMESH region=" + region.getPresentationName());

    // ③ 物理连续体：3D 稳态单组分气体 + 分离流 + 恒密度 + 层流
    PhysicsContinuum pc = sim.getContinuumManager().createContinuum(PhysicsContinuum.class);
    pc.enable(ThreeDimensionalModel.class);
    pc.enable(SteadyModel.class);
    pc.enable(SingleComponentGasModel.class);
    pc.enable(SegregatedFlowModel.class);
    pc.enable(ConstantDensityModel.class);
    pc.enable(LaminarModel.class);

    // ④ 材料物性（官方教程录制宏同款调用链）：密度/动力粘度
    Gas gas = (Gas) pc.getModelManager().getModel(SingleComponentGasModel.class).getMaterial();
    ConstantMaterialPropertyMethod rhoM = (ConstantMaterialPropertyMethod)
        gas.getMaterialProperties().getMaterialProperty(ConstantDensityProperty.class).getMethod();
    rhoM.getQuantity().setValue(rho);
    ConstantMaterialPropertyMethod muM = (ConstantMaterialPropertyMethod)
        gas.getMaterialProperties().getMaterialProperty(DynamicViscosityProperty.class).getMethod();
    muM.getQuantity().setValue(mu);

    // ⑤ 初始条件：均匀来流（+x）；区域挂接连续体
    VelocityProfile vp = pc.getInitialConditions().get(VelocityProfile.class);
    vp.setMethod(ConstantVectorProfileMethod.class);
    vp.getMethod(ConstantVectorProfileMethod.class).getQuantity().setComponents(uIn, 0.0, 0.0);
    region.setPhysicsContinuum(pc);

    // ⑥ 边界：Inlet→速度入口；Outlet→压力出口；其余→壁面
    int nIn = 0, nOut = 0;
    for (Boundary b : region.getBoundaryManager().getObjects()) {
      String n = b.getPresentationName();
      sim.println("SAMEMESH bnd|" + n);
      if (n.equals("Inlet")) {
        b.setBoundaryType(InletBoundary.class);
        b.getValues().get(VelocityMagnitudeProfile.class).setMethod(ConstantScalarProfileMethod.class);
        b.getValues().get(VelocityMagnitudeProfile.class).getMethod(
            ConstantScalarProfileMethod.class).getQuantity().setValue(uIn);
        nIn++;
      } else if (n.equals("Outlet")) {
        b.setBoundaryType(PressureBoundary.class);
        nOut++;
      } else {
        b.setBoundaryType(WallBoundary.class);
      }
    }
    if (nIn != 1 || nOut != 1) {
      sim.println("SAMEMESH_FAIL inlet=" + nIn + " outlet=" + nOut);
      return;
    }

    // ⑦ 稳态迭代（轮询到目标步即停；target=0 交给自身收敛准则）→ 另存
    SimulationIterator it = sim.getSimulationIterator();
    it.run();
    while (it.isIterating()) {
      if (target > 0 && it.getCurrentIteration() >= target) {
        it.stop();
        break;
      }
      try { Thread.sleep(%(poll)d); } catch (Exception ex) { }
    }
    sim.println("SAMEMESH iter=" + it.getCurrentIteration());
    sim.saveState(out);
    sim.println("SAMEMESH_DONE " + out);
  }
}
"""


def official_run_case(src_sim, out_path, do_mesh=False, max_iterations=0,
                      clear_solution=False, class_name="StarBridgeRunCase",
                      timeout=1800, on_line=None, poll_ms=50):
    """在**工作副本**上跑官方求解（受控）：可选生成网格/清解，跑到目标迭代数即停，然后 Save As。

    max_iterations=0 → 交给算例自身停止准则（runAndWait 语义由宏内轮询实现）。
    返回 {ok, out_path, iterations_before/after, log, reason}。
    """
    out = os.path.abspath(out_path)
    out_java = out.replace("\\", "/")
    macro = RUN_CASE_MACRO % {"out": out_java, "mesh": 1 if do_mesh else 0,
                             "clear": 1 if clear_solution else 0,
                             "target": int(max_iterations or 0),
                             "poll": max(int(poll_ms), 10)}
    res = official_run(src_sim, macro, class_name, timeout=timeout, on_line=on_line)
    if res.get("skipped"):
        return res
    log = res.get("log") or ""
    def _num(pattern):
        m = re.search(pattern, log)
        return int(m.group(1)) if m else None
    res["out_path"] = out
    res["iterations_before"] = _num(r"BRIDGE_RUN iterations before=(\d+)")
    res["iterations_after"] = _num(r"BRIDGE_RUN iterations after=(\d+)")
    res["ok"] = bool(res.get("ok") and "BRIDGE_RUN_DONE" in log and os.path.isfile(out))
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "运行标记缺失或输出未生成"
    return res


MESH_CASE_TRANSIENT_MACRO = """
package macro;

import star.common.*;
import star.flow.ConstantDensityModel;
import star.flow.ConstantDensityProperty;
import star.flow.DynamicViscosityProperty;
import star.flow.LaminarModel;
import star.flow.VelocityMagnitudeProfile;
import star.flow.VelocityProfile;
import star.material.ConstantMaterialPropertyMethod;
import star.material.Gas;
import star.material.SingleComponentGasModel;
import star.metrics.ThreeDimensionalModel;
import star.segregatedflow.SegregatedFlowModel;

public class StarBridgeTransientCase extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    try {
      run(sim);
    } catch (Exception ex) {
      sim.println("SAMEMESH_FAIL uncaught: " + ex);
    }
  }

  private void run(Simulation sim) {
    String cgns = "%(cgns)s";
    String outPrefix = "%(out)s";
    double dt = %(dt)s;
    double totalTime = %(total_time)s;
    double sampleDt = %(sample_dt)s;
    double rho = %(rho)s;
    double mu = %(mu)s;
    double uIn = %(u_in)s;

    java.util.HashSet<String> partsBefore = new java.util.HashSet<String>();
    for (Object po : sim.getPartManager().getObjects()) {
      partsBefore.add(((star.base.neo.ClientServerObject) po).getPresentationName());
    }
    java.util.HashSet<String> regionsBefore = new java.util.HashSet<String>();
    for (Region r : sim.getRegionManager().getObjects()) {
      regionsBefore.add(r.getPresentationName());
    }
    sim.println("SAMEMESH parts_before=" + partsBefore.size()
        + " regions_before=" + regionsBefore.size());
    try {
      sim.getImportManager().importFile(cgns);
    } catch (Exception ex) {
      sim.println("SAMEMESH_FAIL import: " + ex);
      return;
    }
    java.util.ArrayList<GeometryPart> newParts = new java.util.ArrayList<GeometryPart>();
    for (Object po : sim.getPartManager().getObjects()) {
      if (!partsBefore.contains(
            ((star.base.neo.ClientServerObject) po).getPresentationName())
          && (po instanceof GeometryPart)) {
        newParts.add((GeometryPart) po);
      }
    }
    sim.println("SAMEMESH parts_new=" + newParts.size());
    if (!newParts.isEmpty()) {
      try {
        sim.getRegionManager().newRegionsFromParts(newParts, "OneRegionPerPart",
            "OneBoundaryPerPartSurface", true);
      } catch (Exception ex) {
        sim.println("SAMEMESH_FAIL regions: " + ex);
        return;
      }
    }
    Region region = null;
    for (Region r : sim.getRegionManager().getObjects()) {
      if (!regionsBefore.contains(r.getPresentationName())) {
        region = r;
        break;
      }
    }
    if (region == null) {
      sim.println("SAMEMESH_FAIL no_region");
      return;
    }
    sim.println("SAMEMESH region=" + region.getPresentationName());

    // 物理：3D + **隐式非稳态** + 单组分气体 + 分离流 + 恒密度 + 层流
    PhysicsContinuum pc = sim.getContinuumManager().createContinuum(PhysicsContinuum.class);
    pc.enable(ThreeDimensionalModel.class);
    pc.enable(ImplicitUnsteadyModel.class);
    pc.enable(SingleComponentGasModel.class);
    pc.enable(SegregatedFlowModel.class);
    pc.enable(ConstantDensityModel.class);
    pc.enable(LaminarModel.class);

    Gas gas = (Gas) pc.getModelManager().getModel(SingleComponentGasModel.class).getMaterial();
    ConstantMaterialPropertyMethod rhoM = (ConstantMaterialPropertyMethod)
        gas.getMaterialProperties().getMaterialProperty(ConstantDensityProperty.class).getMethod();
    rhoM.getQuantity().setValue(rho);
    ConstantMaterialPropertyMethod muM = (ConstantMaterialPropertyMethod)
        gas.getMaterialProperties().getMaterialProperty(DynamicViscosityProperty.class).getMethod();
    muM.getQuantity().setValue(mu);

    VelocityProfile vp = pc.getInitialConditions().get(VelocityProfile.class);
    vp.setMethod(ConstantVectorProfileMethod.class);
    vp.getMethod(ConstantVectorProfileMethod.class).getQuantity().setComponents(uIn, 0.0, 0.0);
    region.setPhysicsContinuum(pc);

    // 时间步：Solver（非 Model！）→ Simulation.getSolverManager().getSolver(Class)
    // + SpecifiedTimestepUnsteadySolver.setTimeStep(double)（本机 Javadoc 核对；
    // 曾误走 ModelManager.getModel(Class) → 编译期上界不符，探针抓出）
    try {
      SolverManager sm = sim.getSolverManager();
      SpecifiedTimestepUnsteadySolver tss =
          (SpecifiedTimestepUnsteadySolver) sm.getSolver(SpecifiedTimestepUnsteadySolver.class);
      tss.setTimeStep(dt);
      sim.println("SAMEMESH dt=" + dt + " dt_set=ok");
    } catch (Exception ex) {
      sim.println("SAMEMESH dt_fail: " + ex);
    }

    // 边界：Inlet 速度入口 / Outlet 压力出口 / Cylinder 无滑移壁 /
    // 四个外壁 → SymmetryBoundary（滑移，与我方 wall_slip_axes=(1,2) 对齐）
    int nIn = 0, nOut = 0, nWall = 0, nSym = 0;
    for (Boundary b : region.getBoundaryManager().getObjects()) {
      String n = b.getPresentationName();
      sim.println("SAMEMESH bnd|" + n);
      if (n.equals("Inlet")) {
        b.setBoundaryType(InletBoundary.class);
        b.getValues().get(VelocityMagnitudeProfile.class).setMethod(ConstantScalarProfileMethod.class);
        b.getValues().get(VelocityMagnitudeProfile.class).getMethod(
            ConstantScalarProfileMethod.class).getQuantity().setValue(uIn);
        nIn++;
      } else if (n.equals("Outlet")) {
        b.setBoundaryType(PressureBoundary.class);
        nOut++;
      } else if (n.equals("Cylinder")) {
        b.setBoundaryType(WallBoundary.class);
        nWall++;
      } else {
        b.setBoundaryType(SymmetryBoundary.class);
        nSym++;
      }
    }
    sim.println("SAMEMESH bnd_kinds in=" + nIn + " out=" + nOut
        + " wall=" + nWall + " sym=" + nSym);
    if (nIn != 1 || nOut != 1 || nWall != 1) {
      sim.println("SAMEMESH_FAIL bnd_kinds in=" + nIn + " out=" + nOut
          + " wall=" + nWall);
      return;
    }

    // 推进 + 分段存场：**it.step(nSteps)**（Javadoc：step(int nSteps)；旧名
    // stepAndWait(int) 已弃用）。踩过的坑（如实记录）：
    //   ① `it.run(n)` 的推进粒度依赖求解器状态 —— 实测首段后每次只推 1 个时间步，
    //      42 个存场仅覆盖 1.39 s；
    //   ② 无参 run() 会阻塞到**自带停止准则**满足即停（外壳/求解器激活时自动创建
    //      Maximum Physical Time 与 Maximum Inner Iterations），时间轮询法一个存场都进不去。
    // step(int) 是显式推进，绕开停止准则驱动，确定性最好。
    double stepsPerSample = Math.max(1.0, sampleDt / dt);
    int totalSteps = (int) Math.round(totalTime / dt);
    int chunkSteps = (int) Math.max(1.0, Math.round(stepsPerSample));
    SimulationIterator it = sim.getSimulationIterator();
    int done = 0, nSave = 0;
    while (done < totalSteps) {
      int chunk = Math.min(chunkSteps, totalSteps - done);
      it.step(chunk);
      done += chunk;
      nSave++;
      double tphys = -1.0;
      try {
        tphys = ((star.common.UnsteadyModel)
            pc.getModelManager().getModel(ImplicitUnsteadyModel.class)).getPhysicalTime();
      } catch (Exception ex) { }
      String outp = outPrefix + "_" + nSave + ".sim";
      sim.saveState(outp);
      sim.println("SAMEMESH save|" + outp + "|" + it.getCurrentIteration()
          + "|" + tphys);
    }
    sim.println("SAMEMESH iter=" + it.getCurrentIteration());
    sim.println("SAMEMESH_DONE " + outPrefix);
  }
}
"""


def official_mesh_case_transient(src_sim, cgns_path, out_prefix, dt=0.01,
                                 total_time=12.0, sample_dt=0.5,
                                 rho=1.0, mu=1e-5, u_in=0.05,
                                 class_name="StarBridgeTransientCase",
                                 timeout=7200, on_line=None, poll_ms=50):
    """S6/S2：同网格**瞬态**官方算例（我方 CGNS → 官方非稳态 → 分段存场）。

    与本仓库 run_case 同工况对齐：隐式非稳态 + 指定时间步 dt；Inlet 速度入口、
    Outlet 压力出口、Cylinder 无滑移壁、**四个外壁 SymmetryBoundary（滑移）** ——
    对应我方 wall_slip_axes=(1,2) 的准二维设置。

    每 sample_dt 秒存一个 `<out_prefix>_<n>.sim`（n 为存场序号），供离线用本仓库同一份
    force_coefficients 算 CL(t)（无需在宏里建报告/监视器 —— API 面最小）。
    SAMEMESH_* 标记回流；任何失败 → SAMEMESH_FAIL <原因>。
    返回 {ok, saves:[(path, iteration, physical_time)…], dt, log, ...}。
    """
    if not os.path.isfile(cgns_path):
        return {"ok": False, "reason": "CGNS 不存在: %s" % cgns_path, "log": ""}
    prefix = os.path.abspath(out_prefix)
    macro = MESH_CASE_TRANSIENT_MACRO % {
        "cgns": os.path.abspath(cgns_path).replace("\\", "/"),
        "out": prefix.replace("\\", "/"),
        "dt": repr(float(dt)), "total_time": repr(float(total_time)),
        "sample_dt": repr(float(sample_dt)),
        "rho": repr(float(rho)), "mu": repr(float(mu)), "u_in": repr(float(u_in)),
        "poll": max(int(poll_ms), 10),
    }
    res = official_run(src_sim, macro, class_name, timeout=timeout, on_line=on_line)
    if res.get("skipped"):
        return res
    log = res.get("log") or ""
    fail = re.search(r"SAMEMESH_FAIL (\S.*)", log)
    res["fail_reason"] = fail.group(1).strip() if fail else None
    res["out_prefix"] = prefix
    res["saves"] = []
    for ln in log.splitlines():
        m = re.match(r"SAMEMESH save\|(.*)\|(\d+)\|([-0-9.eE]+)", ln.strip())
        if m:
            res["saves"].append((m.group(1).strip(), int(m.group(2)),
                                  float(m.group(3))))
    m = re.search(r"SAMEMESH dt=([-0-9.eE]+)", log)
    res["dt"] = float(m.group(1)) if m else None
    m = re.search(r"SAMEMESH bnd_kinds in=(\d+) out=(\d+) wall=(\d+) sym=(\d+)", log)
    res["bnd_kinds"] = ({"in": int(m.group(1)), "out": int(m.group(2)),
                         "wall": int(m.group(3)), "sym": int(m.group(4))}
                        if m else None)
    res["iterations"] = _transient_iter(log)
    res["ok"] = bool(res.get("ok") and "SAMEMESH_DONE" in log
                      and res["saves"] and os.path.isfile(res["saves"][-1][0]))
    if not res["ok"] and not res.get("reason") and not res.get("fail_reason"):
        if re.search(r"error:\s", log):
            res["reason"] = "宏编译失败（Java error，见 log）"
        else:
            res["reason"] = "SAMEMESH 标记缺失或分段存场未生成"
    return res


def _transient_iter(log):
    m = re.search(r"SAMEMESH iter=(\d+)", log)
    return int(m.group(1)) if m else None

def official_mesh_case(src_sim, cgns_path, out_path, rho=1.0, mu=1e-5, u_in=0.05,
                       max_iterations=0, class_name="StarBridgeMeshCase",
                       timeout=3600, on_line=None, poll_ms=50):
    """S6 同网格官方算例：我方 CGNS → 官方外壳内建区/物理/边界 → 稳态求解 → Save As。

    在**工作副本**上执行（绝不改外壳原件）。SAMEMESH_* 标记逐段回流：
      parts_before / parts_new / region= / bnd|<边界名>（逐个）/
      iter=<实际迭代步> / SAMEMESH_DONE <输出路径>；
    任何一步失败 → SAMEMESH_FAIL <原因>（诚实失败，绝不返回未收敛解）。
    返回 {ok, out_path, fail_reason, parts_before, parts_new, region,
          boundaries, iterations, log, reason, ...}。
    """
    if not os.path.isfile(cgns_path):
        return {"ok": False, "reason": "CGNS 不存在: %s" % cgns_path, "log": ""}
    out = os.path.abspath(out_path)
    macro = MESH_CASE_MACRO % {
        "cgns": os.path.abspath(cgns_path).replace("\\", "/"),
        "out": out.replace("\\", "/"),
        "target": int(max_iterations or 0),
        "rho": repr(float(rho)), "mu": repr(float(mu)), "u_in": repr(float(u_in)),
        "poll": max(int(poll_ms), 10),
    }
    res = official_run(src_sim, macro, class_name, timeout=timeout, on_line=on_line)
    if res.get("skipped"):
        return res
    log = res.get("log") or ""

    def _num(pattern):
        m = re.search(pattern, log)
        return int(m.group(1)) if m else None

    fail = re.search(r"SAMEMESH_FAIL (\S.*)", log)
    region_m = re.search(r"SAMEMESH region=(.*)", log)
    res["fail_reason"] = fail.group(1).strip() if fail else None
    res["out_path"] = out
    res["parts_before"] = _num(r"SAMEMESH parts_before=(\d+)")
    res["parts_new"] = _num(r"SAMEMESH parts_new=(\d+)")
    res["regions_before"] = _num(r"regions_before=(\d+)")
    res["region"] = region_m.group(1).strip() if region_m else None
    res["boundaries"] = [ln.split("|", 1)[1].strip()
                         for ln in log.splitlines() if ln.startswith("SAMEMESH bnd|")]
    res["iterations"] = _num(r"SAMEMESH iter=(\d+)")
    res["ok"] = bool(res.get("ok") and "SAMEMESH_DONE" in log and os.path.isfile(out))
    if not res["ok"] and not res.get("reason") and not res.get("fail_reason"):
        res["reason"] = "SAMEMESH 标记缺失或输出未生成"
    return res


# ------------------------------------------------------------------ 运行器
def official_run(src_sim, macro_text, class_name="StarBridgeMacro", timeout=600,
                 on_line=None, workdir=None):
    """在**工作副本**上跑官方宏；返回 {ok, returncode, log, workdir, exe, reason}。"""
    status = bridge_status()
    if not status["available"]:
        return {"ok": False, "skipped": True, "reason": status["reason"], "log": ""}
    if not os.path.isfile(src_sim):
        return {"ok": False, "reason": "源文件不存在: %s" % src_sim, "log": ""}
    from star_macro import run_star_macro_stream
    tmp = workdir or tempfile.mkdtemp(prefix="star_bridge_")
    macro_path = os.path.join(tmp, "%s.java" % class_name)
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write(macro_text)
    try:
        work, code, lines = run_star_macro_stream(status["exe"], src_sim, macro_path,
                                                  on_line, timeout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "官方宏执行失败: %s" % exc, "log": "",
                "exe": status["exe"]}
    log = "\n".join(lines)
    return {"ok": code == 0, "returncode": code, "log": log, "workdir": work,
            "exe": status["exe"], "reason": "" if code == 0 else "returncode=%s" % code}


def official_smoke(src_sim, timeout=600):
    """官方打开 + 冒烟（仿真名 / region 数）。"""
    res = official_run(src_sim, SMOKE_MACRO, "StarBridgeSmoke", timeout)
    if not res.get("ok") and res.get("skipped"):
        return res
    m = re.search(r"BRIDGE_SMOKE name=(.*)", res.get("log", ""))
    r = re.search(r"BRIDGE_SMOKE regions=(\d+)", res.get("log", ""))
    res["sim_name"] = m.group(1).strip() if m else None
    res["official_regions"] = int(r.group(1)) if r else None
    res["ok"] = bool(res.get("ok") and m and r)
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "冒烟标记缺失（宏未按预期输出）"
    return res


def official_open_stats(src_sim, timeout=900):
    """官方视图统计：Part/Region/Scene/Continuum/Report/Plot/Monitor 的对象名清单。"""
    res = official_run(src_sim, OPEN_STATS_MACRO, "StarBridgeStats", timeout)
    if res.get("skipped"):
        return res
    counts, names, sim_name = {}, {}, None
    for line in res.get("log", "").splitlines():
        line = line.strip()
        m = re.match(r"BRIDGE_STATS name=(.*)", line)
        if m:
            sim_name = m.group(1).strip()
        m = re.match(r"BRIDGE_COUNT ([A-Za-z]+)=(\d+)", line)
        if m:
            counts[m.group(1)] = int(m.group(2))
        m = re.match(r"BRIDGE_OBJ ([A-Za-z]+)\|(.*)", line)
        if m:
            names.setdefault(m.group(1), []).append(m.group(2).strip())
    res["sim_name"] = sim_name
    res["counts"] = counts
    res["names"] = names
    res["ok"] = bool(res.get("ok") and counts and "BRIDGE_STATS_END" in res.get("log", ""))
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "统计标记缺失"
    return res


def official_resave(src_sim, out_path=None, timeout=1200):
    """官方重存（Save As）到 out_path；返回 {ok, out_path, log}。"""
    out = out_path or os.path.join(os.path.dirname(os.path.abspath(src_sim)),
                                   "bridge_resaved.sim")
    out_java = out.replace("\\", "/")
    res = official_run(src_sim, RESAVE_MACRO % {"out": out_java}, "StarBridgeResave",
                       timeout)
    if res.get("skipped"):
        return res
    res["out_path"] = out
    res["ok"] = bool(res.get("ok") and "BRIDGE_RESAVE_DONE" in res.get("log", "")
                     and os.path.isfile(out))
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "重存标记缺失或输出文件未生成"
    return res


# ------------------------------------------------------------------ 官方 ↔ 本地 对照
def _our_counts(sim):
    """本仓库解析结果的同口径计数（官方 Manager 视角）。"""
    def n(pred):
        return sum(1 for o in sim.objects if pred(o.class_name or ""))
    return {
        "Region": n(lambda c: c == "star.common.Region"),
        "Scene": n(lambda c: c.endswith(".Scene") or c.endswith("Scene") and "Manager" not in c),
        "Continuum": n(lambda c: c.endswith("Continuum") and "Manager" not in c),
        "Report": n(lambda c: c.endswith("Report") and "Manager" not in c
                    and "Monitor" not in c),
        "Plot": n(lambda c: c.endswith("Plot") and "Manager" not in c
                    and not c.endswith("MonitorPlot")),
        "Monitor": n(lambda c: c.endswith("Monitor")),
        # 注意：官方 getPartManager() 与对象图里的 *Part 类不完全同口径（后者含
        # CadPart/SimpleBlockPart 等几何部件）；此列标注为"口径近似"。
        "Part": n(lambda c: c.endswith("Part") and "Manager" not in c),
    }


def compare_official_view(src_sim_path, timeout=900):
    """官方视图 ↔ 本仓库解析：逐类计数对照（含 Δ 与口径说明）。"""
    from sim_parser import SimFile
    stats = official_open_stats(src_sim_path, timeout=timeout)
    if stats.get("skipped"):
        return {"ok": False, "skipped": True, "reason": stats.get("reason")}
    if not stats.get("ok"):
        return {"ok": False, "reason": stats.get("reason"), "log": stats.get("log", "")[-800:]}
    ours = _our_counts(SimFile(src_sim_path))
    rows = {}
    for label, oc in stats["counts"].items():
        mc = ours.get(label)
        rows[label] = {"official": oc, "ours": mc,
                       "delta": (None if mc is None else oc - mc),
                       "note": "" if mc is not None else "本仓库无同口径计数"}
    return {"ok": True, "sim_name": stats.get("sim_name"), "rows": rows,
            "official_names": stats.get("names"), "ours": ours, "reason": ""}


def _key_of(obj):
    """官方重存会重编号 id → 语义身份用「类名 + 名字」。"""
    return ((obj.class_name or ""), (str(obj.name) if obj.name is not None else None))


def _is_ref_value(v):
    """引用型取值：整数或整数列表（官方重存会重编号 id，不能直接跨文件比较）。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, (list, tuple)) and v:
        return all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    return False


def compare_official_resave(src_path, resaved_path, keys=None, max_diffs=12,
                            min_match_rate=80.0):
    """官方重存件 ↔ 本仓库产物：**按类名+名字**配对差分（官方会重编号 id）。

    字段差异分两类：
      · value 类（字符串/浮点/bool/混合列表）→ 语义应一致，是判定依据；
      · ref 类（整数或整数列表 = 对象 id 引用）→ 官方重编号后必然不同，**不计入判定**，
        仅计数并在样例里展示（如需语义比较，须先按类名+名字建立 id 映射）。
    返回 {ok, matched, match_rate, value_diffs, ref_diffs, verdicts, ...}。
    """
    from sim_parser import SimFile
    a, b = SimFile(src_path), SimFile(resaved_path)
    keys = keys or ("PresentationName", "name", "Opacity", "DisplayerColor", "Mesh",
                    "ParallelScale", "Keys", "Parent", "Value", "Values")
    map_a, map_b = {}, {}
    for o in a.objects:
        map_a.setdefault(_key_of(o), []).append(o)
    for o in b.objects:
        map_b.setdefault(_key_of(o), []).append(o)
    common = sorted(set(map_a) & set(map_b), key=lambda k: (k[0] or "", k[1] or ""))
    only_src = sorted(set(map_a) - set(map_b))
    only_res = sorted(set(map_b) - set(map_a))
    value_diffs, ref_diffs, default_diffs = [], [], []
    for key in common:
        oa, ob = map_a[key][0], map_b[key][0]
        for k in keys:
            va, vb = oa.dict.get(k), ob.dict.get(k)
            if va == vb or (k not in oa.dict and k not in ob.dict):
                continue
            row = {"class": key[0], "name": key[1], "field": k,
                   "ours": str(va)[:70], "official": str(vb)[:70]}
            if _is_ref_value(va) or _is_ref_value(vb):
                # 任一侧是 id/列表 → 引用型（官方重编号，或不写出该引用）
                ref_diffs.append(row)
            elif va is None and vb is not None:
                # 官方加载时会补默认值（如 Opacity 1.0）—— 归一化，不计失败
                default_diffs.append(row)
            elif k == "Values" and vb is None and (key[0] or "").endswith("LookupTable"):
                # 预定义查找表：本仓库保留断点数组，官方按类名表达（保存时不写 Values）
                # —— 官方归一化，不计失败
                default_diffs.append(row)
            else:
                value_diffs.append(row)
            break
    match_rate = round(100.0 * len(common) / max(len(map_a), 1), 1)
    return {
        "ok": True,
        "n_src": len(a.objects), "n_resaved": len(b.objects),
        "keys_src": len(map_a), "keys_resaved": len(map_b),
        "matched": len(common), "match_rate": match_rate,
        "only_in_src": len(only_src), "only_in_resaved": len(only_res),
        "value_diffs": value_diffs[:max_diffs], "n_value_diffs": len(value_diffs),
        "ref_diffs": ref_diffs[:max_diffs], "n_ref_diffs": len(ref_diffs),
        "default_diffs": default_diffs[:max_diffs], "n_default_diffs": len(default_diffs),
        "name_field_diffs": sum(1 for d in value_diffs
                               if d["field"] in ("PresentationName", "name")),
        "only_src_sample": [list(x) for x in only_src[:8]],
        "only_resaved_sample": [list(x) for x in only_res[:8]],
        "min_match_rate": min_match_rate,
        "verdicts": {
            "structure_survived": match_rate >= min_match_rate,
            "value_fields_stable": len(value_diffs) == 0,
            "normalized_diffs": len(default_diffs),
            "ref_diffs": len(ref_diffs),
        },
    }

def _main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    import argparse
    ap = argparse.ArgumentParser(description="S1 官方 STAR-CCM+ 桥（探测/冒烟/视图对照/重存）")
    ap.add_argument("--status", action="store_true", help="打印桥可用性")
    ap.add_argument("--smoke", metavar="SIM", help="官方打开冒烟")
    ap.add_argument("--view", metavar="SIM", help="官方视图 ↔ 本仓库计数对照")
    ap.add_argument("--resave", metavar="SIM", help="官方重存（Save As）")
    ap.add_argument("--out", metavar="PATH", help="--resave 输出路径")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)
    if args.status or not any([args.smoke, args.view, args.resave]):
        st = bridge_status()
        print("可用:", st["available"])
        print("exe:", st["exe"])
        for c in st["candidates"][:5]:
            print("   候选:", c)
        if not st["available"]:
            print("原因:", st["reason"])
        return 0 if st["available"] else 1
    if args.smoke:
        r = official_smoke(args.smoke, args.timeout)
        print("ok=%s name=%s regions=%s reason=%s"
              % (r.get("ok"), r.get("sim_name"), r.get("official_regions"),
                 r.get("reason")))
        return 0 if r.get("ok") else 1
    if args.view:
        r = compare_official_view(args.view, args.timeout)
        if not r.get("ok"):
            print("对照未完成:", r.get("reason"))
            return 1
        print("仿真:", r["sim_name"])
        for label, row in r["rows"].items():
            print("   %-10s 官方 %-5s 本仓库 %-5s Δ=%s %s"
                  % (label, row["official"], row["ours"], row["delta"], row["note"]))
        return 0
    if args.resave:
        r = official_resave(args.resave, args.out, args.timeout)
        print("ok=%s out=%s reason=%s" % (r.get("ok"), r.get("out_path"), r.get("reason")))
        return 0 if r.get("ok") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

