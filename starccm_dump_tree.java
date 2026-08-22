// starccm_dump_tree.java
// 用 STAR-CCM+ 官方 Java API 交叉验证 .sim 解析结果（需要 license）。
// 运行: starccmw.exe -batch starccm_dump_tree.java adjointWing_start.sim
package macro;

import star.common.*;

public class starccm_dump_tree extends StarMacro {

  private void dump(Simulation sim, String label, Iterable<?> objs) {
    for (Object o : objs) {
      // toString() 通常返回 "PresentationName"（如 "Part 1/adjointWing"）
      sim.println(label + ": " + String.valueOf(o));
    }
  }

  public void execute() {
    Simulation sim = getActiveSimulation();
    sim.println("=== OFFICIAL VIEW: " + sim.getPresentationName() + " ===");
    dump(sim, "Part", sim.getPartManager().getObjects());
    dump(sim, "Region", sim.getRegionManager().getObjects());
    dump(sim, "Scene", sim.getSceneManager().getObjects());
    dump(sim, "Continuum", sim.getContinuumManager().getObjects());
    dump(sim, "Report", sim.getReportManager().getObjects());
    dump(sim, "Plot", sim.getPlotManager().getObjects());
    dump(sim, "Monitor", sim.getMonitorManager().getObjects());
    sim.println("=== END ===");
  }
}
