// resave_sim.java
// 把二进制状态表时代保存的 .sim 用当前版本重存为 ASCII 状态表格式，
// 生成"同内容双格式"配对标例，用于推敲二进制数值流文法。
// 运行: starccmw.exe -batch resave_sim.java <input.sim>
// 输出: **当前工作目录**下 resaved_<name>.sim（相对路径 → 由调用方的 cwd 决定，
//       便于在临时工作副本目录里做差分；S1 起不再写死绝对路径）
package macro;

import star.common.*;

public class resave_sim extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    String out = "resaved_"
        + sim.getPresentationName().replaceAll("\\W", "") + ".sim";
    sim.println("RESAVE input=" + sim.getPresentationName());
    sim.saveState(out);
    sim.println("RESAVE_DONE " + out);
  }
}
