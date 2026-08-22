// resave_sim.java
// 把二进制状态表时代保存的 .sim 用当前版本重存为 ASCII 状态表格式，
// 生成"同内容双格式"配对标例，用于推敲二进制数值流文法。
// 运行: starccmw.exe -batch resave_sim.java <input.sim>
// 输出: 同目录 resaved_<name>.sim
package macro;

import star.common.*;

public class resave_sim extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    String out = "D:/training/caedecoder/stardecoding/resaved_"
        + sim.getPresentationName().replaceAll("\\W", "") + ".sim";
    sim.println("RESAVE input=" + sim.getPresentationName());
    sim.saveState(out);
    sim.println("RESAVE_DONE " + out);
  }
}
