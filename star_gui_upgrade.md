# star_gui 升级规划（对照 STAR-CCM+ 20.02 / cabdecoding）

> 日期：2026-08-22　状态：实施中（U1 起按功能点提交 GitHub）
> 前序：M0–M6 已完成只读查看器骨架（见 `star_gui_plan.md`）。本轮是**界面同构升级**，
> 不改解析层契约（`sim_parser` / 21 文件回归仍每里程碑必跑）。

## 0. 对照基准

| 基准 | 用途 |
| --- | --- |
| STAR-CCM+ 20.02.007-R8 实机界面（截图：深色、中文菜单、左树+属性、中 3D 标签页、底输出） | 窗格布局、菜单名、工具栏分组、仿真树用户视图 |
| `starccmw.exe`（`…\star\lib\win64\clang17.0vc14.2-r8\lib\`）+ 官方 API 对象树 | 树节点与 Manager.Keys 一一对应（已用 dump 宏交叉验证） |
| `D:\training\cgns\cabdecoding`：`cab_gui.py` / `cab_panes.py` / `cab_vtk.py` | PaneFrame、分组工具栏、方向指示器左下、正交视图、法向着色、`_nyi`、离屏测试 |

## 1. 现状差距（M6 完成后仍不像 STAR-CCM+ 的点）

| # | 差距 | 截图 / cabdecoding 对照 |
| --- | --- | --- |
| 1 | 左树是对象图森林（Simulation→ManagerManager→…），不是用户树 | STAR-CCM+：**几何 / 区域 / 连续体 / 求解器 / 场景 / 绘图 / 工具** |
| 2 | 工具栏仅 Open/Close/Fingerprint | 双行：文件 · 求解控制 · 网格/场景 · **六向+等轴测** · 线框/实体/透明 |
| 3 | 菜单英文，缺 求解 / 连接 / 窗口 | 文件 编辑 网格 求解 工具 连接 窗口 帮助 |
| 4 | `orientation_marker_widget` 已写但**未挂到视口**；相机预设缺失 | 截图左下角 XYZ 三联；cab_vtk 视口 `(0,0,s,s)` |
| 5 | Fit/线框/边线绑死初始 `self.viewport`，场景标签页切换后失效 | 应对 **当前 Graphics 标签** 发命令 |
| 6 | 属性表三列（属性/类型/值） | STAR-CCM+：两列（属性 \| 值）+ 标题「对象名 - 属性」 |
| 7 | i18n 表几乎未接到界面 | 默认中文 + 英文术语 |
| 8 | 面网格无点法向、无渐变背景 | cab_vtk `_tris_to_polydata` + `vtkPolyDataNormals` |

明确仍不做（NYI，菜单保留）：属性回写 `.sim`、网格生成交互、求解运算、CAD 建模、B-Rep 重建（T 块文法未闭环，见 `function_gap_analysis.md` §2/§5）。

## 2. 技术路线（不变 + 补齐）

- **GUI**：PyQt5；**3D**：VTK `QVTKRenderWindowInteractor`；**数据**：只读 `SimFile`。
- **仿真树数据源**：不再展开 Parent/Keys 全图，而按官方管理器 `Keys` 建用户树
  （`RegionManager` / `ContinuumManager` / `SolverManager` / `SceneManager` /
  `PlotManager` / `MonitorManager` / `SimulationPartManager` / `FieldFunctionManager`）。
  与 `semantic_report()` 同源，可单测。原 `tree_roots()` 保留作调试/全图浏览。
- **3D**：移植 cab_vtk 的方向指示器（左下）、正交平面相机、`vtkPolyDataNormals`、
  边线 `vtkFeatureEdges` 回退；场景相机仍走 `CurrentView`。
- **命令**：菜单/工具栏集中注册；未映射项 `_nyi` 写输出窗。
- **视口**：`current_viewport()` = 当前 Graphics 标签中的 `Star3DViewport`。

## 3. 目标布局（对齐截图）

```
+- 菜单: 文件  编辑  网格  求解  工具  连接  窗口  帮助 -------------------+
+- 工具栏1: 打开/关闭 | 运行·暂停·步进·停止(NYI) | 网格/场景 ---------------+
+- 工具栏2: Fit Reset | +X -X +Y -Y +Z -Z Iso | 实体 线框 边线 透明 ---------+
+----------------------+--------------------------------------------------+
| 模型 / 场景/绘图      |  Graphics 标签页（每 Scene 一页 + Info）            |
|  Geometry            |  +--------------------------------------------+ |
|  Continua            |  | 3D 视口（网格/部件 + 左下方向指示器）        | |
|  Regions             |  | 右键: Fit / 线框 / 实体 / 按 Part 着色     | |
|  Solvers             |  +--------------------------------------------+ |
|  Scenes / Plots      |                                                  |
|  Tools               |                                                  |
| ---- 属性 ------------|                                                  |
|  「Mesh Scene 1 - 属性」                                                |
+----------------------+--------------------------------------------------+
| 输出（文件名标签） / 进度                                                 |
+-------------------------------------------------------------------------+
| 状态栏: (x,y,z) · 顶点/面数 · 单位 · 选择                                 |
+-------------------------------------------------------------------------+
```

## 4. 里程碑

| 点 | 内容 | 产出 | 提交 |
| --- | --- | --- | --- |
| U0 | 本规划 | `star_gui_upgrade.md` | 本文 |
| U1 | STAR-CCM+ 分组仿真树 | `StarSceneModel.sim_tree()` + 树标题 | 本轮 |
| U2 | 双行工具栏 + 中文菜单 + 窗口菜单 + 当前视口 | `star_gui.py` / `star_gui_i18n.py` | 本轮 |
| U3 | 专业 3D：方向指示器、六向/等轴测、法向、拾取同步 | `star_gui_vtk.py` / `Star3DViewport` | 本轮 |
| U4 | 两列属性检查器 + 输出窗/状态栏打磨 | `star_gui_panes.py` | 本轮 |
| U5 | 回归测试 + README | `tests/test_gui_u*.py` | 本轮 |

每个功能点完成后 **git commit 并 push origin**。解析层 `self_test` 不回归。

## 5. U1 仿真树规格

根 = `star.common.Simulation`（PresentationName / name）。子文件夹（空则仍显示）：

| 文件夹 | 数据源 |
| --- | --- |
| Geometry | `SimulationPartManager.Keys`（CadPart / SimpleBlockPart / MeshPart）；Part 下挂 `PartSurfaces` |
| Continua | `ContinuumManager.Keys`；连续体下挂 `ModelManager.Keys`（友好类名） |
| Regions | `RegionManager.Keys`；区域下挂 Boundaries（`BoundaryManager.Keys`） |
| Solvers | `SolverManager.Keys` |
| Reports | `ReportManager.Keys` |
| Plots | `PlotManager.Keys` |
| Monitors | `MonitorManager.Keys` |
| Scenes | `SceneManager.Keys`；场景下挂 `DisplayerManager.Keys` |
| Tools | Field Functions / Coordinate Systems / Units（管理器节点，不展开 112 个单位） |

无 `PresentationName` 的求解器/模型：剥 `Solver`/`Model` 后缀后按驼峰拆词
（`CoupledImplicitSolver` → `Coupled Implicit`）。

## 6. 验收

- adjointWing：树顶层含 Geometry / Continua / Regions / Solvers / Scenes / Tools；
  Regions→Fluid Domain→Inlet 等边界；Scenes→Mesh Scene 1→Mesh 1。
- 当前场景标签页 Fit / 线框 / 六向视图生效；左下角方向指示器可见（有头模式）。
- 属性标题随选中对象变化；两列。
- `python tests/run_all.py` GUI 测试全绿；`self_test.py` 解析层不回归。
