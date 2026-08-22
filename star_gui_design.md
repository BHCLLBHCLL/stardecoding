# star_gui 架构设计（STAR-CCM+ .sim 查看器）

> 日期：2026-08　状态：设计稿（待评审后按 star_gui_plan.md 实施）
> 数据层：本项目 sim_parser.py（21/21 语料解析）+ semantic_dict.py + extract_mesh() +
> semantic_report() + layers 全量建树。
> 参考路线（用户指定）：
> 1. D:/training/cgns/cabdecoding —— cab_gui.py / cab_panes.py / cab_vtk.py /
>    CAB_GUI_DESIGN.md / DEV_PLAN.md（PyQt5 + VTK + numpy，PaneFrame/AppIcons/_nyi/
>    离屏测试范式，最完整参考）
> 2. D:/training/caedecoder/cstdecoding —— cst_gui.py / cst_panes.py
>    （CST 风格 Ribbon + CST3DCanvas/CST3DViewport，其 docstring 明确"技术路线匹配
>    cabdecoding"）
> 3. STAR-CCM+ 2025 实机 UI + 用户手册（doc/en/online/STARCCMP：Simulation tree、
>    Properties window、Graphics window、Output window、Plot window）

---

## 1. 目标与范围

**目标**：star_gui.py —— 只读的 .sim 项目查看器：打开文件 → 仿真树浏览 → 属性查看 →
3D 场景/网格显示 → 场景/视图切换 → 消息与状态。

**范围内（v1）**：

| 能力 | 数据来源（已有解析能力） |
| --- | --- |
| 打开 .sim（含 ZIP 容器变体） | SimFile（21 文件全通过） |
| 仿真树：Simulation/Continua/Models/Regions/Boundaries/Parts/Scenes/Plots/Monitors/Reports/Units | build_tree + layer_census + semantic_report |
| 属性面板：选中对象的所有序列化属性 + 别名解析 | SimObject.dict + resolve_class |
| 3D 网格显示（分 Part 着色、线框/实体、Fit/Reset） | extract_mesh()（18/19 自洽）+ Part.TriangleCount |
| 场景标签页 + 显示器（PartDisplayer 等）+ 视图相机 | semantic_report（Scene→Displayer→View） |
| 状态栏（坐标拾取、对象数、单位）/ 消息窗口 / 进度条 | SimFile + QVTK 拾取 |
| 导出 STL / 摘要 / 层级报告 | export_stl / summary / layer_census |

**明确不做（NYI，菜单保留入口，触发写消息窗口）**：属性编辑与回写 .sim、求解/后处理
运算、CAD 建模、网格生成交互——与 cabdecoding 的 _nyi 范式一致。

---

## 2. 技术路线

- **GUI**：PyQt5 ≥5.15（与 cabdecoding/cstdecoding 一致；本机已装）
- **3D**：VTK ≥9.3（QVTKRenderWindowInteractor 嵌入；本机已装）
- **数值**：numpy（已装）；可选 trimesh/meshio 用于后续导出扩展
- 依赖文件：requirements-gui.txt（对齐 cabdecoding 同名文件）
- 为什么不用 PySide6/pyqtgraph：与参考仓库保持一致便于互相借鉴（AppIcons/PaneFrame/
  QSS 命名、离屏测试基建可直接移植），且 PyQt5+VTK 已在参考项目验证。

---

## 3. 主窗口布局（对齐 STAR-CCM+ 2025）

    +- 标题栏 / 菜单栏: File Edit Mesh Scene Plot Tools Help ---------------+
    +- 工具栏: Open Save Fit/Reset 线框·实体 显隐 视图按钮 -------------------+
    +----------------------+-----------------------------------------------+
    | Simulation Tree      |  Graphics Window（场景/绘图 标签页）              |
    | + Simulation        | +---------------------------------------------+ |
    | | + Continua        | |  3D 视口（QVTKRenderWindowInteractor）        | |
    | | | + Physics 1     | |  网格/部件 actors + 轴 + 方向指示器            | |
    | | + Regions         | |  （右键: Fit / 线框 / 实体 / 按 Part 着色）     | |
    | | | + Fluid Domain  | +---------------------------------------------+ |
    | | + Parts           |                                                |
    | | + Scenes/Plots    |                                                |
    | | + Reports/Monitors|                                                |
    | +---- Properties ---+                                                |
    | |  选中对象: 属性表  |                                                |
    +----------------------+-----------------------------------------------+
    | Output / Messages / Progress（QTabWidget 底部）                        |
    +-----------------------------------------------------------------------+
    | 状态栏: (x,y,z) 坐标 · 顶点/面数 · 单位 · 选择模式                        |
    +-----------------------------------------------------------------------+

与 STAR-CCM+ 手册对应：Simulation tree（左侧导航）、Graphics window（中间场景/
绘图标签页，每个 Scene/Plot 一个标签）、Properties（左下）、Output/Messages（底部）、
状态栏。参考实现：cabdecoding 的 TreeListView/ControlWindow/Draw/Message 五窗格与
cstdecoding 的 NavigationTree/PropertyInspector/CST3DViewport/MessageWindow。

---

## 4. 模块划分与类设计

    stardecoding/
    +- star_gui.py            # 入口 + StarMainWindow（菜单/工具栏/布局/打开/命令分发）
    +- star_gui_panes.py      # 窗格：SimulationTree / PropertiesPanel / GraphicsTabs /
    |                         #   MessageWindow / ProgressPanel / StatusBar / PaneFrame
    +- star_gui_vtk.py        # 场景图→VTK：build_scene / mesh actors / axes / 方向指示器
    |                         #   / 相机视图（VisView→camera）/ 拾取
    +- star_gui_icons.py      # AppIcons（QStyle 标准图标 + 内嵌 SVG/PNG，对齐 cab_icons.py）
    +- star_gui_theme.qss     # QSS 主题（深色背景、CAE 风格、objectName 选择器）
    +- star_gui_i18n.py       # （可选）中英文字符串表，对齐 cab_i18n.py
    +- requirements-gui.txt   # PyQt5 / vtk / numpy
    +- tests/                 # pytest：模型层单测 + 21 文件冒烟 + offscreen GUI 冒烟

### 4.1 数据流（单一数据源：SimFile）

    SimFile --→ StarSceneModel（适配器，只读包装）
      +- tree()      → SimulationTree（QTreeWidget，复用 build_tree/children）
      +- properties()→ PropertiesPanel（QTableWidget，SimObject.dict + resolve_class）
      +- scenes()    → GraphicsTabs（每 Scene 一页：scene→displayers→parts/颜色）
      +- mesh()      → star_gui_vtk.build_scene(...)（extract_mesh 的面/顶点）
      +- report()    → 状态栏/消息（层统计、版本指纹、长度自校验结果）

关键约束：
- **视图键（view keys）**：沿用 cst_gui 的命名视图范式——每个场景/显示器/部件一个
  key（如 scene:Mesh Scene 1、part:Fluid Domain），选中/显隐/颜色经 key 索引，
  树、属性、3D 三处选中同步。
- **命令分发 + _nyi**：菜单/工具栏动作集中注册（File/Edit/Mesh/Scene/Plot/Tools/Help），
  未实现项调用 _nyi(name) 写消息窗口（对齐 cabdecoding）。
- **不拷贝数据**：3D actor 直接由 extract_mesh 的 numpy 数组构造 vtkPolyData
  （复用 cab_vtk 的 _tris_to_polydata 模式）；大模型（13MB 文件约 10 万面）按需延迟构建。

### 4.2 类清单（关键类 + 职责）

| 类 | 职责 | 关键方法 |
| --- | --- | --- |
| StarMainWindow | 主窗口/菜单/工具栏/命令 | open_file, fit_view, toggle_wireframe, _nyi |
| StarSceneModel | SimFile 适配层 | tree(), properties(oid), scene_graph() |
| SimulationTree | 左树（QTreeWidget） | rebuild(roots), select→信号 |
| PropertiesPanel | 属性表 | show_object(obj) |
| GraphicsTabs | 场景/绘图标签页容器 | add_scene_tab(scene), current_view() |
| Star3DViewport | QVTK 视口（单场景） | set_mesh(actors), set_camera(view), pick() |
| star_gui_vtk.build_scene | 场景图→actors | 面网格/线框/轴/方向指示器/边界盒 |
| MessageWindow / ProgressPanel | 底部输出/进度 | log(), nyi(), progress() |
| AppIcons | 图标引擎 | get(name), 主题化 |

### 4.3 3D 显示技术细节（几何解析显示路线）

| 显示项 | 实现路线（借鉴参考） |
| --- | --- |
| 网格面 | extract_mesh() → (N,3) 顶点 + (M,3) 面 → vtkPolyData（复用 cab_vtk 的 _tris_to_polydata）；按 Part 分组 actor、独立显隐/颜色 |
| 网格边 | vtkExtractEdges + edges actor（cab_vtk.edges_actor：0.15 深灰边线） |
| 线框/实体/半透明 | actor.GetProperty() SetRepresentation/SetOpacity（cab_vtk.shaded_poly_actor） |
| 方向指示器 | vtkOrientationMarkerWidget（cab_vtk.orientation_marker_widget，左上角小轴） |
| 全局轴/原点 | axes_actor + world_origin_marker_actors（cab_vtk 同款） |
| 视图相机 | Scene 的 VisView/CurrentView 对象属性 → 位置/焦点/平行/透视（渐进实现，v1 先 Fit/±X±Y±Z 六视图 + Isometric） |
| 颜色 | Part 颜色（object graph 中的 colour/RGB 属性与状态表 _COLOUR 记录）；缺省调色板按 Part 循环 |
| 拾取 | vtkCellPicker → 面 id → 状态栏 (x,y,z) + 树同步选中 |
| CAD 几何（渐进） | v1 只显示网格（Representation=Mesh）；M5 起解析状态表 T 块/数组中的几何定义（coordinates/normals 已在状态表确认），对齐 cabdecoding 的 Parasolid 面片路线（ps_facet2_nodes/ps_tessellate） |

### 4.4 场景/绘图标签页

- 每个 star.vis.Scene 一个标签页（semantic_report 已给 Scene→Displayer→View）；
  场景无 3D 数据时显示占位文本。
- Plots/Monitors/Reports：v1 在仿真树与属性面板可见；绘图曲线重建列入后续（M6+）。

---

## 5. 测试与验收

- **模型层单测**（无 GUI）：StarSceneModel 在 21 文件语料上的 tree/properties/scene
  结果与 semantic_report/self_test 交叉断言。
- **GUI 离屏冒烟**：QT_QPA_PLATFORM=offscreen + VTK offscreen，打开文件→建树→建
  actor→渲染一帧→关闭（复用 cabdecoding 的离屏测试范式）。
- **回归**：self_test.py（解析层）+ batch_parse.py（21 文件）保持不变，GUI 层
  不回归解析层契约。
- **验收标准**：21 文件全部可打开；18/19 个有网格文件 3D 面网格显示正确（面数/顶点
  数与 extract_mesh 一致）；adjointWing 打开≤3s（13MB 文件≤10s）；崩溃率=0。

---

## 6. 开放事项（待确认）

1. 界面语言：默认中文 + 英文术语（对齐本仓库文档）；是否需 star_gui_i18n.py 全量
   双语可切换？
2. 主题：深色（对齐 STAR-CCM+ 深色）还是浅色？先做深色 + QSS 可换。
3. 是否允许 --cli 无窗口模式（导出 STL/报告）？（建议提供，复用现有 CLI。）
4. 后续是否加入"轻量编辑/回写"（改名/颜色等）？v1 只读，回写风险高（见
   function_gap_analysis.md 第 5 节）。
