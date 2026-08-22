# star_gui 开发计划（按 star_gui_design.md 实施）

> 工作方式：与上一阶段（差距改进1~7）一致——每个里程碑完成即 git 提交并推送
> GitHub，同步更新本文件的勾选状态；解析层回归（self_test.py + batch_parse.py
> 21 文件）在每个里程碑末尾必跑。

## 里程碑总览

| 里程碑 | 内容 | 产出 | 状态 |
| --- | --- | --- | --- |
| M0 | 骨架：主窗口 + 打开文件 + 摘要 | star_gui.py 可运行 | 完成（09b9e25 之后提交） |
| M1 | 仿真树 + 属性面板 + 层报告 | 左栏可用 | 完成 |
| M2 | 3D 网格显示（核心） | 面网格 + 交互 | 完成 |
| M3 | 场景/显示器/视图 + 颜色 | 场景标签页 | 完成 |
| M4 | 消息/进度/状态栏 + 导出 + CLI 模式 | 完整窗口 | 完成 |
| M5 | Part 显隐勾选 + Region/Boundary 高亮 + 边线模式（几何显示渐进） | 几何增强 | 完成 |
| M6 | 打磨：主题/图标/i18n/测试/文档 | 发布候选 | 完成 |

> 收尾补充：M6 后新增 star_gui_i18n.py（可选双语基础设施：tr()/set_language，默认中文+英文术语）；PyInstaller 打包脚本留作后续（可选）。

---

## M0 —— 骨架与打开文件（1 个提交）

任务：
1. requirements-gui.txt（PyQt5/vtk/numpy）。
2. star_gui.py：StarMainWindow + PaneFrame 骨架（空窗格占位），菜单/工具栏按
   star_gui_design.md 第 3 节布局注册（全部动作接 _nyi）。
3. Open 对话框 → SimFile 加载（后台 QThread + 进度条）→ 摘要窗格显示
   summary()/version_fingerprint()/check_state_length()/layer_census()。
4. star_gui_icons.py AppIcons 初版（QStyle 标准图标起步）。

验收：命令行 python star_gui.py 出窗口；打开 adjointWing_start.sim 显示摘要；
offscreen 冒烟测试通过（tests/test_gui_smoke.py：QT_QPA_PLATFORM=offscreen）。

提交：M0 star_gui skeleton: main window, file open, summary pane

---

## M1 —— 仿真树 + 属性面板（1 个提交）

任务：
1. StarSceneModel 适配器（tree/properties 接口，纯数据、可单测）。
2. SimulationTree：build_tree/children → QTreeWidget；按 layer_census 分层图标
   （几何/网格/物理/场景/后处理各一图标）；节点勾选显隐（与 3D 联动的 view key 先
   挂空实现）。
3. PropertiesPanel：选中对象 → 属性表（名称/类型/值三列；别名解析列）；
   双击树节点/选中属性 → 高亮引用对象（objmap 跳转）。
4. tests/test_scene_model.py：21 文件 tree/properties 与 semantic_report 交叉断言。

验收：adjointWing 树形完整（Continua/Regions/Parts/Scenes/Plots/Monitors）；
点选 Region 显示属性含 PresentationName=Fluid Domain。

提交：M1 star_gui: simulation tree + properties panel + StarSceneModel

---

## M2 —— 3D 网格显示（核心里程碑，1-2 个提交）

任务：
1. star_gui_vtk.py 初版：build_scene()——extract_mesh() 的 (N,3)/(M,3) →
   vtkPolyData；分 Part 建 actor（MeshPart/CadPart 各自的 TriangleCount×3 面数组，
   优先复用改进2的 part: 匹配逻辑）；edges actor；axes + orientation marker。
2. Star3DViewport：QVTKRenderWindowInteractor 嵌入 GraphicsTabs；Fit/Reset；
   线框/实体切换；鼠标旋转/平移/缩放（默认 interactor style）。
3. 深色背景渐变（STAR-CCM+ 风格）。
4. 面拾取 → 状态栏 (x,y,z) + 树选中同步。
5. 21 文件批量冒烟：18/19 个有网格文件出网格（复用 mesh_validate 的判定）。

验收：adjointWing 显示 2824 面/1412 顶点；checkValve 4 个 Part 分色显示；
genericHelicopter（24 万面）渲染帧率可交互（vtkPolyData 单 actor/part）。

提交：M2 star_gui: 3D mesh viewport (extract_mesh→VTK, per-part actors, picking)

---

## M3 —— 场景/显示器/视图 + 颜色（1 个提交）

任务：
1. GraphicsTabs：每 star.vis.Scene 一页；无 3D 数据显示占位。
2. 场景→显示器映射：PartDisplayer→Parts（semantic_report 已有）；显示器颜色/透明度
   属性（object graph）→ actor property。
3. 视图相机：VisView/CurrentView 的 Position/FocalPoint/ParallelProjection 属性 →
   vtkCamera（缺省回退 Fit/Isometric；±X±Y±Z 六视图按钮）。
4. 场景右键：Fit / 线框 / 实体 / 全部显隐。

验收：adjointWing 的 Mesh Scene 1 标签页显示网格，视图 Current 2 相机参数生效
（与官方保存视图一致）；checkValve 双场景（Geometry Scene 1 / Mesh Scene 1）切换正常。

提交：M3 star_gui: scene tabs, displayer colors, saved view cameras

---

## M4 —— 完整窗口：消息/进度/状态栏/导出/CLI（1 个提交）

任务：
1. MessageWindow（日志 + _nyi 文案）、ProgressPanel（打开/渲染进度）、StatusBar
   （坐标/面数/单位/层统计）。
2. 导出菜单：STL（export_stl）、摘要 txt、层级报告 JSON（semantic_report/layers）。
3. --cli 无窗口模式：python star_gui.py --cli file.sim --export out 复用现有 CLI
   能力（便于脚本化）。
4. 21 文件 GUI 批量冒烟（offscreen：打开→建树→渲染→导出→关闭）。

验收：全流程无崩溃；export STL 与 sim_parser --mesh-export 输出一致。

提交：M4 star_gui: message/progress/status panes, export menus, --cli mode

---

## M5 —— CAD 几何显示（渐进）+ 边界着色（1-2 个提交）

任务（依赖 function_gap_analysis.md 第2/3节的持续逆向）：
1. 状态表 T 块/数组中的几何定义 → 几何面片 actor（坐标已在状态表确认；对齐
   cabdecoding 的 Parasolid 面片路线）。
2. Boundary/Interface 高亮：Region→Boundary→面片映射（树勾选边界 → 着色对应面）。
3. 网格-几何双 Representation 切换（PartRepresentation Geometry/Mesh 对象已解析）。

验收：含 cadmodeler 数据的文件（airfoil/manifold）显示几何面片；边界勾选着色
正确性以对象图元数据交叉验证。

提交：M5 star_gui: CAD geometry display + boundary highlighting

---

## M6 —— 打磨与发布（1 个提交）

任务：
1. star_gui_theme.qss 深色主题 + objectName 选择器；AppIcons 补齐。
2. star_gui_i18n.py（中文/英文切换，若开放事项确认）。
3. pytest 全套（模型层 + offscreen GUI + 21 文件）；README 用法章节更新。
4. （可选）PyInstaller 打包脚本。

验收：pytest 全绿；README/function_gap_analysis.md 状态更新；GitHub 最新提交
可 pip install -r requirements-gui.txt 后 python star_gui.py file.sim 直接运行。

提交：M6 star_gui: theme, icons, i18n, tests, docs

---

## 依赖、风险与对策

| 项 | 说明 | 对策 |
| --- | --- | --- |
| 大文件渲染 | 13MB .sim 约 24 万面 | vtkPolyData 单 actor/part、延迟构建、进度条 |
| 二进制状态表几何 | T 块文法未完全还原（gap 第2节） | M5 仅对已确证字段（坐标/面索引）做渐进支持，不阻塞 M0-M4 |
| 无网格文件 | methaneOnPt/AXMGeometry 等 | 树/属性/占位页正常，3D 显示空场景提示 |
| 解析层契约 | GUI 依赖 SimFile 接口 | 解析层回归（self_test/batch）每里程碑必跑，GUI 只读不写 |
| PyQt5 兼容 | 参考仓库同为 PyQt5 | 锁定 >=5.15；必要时 pin 版本 |

## 开放事项（实施前需用户拍板，见 star_gui_design.md 第 6 节）

1. 界面语言（默认中文+英文术语 / 全双语可切换）
2. 主题（默认深色）
3. --cli 无窗口模式是否纳入（建议纳入）
4. v1 是否严格只读（建议只读，回写列入远期）