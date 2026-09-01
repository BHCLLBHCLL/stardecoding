# star_gui 对标 STAR-CCM+ 20.02（除求解运算）

能力：`view` 已实现查看 · `session` 会话可改（重开丢失） · `persist` 可写回（Save As → 重开仍在） · `disabled` 灰显 · `needs_kernel` 需网格/CAD 内核 · `macro` 有 starccmw 时经宏桥在工作副本上真跑

> 2026-08-23 G0 复审：F0–F8 全部落地后的逐条代码审计（`star_gui.py` / `star_gui_panes.py` /
> `star_gui_vtk.py` / `star_gui_document.py` / `star_gui_commands.py` / `sim_writer.py`），
> 基线 `self_test.py` + `tests/run_all.py`(14 文件) 全绿。后续总路线见 `parity_100pct_plan.md`
> （G/W/C/N/P/V/A/X 八波）；F 波详情见 `star_gui_next.md`。
> 本表按代码审计，不按菜单存在与否乐观估计。

## 菜单栏

| 条目 | 能力 |
| --- | --- |
| 文件>新建/打开/关闭/最近/退出 | view |
| 文件>重新加载 | session（`confirm_discard_dirty` 脏提示，与关闭/新建对齐） |
| 文件>保存/另存为 | persist（`save_sim`：对象行 patches + 新对象 created 插入 + 数组覆盖 array_patches + 删除 deleted 摘 Keys） |
| 文件>全部保存/自动保存/模板 | disabled（Save All = `_nyi`；自动保存/模板无菜单项） |
| 文件>导入 CAD/表面 | persist（`mesh_io.read_surface` 真读 STL/OBJ → MeshPart `ImportedVertices/ImportedFaces`；CAD 无 Parasolid 时同路径三角化） |
| 文件>导入体网格 | persist（CCM 经 `ccm_io` 读**边界三角化**入 MeshPart，记 `CcmCellCount`；不重建体单元——见 G3） |
| 文件>导出 STL/摘要/报告 | view（STL 三粒度：选中 Scene 按场景、选中 Part 按分块、否则全局；摘要/报告 JSON） |
| 编辑>撤销/重做 | session（CommandBus） |
| 编辑>复制/粘贴/删除/重命名 | persist（复制→created 插入对象图行；删除→Keys 摘除；重命名→对象行补丁） |
| 编辑>上一选择/下一选择/按名称搜索 | session |
| 网格>生成表面网格 | macro（找到 starccmw 则写宏 → 工作副本 `-batch` → 加载 out.sim；注意宏体实为 `generateVolumeMesh()`，语义错位待 G/W 波宏模板细分） |
| 网格>生成体网格/清除/转 2D | needs_kernel（`_kernel_nyi`） |
| 网格>缩放 | persist（`TransformPartCommand` → Float8 顶点/`ImportedVertices` 回写，可撤销） |
| 网格>诊断 | view（指纹/长度/ClassVersions 自洽标志） |
| 场景>（本客户端保留，官方在 Vis 工具栏） | view |
| 求解/连接 | disabled（Run/Pause/Step/Stop 与 Server 均 `setEnabled(False)`，测试锁定） |
| 工具>诊断/选项 | view / session |
| 窗口>树/属性/输出/绘图/CAD | view（checkable 开关） |
| 帮助>关于 | view |

> 「文件>宏」菜单已移除（旧表标 disabled）。宏能力现挂接在 网格>生成表面网格 与
> 树右键>执行网格操作（`_try_star_macro`）；无录制/回放（A1）。

## 工具栏

| 组 | 能力 |
| --- | --- |
| 系统：新建/打开/保存 | persist（保存走 `save_sim`） |
| 编辑：复制/粘贴/选择历史 | persist / session（选择历史 session） |
| 网格生成：导入/修复/生成 | 导入 persist；修复 needs_kernel；生成 macro（同上宏桥） |
| 求解 | disabled |
| Vis：适配/视图/透明/网格开闭/派生零件/标量着色 | view + session；标量着色在**点数或面数吻合**的一维数组上生效（`cmd_scalar_color`），无候选数组时禁用并说明 |
| 选择：框选缩放 / 测距 / Parts 过滤器 | 框选=VTK rubber band 真实现；测距=两点真实现；Parts 过滤器=勾选 Part/PartSurface → `Collector.Keys` persist |
| 3D-CAD | session 外壳（模式切换 + 三角化变换/显隐；草图 needs_kernel） |

## 树 / 3D 右键

| 条目 | 能力 |
| --- | --- |
| 显隐/仅显示/高亮 | session（`VisibilityCommand`/`ShowOnlyCommand` 走命令总线，可撤销；高亮：Boundary 用 FaceTypes 切**精确**面子块着色，非整 Part 近似） |
| 重命名/删除/复制/粘贴 | persist（同编辑菜单路径） |
| 变换（平移/旋转/缩放） | persist（Float8 / `ImportedVertices` 回写，可撤销） |
| 指定到区域 | persist（已有 PartGroup 的 Keys 补丁 + 树/3D 即时可见） |
| 新建场景/添加显示器/Representation/Parts 过滤器 | 新建=persist（created 插入 + 重建 3D）；Representation 切换=session（按 `representation_source_id` 重建该场景页）；Parts 过滤器=persist |
| 执行网格操作/新建自动网格 | macro（`execute_mesh` 走宏桥）/ needs_kernel |
| 3D：适配/视图/线框实体边线/复制图像/框选缩放/测距 | view + session |
| 3D-CAD：进入模式 | session |

## 属性 / CAD / 绘图 / 体网格

| 条目 | 能力 |
| --- | --- |
| 类型化属性编辑 | persist（bool/数/色/矢量/数值列表可编辑；`ClassName/Parent/Simulation/NameManager` 等只读） |
| 保存视图 | persist（`persist_view` 写入 Scene `CurrentView` 对象行，不再只放内存） |
| 3D-CAD 模式外壳 | session（剖面/变换走三角化；草图/拉伸 needs_kernel） |
| 绘图标签（监视器/报告数值） | view（Monitor/Report/Plot/Residual 标量 + 1D 数组折线≤512 点；明确标注非求解器采样；无数据降级为文本） |
| 体网格线框 | view（`extract_volume_mesh` + `volume_mesh_actors`；数组对不上时禁用并写原因——精确抽取待 G3） |

## G0 复审结论（相对 `star_gui_next.md` E8 口径的变化）

1. **过时口径修正 3 处**：树右键「重命名/删除/复制」session→persist；「文件>宏 disabled」→菜单已移除（宏桥挂在网格菜单）；「网格>生成表面 needs_kernel」→macro（有 starccmw 真跑）。
2. **F3/F4/F5/F6 均已按 star_gui_next.md 落地并经代码验证**：真 STL 读写、变换落盘、框选缩放/测距、Parts 过滤器、边界 FaceTypes 精确高亮、体网格线框（尽力）、1D 曲线与标量着色（有数据才开）。
3. **遗留不精确点**（转入 G/W 波）：
   - 「生成表面网格」宏体实为 `generateVolumeMesh()`，表面/体网格宏模板未细分（W/A 波宏映射表）；
   - 体网格导入只取边界三角化，体单元表未重建（G3）；
   - `Save All`/AutoSave/.simt 模板未实现（X1）；
   - 求解/连接保持诚实禁用（P 波）。
