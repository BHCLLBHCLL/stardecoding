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
| 文件>全部保存/自动保存/模板 | persist/session（`cmd_save_all` 覆盖工作区全部已打开仿真文档、跨窗口逐一落盘（X2，`_open_documents()`/`WORKSPACE`）；`AutoSave` 可勾选 + `QTimer` 环 + `QSettings` 持久化；`AutoSave Now`/`Checkpoint` 出 `base@N.sim` 快照并按 keep 轮转；`make_backup` 覆盖写前存 `path~`；`Save As Template...`/`New from Template...` 支持 `.simt`） |
| 文件>导入 CAD/表面 | persist（`mesh_io.read_surface` 真读 STL/OBJ → MeshPart `ImportedVertices/ImportedFaces`；CAD 无 Parasolid 时同路径三角化） |
| 文件>导入体网格 | persist（CCM 经 `ccm_io` 读**边界三角化**入 MeshPart，记 `CcmCellCount`；不重建体单元——见 G3） |
| 文件>导出 STL/摘要/报告 | view（STL 三粒度：选中 Scene 按场景、选中 Part 按分块、否则全局；摘要/报告 JSON） |
| 编辑>撤销/重做 | session（CommandBus） |
| 编辑>复制/粘贴/删除/重命名 | persist（复制→`Clip` 子树快照并登记全局 `CLIPBOARD`（记录来源文档）；粘贴→同文档走 `CopyObjectCommand`（created 插入对象图行）/跨文档走 `paste_clip`（全新会话 id + `remap_value` 整型引用重映射 + `analogous_parent_id` 挂到目标同类父），X2；删除→Keys 摘除；重命名→对象行补丁） |
| 编辑>上一选择/下一选择/按名称搜索 | session |
| 网格>生成表面网格 | macro（找到 starccmw 则写宏 → 工作副本 `-batch` → 加载 out.sim；注意宏体实为 `generateVolumeMesh()`，语义错位待 G/W 波宏模板细分） |
| 网格>生成体网格 | kernel（`cmd_generate_volume_mesh` 本地 N 波流水线：表面细分→tet（scipy/Gmsh 双路由）→质量→重编号，N1+N3；水密守门+HEADLESS 安全；N6 统一质量 histogram/repair 接入 `mesh_quality.py`） |
| 网格>生成多面体网格 | kernel（`cmd_generate_poly_mesh`：tet→Voronoi 对偶 poly 单元，N3b；结果存 `_poly_mesh_result`） |
| 网格>生成 Trimmer 网格 | kernel（`cmd_generate_trimmer_mesh`：八叉树分级加密+表面切割单元，N3b；结果存 `_trimmer_mesh_result`） |
| 网格>清除/转 2D | needs_kernel（`_kernel_nyi`） |
| 网格>缩放 | persist（`TransformPartCommand` → Float8 顶点/`ImportedVertices` 回写，可撤销） |
| 网格>诊断 | view（指纹/长度/ClassVersions 自洽标志）；kernel N6 侧在 `mesh_quality.py` 提供质量 histogram/repair/`mesh_interface.py` interface/`mesh_amr.py` AMR 钩子，供运行环（P10）与湍流（P6）联调 |
| 场景>（本客户端保留，官方在 Vis 工具栏） | view |
| 求解/连接 | disabled（Run/Pause/Step/Stop 与 Server 均 `setEnabled(False)`，测试锁定） |
| 工具>诊断/选项 | view / session |
| 窗口>树/属性/输出/绘图/CAD | view（checkable 开关） |
| 帮助>帮助内容/文档目录/许可信息/关于 | view（X3：`star_gui_help.py` 解析 `doc_javadoc_catalog.md`（55 个 star.* 包总览 + 18 章节详解）并联动词义字典 `layer_of`/`LAYER_CN`/`resolve_class`；`HelpWindow`（`star_gui_helpwin.py`）检索框 + 主题列表 + 文本浏览。`cmd_help_contents`（F1）按仿真树选中对象 `ClassName` 给出**上下文语义帮助**（现名/包作用/语义层/相关条目），`cmd_help_documentation` 打开文档目录、`cmd_help_licensing` 弹诚实许可声明，`cmd_about` 复用 `about_text()`） |

> 「文件>宏」菜单已移除（旧表标 disabled）。宏能力现挂接在 网格>生成表面网格 与
> 树右键>执行网格操作（`_try_star_macro`）；A1 已落地 Java 宏**录制/回放**——见下节「自动化生态（A 波）」。

## 自动化生态（A 波）

| 能力 | 实现 |
| --- | --- |
| Java 宏录制（A1） | session（`macro_record.py`：`MacroRecorder.attach(bus)` 挂 `CommandBus.on_execute` 钩子，把 GUI 命令 → `.java`；`COMMAND_JAVA_MAP` 7 命令全覆盖——`Rename/SetProperty/Visibility` 产可编译 ClientServerObject 调用，`ShowOnly/TransformPart/Delete/Copy` 私有语义无公开 API 则产 `// [best-effort]` 注释；`record_operation` 登记网格/物理/求解/后处理动作；`to_java`/`save` 落 ASCII `.java`） |
| 宏回放（A1） | macro（`MacroRecorder.replay` → `star_macro.run_star_macro_stream`：`starccmw -batch` 工作副本周转 + 逐行日志回流） |
| Python 脚本 API（A2） | session（`star_api.py`：镜像 `star.*` ClientServerObject 对象模型——`ClientServerObjectKey` 惰性定位 + `ClientServerObject` 及 14 语义子类（含 `Boundary.region`/`Displayer.scene`/`Model.continuum` 反查）+ `Simulation` 集合/`get`/`get_by_name`；写操作经 CommandBus 可撤销；`run_python_script` 注入 `star`/`sim`/`objects` 命名空间） |
| 参数研究 / DOE（A3） | session（`design_study.py`：`DesignParameter` + 4 类 DOE（full_factorial/ofat/latin_hypercube/random_sampling）+ `DesignTable`/`ResponseTable`（统计/最优/CSV）+ `DesignStudy.run(workers)` 并行批次 + 逐算例错误隔离；`bind_object` 对接 A2 对象属性） |

## 客户端体验收尾（X 波）

| 能力 | 实现 |
| --- | --- |
| 会话生命周期（X1） | persist/session（`star_gui_session.py`：覆盖写前 `path~` 备份；`AutoSave` 策略 enabled/interval/keep/trigger + `base@N.sim` 快照命名与保留最近 N 份轮转；CHECKPOINT 触发文件一次性消费；`.simt` 模板扩展名。GUI：`cmd_save_all`/`cmd_save_template`/`cmd_new_from_template`/`cmd_toggle_autosave`/`cmd_autosave_now`/`cmd_checkpoint` + `QTimer` 自动保存环 + `QSettings` 持久化，详见「文件>全部保存/自动保存/模板」） |
| 多仿真文档窗口（X2） | session（`star_gui_documents.py`：`DocumentWorkspace` 多文档登记 —— `add(doc, owner=)`/`remove`/`owner_of`/`activate`/`active`/`documents`/`paths`/`find_by_path`/`close_all`；`star_gui.py`：模块级 `OPEN_WINDOWS` 窗口登记 + `File>New Window`（Ctrl+Shift+N）→ `cmd_new_window` 开独立 `SimDocument` 窗口并 `WORKSPACE.add(owner=win)`；`on_file_loaded`/`close_sim`/`closeEvent` 同步工作区登记/注销） |
| 跨仿真复制粘贴（X2） | persist（`star_gui_documents.py`：源子树快照 `Clip`（`copy_subtree` 记录根源 `ClassName` 到 `parent_class`）→ 目标文档 `plan_id_mapping` 从 `_next_id` 起分配非冲突会话 id → `sim_writer.remap_value` 递归重写 Keys/Parent 等整型引用 → 登记 `created`/`objmap`/`objects` → `analogous_parent_id(doc, class_name)` 按类名挂到目标同类父（显式 parent 优先）；落盘由 `sim_writer.created_id_mapping` 把会话 id 映射到图序号 id，保证写出对象图引用自洽） |
| 帮助系统（X3） | view（`star_gui_help.py`：解析 `doc_javadoc_catalog.md` 包总览表（55 包）+ 18 章节详解 → `packages`/`package_of`/`package_description`/`package_section`/`class_bullets`；联动词义字典 `semantic_layer`（`layer_of`+`LAYER_CN`）/`resolve_class`（旧类名升级）→ `explain`/`class_help_text`/`search`/`help_contents_text`/`documentation_text`/`licensing_text`/`about_text`。`star_gui_helpwin.py`：`HelpWindow(QDialog)` 检索框 + 主题列表 + 文本浏览。GUI：`Help>Help Contents`（F1，按选中对象 `ClassName` 给上下文帮助）/`Documentation`/`Licensing`/`About`） |

## 工具栏

| 组 | 能力 |
| --- | --- |
| 系统：新建/打开/保存 | persist（保存走 `save_sim`） |
| 编辑：复制/粘贴/选择历史 | persist / session（选择历史 session） |
| 网格生成：导入/修复/生成 | 导入 persist；修复 needs_kernel；生成 macro（同上宏桥） |
| 求解 | session/persist（Run/Pause/Step/Stop + 控制器 + 残差实时曲线，P10） |
| Vis：适配/视图/透明/网格开闭/派生零件/标量着色 | view + session；标量着色在**点数或面数吻合**的一维数组上生效（`cmd_scalar_color`），无候选数组时禁用并说明 |
| 选择：框选缩放 / 测距 / Parts 过滤器 | 框选=VTK rubber band 真实现；测距=两点真实现；Parts 过滤器=勾选 Part/PartSurface → `Collector.Keys` persist |
| 3D-CAD | session 外壳（模式切换 + 三角化变换/显隐）；草图/拉伸/旋转/放样/管道（OCC 构造算子 `occ_builder.py`）+ 布尔/圆角/倒角/抽壳/阵列/镜像（OCC 编辑算子 `occ_edit.py`）+ 表面修复（hole fill/coarse/fine/质量指标 `occ_repair.py`）+ 表面包裹（收缩包裹/局部加密/特征捕捉 `occ_wrap.py`），产物三角化入图可显示、落 STEP/IGES/BREP |

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
| 二进制状态表（G9，写侧前置） | view/persist 前置（`parse_state_table_binary` 长度前缀完整文法 + `serialize_binary_records` 逐字节可逆；4 个 binary 文件往返一致；CLI `--binary-verify`） |
| 数组块变长替换/删除（W1） | persist（`apply_array_ops`：replace/delete 已有数组块 → 全量重定位后续对象 line/数组 offset + 迭代重算头部 StatePosition（位数收敛、同分区重指向）；主状态表数组变长明确拒绝留 W2；CLI `--array-op` `replace:IDX=N`/`delete:IDX`+`--edit-out`；变长替换/删除→重开对象图/数组数/StatePosition 一致） |
| 状态表安全编辑（W2） | persist（`edit_binary_state_records` 只动已确证记录（G1/G9 产物）：named 头等宽字段 id/flags/version/等长 name 可改写，其余字节/尾部/魔数逐字不变并差分验证；变长编辑明确拒绝留 W1；CLI `--state-edit`+`--edit-out` 原位替换 Character1 载荷达成真实 Save As；单/多段魔数 binary 文件编辑→重开一致） |
| ClassVersions 一致性维护 + NameManager 写入（W4） | persist（`maintain_class_versions`：create 后重写尾部 ClassVersions `Versions`（新对象类计数累加/新类增键、其余类原值），id=图序号+2 严格连续 `check_sequential_ids()`；`write_name_manager` 保守等宽写既有 `ObjectId`（width=0），变长新增/名字大表如实拒绝留 W1；原始 NameManager 空标记保留；CopyObject→Save As 重开 matched 不下降且尾段合法） |
| 引用/字典/嵌套结构属性全可写（W5） | persist（`audit_write_references`：semantic_dict DOWN/UP 白名单扩展写侧，down 引用集合须可解析、up 标量引用须可解析/None、NON_REF 枚举跳过；悬空/已删除引用捕获；`format_repr` 嵌套 dict/list/str/float/None 忠实往返；重设 Parent（up）+ MonitorPrintOrder 嵌套 dict + DisplayerColor 嵌套 list→Save As 重开全命中、审计为空） |
| ZIP/PK 容器写出（W3） | persist（`save_sim` 容器读写：`PK` 头检测→读侧取单条/最大条目为**主载荷**（补丁基底），写侧重打包成 ZIP 容器（保 `sim.container_entry` 条目名 + DEFLATED）；明文 `.sim` 不受影响；合成 ZIP 打补丁 Save As → 仍 `PK` 头 / 条目名保持 / 补丁命中载荷 / 对象图/数组不变；无补丁纯往返重开完全一致） |
| 差分回归自动化（W6） | test（`tests/test_diff_regression.py`：读语料→类型化补丁→Save As→重读→结构自检差分（对象/数组/Contants/id 序号）+ `compare_object_graph` 源/输出路径差分（仅目标字段 diff）；数组等宽载荷写回命中；`try_official_resave` 官方差分门控，无许可 auto-skip 结构自检主路径） |
| 物理参数语义行（G7: 模型/值/运动） | persist（P1 写侧：物理量/选项/嵌套组标量/顶层标量描述符行可编辑——kind/oid/key 锚点经 objmap 路由 SetPropertyCommand，Save 走 patches 整行替换；CLI `--physics` 同源只读） |
| 场函数表达式求值器（P2） | view（`field_fn.py` 词法-语法-求值全链：数学/矢量/逻辑 + 嵌套三元 + 插值器（`interpolateTable(@Table("name"), "column", SPLINE|LINEAR, "units", ${Position}[i])`）+ 惰性 `alternateValue`/`altValue` 多参；与官方语法对齐——变量引用 `${Name}`/`$$Name`/`$$$Name`、矢量字面量 `[a,b,c]`、张量点方法 `$$$A.eigValue(i)`/`$$$A.mag()`；依赖网格/几何（insidePart/distanceToPart/curl/div/grad）诚实 `NotImplementedError`，`compile_expression` 编译预检 + 错误路径诚实拒绝；`tests/test_field_fn.py` 39 项全绿，occ+scdm 两环境 self_test/batch_parse/run_all 无回归） |
| 初始化器（P3） | view（`init_solver.py` `Initializer`：常量/表格/场函数声明 → `compile()` 预检（source_field 登记+语法）→ `value()` 单点求值（常量直返/函数按 `${Position}`/`${Time}`）→ `field()` 批量求值（vector auto/mag）→ `apply_initial()` Run 前对 `solver.vertices` 求值写初场）；`source_field` 指定初场场函数；接入 `solver_run.py`——`DemoDiffusionSolver`/`SolverBackend` 注入 initializer（`set_initializer`/`_set_initial_field`/`_initialize_field`，缺省回落零场+源边界值），`run_loop` 循环起始注入；错误路径诚实拒绝（source_field 缺失/未知函数/语法错误/初场维度不匹配）；`tests/test_init_solver.py` 16 项全绿，occ+scdm 两环境 self_test/batch_parse/run_all 无回归） |
| FVM 离散核心（P4） | solver（`fvm_core.py` `FVM` 四面体单元中心面拓扑 + `grad_gauss`/`grad_lsq` 梯度、`limiter`/`_barth_ratio` Barth-Jespersen TVD 限制器、`diffusion_flux`/`convection_flux_upwind`/`convection_flux_central` 通量、`face_value` 面插值，全部 numpy 向量化（scdm 无 scipy 可运行，occ 可选 scipy.sparse）；`cube_tet_mesh` 网格构造；接入 `solver_run.py` `FvmDiffusionSolver`（单元中心场/节点初场四顶点均值映射/dt 稳定步长 `0.8/max(Σ γA/d/V)`/显式前向欧拉，`SolverBackend` 状态机可直接驱动）；`tests/test_fvm_core.py` 26 项全绿 + `solver_run` 6 项 FVM 集成测试，occ+scdm 两环境 self_test/batch_parse/run_all 无回归） |
| 压力基求解器（P5） | solver（`pressure_solver.py` `PressureSolver` 基于 P4 `FVM` 单元中心面拓扑的**速度-压力耦合**稳态不可压 SIMPLE 分离求解器：稀疏线性 `solve_linear` COO 双路径——occ scipy CSR + `pyamg` AMG / `spilu`+`bicgstab` / `spsolve`，scdm 纯 numpy Gauss-Seidel/SOR（对角占优收敛）；SIMPLE 外部迭代（3 动量预测→Rhie-Chow 面质量通量（`d_f[(∇p)_interp·n − (p_N − p_O)/d_n]` 消除棋盘）→压力修正泊松→`p += α_p p'`/`u -= d∇p'`+直接面通量修正 `mdot_f += γ(p'_O−p'_N)`→`_fix_boundary_mdot` 出口全局守恒 `sum(mdot)=0`）；BC 速度 Dirichlet 入口/壁+出口零梯度、压力 Neumann+出口压力参考；欠松弛 α_u=0.7/α_p=0.3；残差=归一化连续性质量不平衡收敛曲线健康；`step()`/`_initialize_field`/`residual`/`monitor_payload` 满足 P10 SolverBackend 可直接驱动到 COMPLETED；`tests/test_pressure_solver.py` 20 项全绿 + `self_test.py` P5 锚点，occ+scdm 两环境 self_test/batch_parse/run_all 无回归） |
| 湍流族（P6） | solver（`turbulence.py` 壁面处理/壁面函数：`wall_distance`/`y_plus`/`wall_function`（粘性底层-对数律光滑混合）/`wall_shear` + 边界分类/应变率；湍流模型族 `SpalartAllmarasSolver`（一方程 ν̃）/`KEpsilonSolver`/`KOmegaSSTSolver`（F1/F2 混合+CD_kw）/`LESSmagorinskySolver`（Smagorinsky C_s+Van Driest 阻尼）；`make_model` 注册表别名不敏感 + `TurbulenceSolver` P10 SolverBackend 门面；ν_t 经 `PressureSolver(turb_model=...)` 注入动量扩散 `(μ+ρν_t)A/d`，SIMPLE 环内 `_update_turbulence()` 解湍流输运，基线零 ν_t 无回归 + 4 模型集成残差收敛健康；`tests/test_turbulence.py` 21 项全绿 + `self_test.py` P6 锚点，occ+scdm 两环境 self_test/batch_parse/run_all 无回归） |
| 能量/传热（P7） | solver（`energy.py` 温度标量输运：`assemble_energy_transport` 保守形式 `∇·(ρCp·u·T)−∇·(κ_eff∇T)=S_T`，κ_eff=κ+κ_t、κ_t=Cp·ρ·ν_t/Pr_t + `solve_energy` 欠松弛迭代；传热边界全谱系 `BND_ADIABATIC`/`BND_FIXED_TEMP`/`BND_FIXED_FLUX`/`BND_ROBIN`（对流 `q=h(T_ref−T_w)`）/`BND_RADIATION`（`εσT⁴`），边界对流入流源流用比热 `c=cp·mdot` 严谨守恒；边界分类 `classify_thermal`/`thermal_btypes`；`EnergySolver`（`set_materials`/`set_radiation_volume`/`update(u,v,w,mdot,nu_t)`）+ P10 SolverBackend 门面；`ConjugateSolver` 共轭传热（固-液两域/`set_solid_material`/多次扫掠）；`RadiationModel` Stefan-Boltzmann 简化辐射（`net_emission`/`h_rad`/`volume_source`/`linear_source`）；工厂 `make_energy(model="energy"|"conjugate"|"cht")`；Boussinesq 浮力耦合——`PressureSolver(beta,gravity,buoy_ref_temp,inlet_temp,energy_model=...)` 惰性 `_ensure_energy_model()` + SIMPLE 环尾 `_update_energy()`，动量装配注入 `S=−ρ₀β(T−T₀)g·V`；`tests/test_energy.py` 17 项全绿 + `self_test.py` P7 锚点，occ+scdm 两环境 self_test/batch_parse/run_all 无回归） |
| 多相 VOF→Mixture→DPM（P8） | solver（`vof.py` 两相相体积分数 α 守恒输运 + PLIC 几何重构 + 表面张力：物性混合 `blend_property`/`two_phase_rho`/`two_phase_mu`（ρ=αρ₁+(1-α)ρ₂）；PLIC `plic_normal`（Green-Gauss ∇α+退化回退）/`plic_plane_offset`（二分）/`keep_phase1_volume`（`mesh_poly.clip_convex` 体积截断，输出夹取 [0,V]）/`plic_reconstruct`（逐单元界面满足目标 α）；守恒输运 `face_upwind_alpha`/`advance_alpha`/`auto_dt`（显式 CFL 受限一阶上风+界面压缩+[0,1] 裁剪+守恒缩放，入流注相 1）；表面张力 `surface_tension_force`（CSF）；`VofSolver`（`update`/`rho`/`mu`/`normal`/`offset`/P10 SolverBackend 门面）+ 工厂 `make_vof`；单流体耦合——`PressureSolver(vof_model="vof")` 惰性 `_ensure_vof_model()` + SIMPLE 环尾 `_update_vof()`，`_face_rho`/`_face_mu` 恒参考密度保持单相矩阵稳定（密度差显式重力体源因水/气密度比过大致 SIMPLE 发散已移除，静水/浮力留待自由面算例 g_rgh 形式）；`tests/test_vof.py` 16 项全绿 + `self_test.py` P8 锚点，occ 环境 test_vof 全绿无回归）；Mixture（多相 drift-flux）——`mixture.py` 物性混合 `mixture_rho`/`mixture_mu`/`blend_nphase`（ρ_m=Σα_kρ_k、μ_m=Σα_kμ_k）+ Schiller-Naumann 拖曳 `drag_coefficient` + 代数滑移 `slip_velocity`/漂移 `drift_velocities`（Σα_k u_dr,k=0 总体积守恒）+ 守恒输运 `advance_mixture`（半隐式有界）/`auto_dt`；`MixtureSolver`（alphas/rho/mu/slip/drift/P10 门面）+ 工厂 `make_mixture`；单流体耦合——`PressureSolver(mixture_model="mixture")` 惰性 `_ensure_mixture_model()` + SIMPLE 环尾 `_update_mixture()`，`rho`/`mu` 属性经 mixture_model 读混合物性；`tests/test_mixture.py` 13 项全绿 + `self_test.py` P8 Mixture 锚点。DPM（离散相粒子）——`dpm.py` `locate_cells` 四面体 barycentric 定位 + 半隐式欧拉运动积分 `advance`（dU_p/dt=(U_f−U_p)/τ+g(ρ_p−ρ_f)/ρ_p）+ `inject` 入口注入 + 连续相耦合源 `coupling_source`（−Σ m_p (U_f−U_p)/τ / V_c）；`DpmSolver`（n_particles/n_active/n_escaped/trajectories/P10 门面）+ 工厂 `make_dpm`；单流体耦合——`PressureSolver(dpm_model="dpm")` 惰性 `_ensure_dpm_model()` + SIMPLE 环尾 `_update_dpm()` + `_assemble_momentum` 注入拖曳反作用力体源；`tests/test_dpm.py` 10 项全绿 + `self_test.py` P8 DPM 锚点，occ 环境全绿无回归） |
| 燃烧算例：组分输运 + 反应动力学（P9） | solver（`species.py` 组分质量分数输运：组分换算 `mass_to_mole_frac`/`mole_to_mass_frac`/`mixture_molar_mass`/`gas_density`（X_i=(Y_i/Mw_i)/Σ_j(Y_j/Mw_j)、W_mix=1/Σ_j(Y_j/Mw_j)、ρ=P W_mix/(R T)，零分数全零保护免除零）；单组分对流-扩散输运 `assemble_species_transport`（∂(ρY)/∂t+∇·(ρuY)=∇·(ρD_eff∇Y)+S，D_eff=D_mol+nu_t/Sc_t）+ `solve_species` 欠松弛；边界分类 `classify_species`（入口/出口/壁面/边界面全集全覆盖不重叠）+ `volume_mass`；`SpeciesSolver`（N-1 输运+补齐组分保证 ΣY=1、update/Y/w_mix/P10 SolverBackend 门面）+ 工厂 `make_species`（species/multispecies/mass_fraction）。燃烧反应动力学 `combustion.py`：全局单步 Arrhenius 反应 `GlobalReaction`（ω=A T^β exp(−Ea/(RT)) Π_i[C_i]^ν_i、`mass_source` 按 Mw_i ν_i 自动满足 ΣS_i=0、`heat_release` Q=ω H_rxn）+ 火焰诊断 `laminar_flame_speed`/`flame_thickness`（δ_L=α/S_L）/`combustion_progress` + 点火器 `make_ignitor`（Temperature/Spark（`_safe_norm`）/ProgressVariable 调制反应速率+火花热源）；`CombustionModel`（内嵌 SpeciesSolver、set_temperature/update(...,T=...)、heat_release/omega/ignited/P10 门面）+ 工厂 `make_combustion`（combustion/globe/global）。PressureSolver 耦合——`PressureSolver(species_model=...,combustion_model=...,combustion_inlet_fractions=...,combustion_ref_temp=...)` 惰性 `_ensure_species_model()`/`_ensure_combustion_model()`/`_update_species()`/`_update_combustion()` 注入，SIMPLE 环尾推进组分/燃烧源（燃烧单向读能量模型 T 供 Arrhenius 速率）、`monitor_payload` 增 sp_*/combo_* 键；纯 numpy 约束范数走 `solver_run._safe_norm`；`tests/test_combustion.py` 19 项全绿 + `self_test.py` P9 锚点，occ 环境 self_test 全绿无回归） |
| 多相欧拉-欧拉（P10 多相流算例） | solver（`eulerian_multiphase.py` 欧拉-欧拉多相：物性混合 `mixture_rho`/`mixture_mu`（复用 mixture.py）+ 相间作用力全谱系——拖曳交换 `drag_exchange`（Schiller-Naumann `drag_coefficient` + Wen-Yu 空泡率修正 `f(α_c)=α_c^-2.65`，低 Re 回归 Stokes `18μ_c α_d/d²`）、升力 `lift_force`（Legendre-Magnaudet 简式 `−C_L α_d ρ_c u_rel×(∇×u_m)`）、虚拟质量 `virtual_mass_force`（`C_vm α_d ρ_c a_m`，a_m 用对流加速度 `(u_m·∇)u_m` 近似）、壁面润滑 `wall_lubrication_force`（Antal 公式 `−C_wl α_d ρ_c |u_rel|²/d·n_wall`，`C_wl=max(0, Cw1+Cw2 d/y_w)`）；群体平衡 `population_balance_source`（聚并 i→j / 破碎 j→{k<j} / 成核 0→1，Σ_k S_k=0 守恒）+ 核函数 `coalescence_kernel`/`breakup_kernel`/`nucleation_rate`；`EulerianMultiphaseSolver`（N 相相体积分数 α 守恒、`update` 返归一化残差、`alphas`/`rho`/`mu`/`slip`/`drift`/`relative_velocities`/`phase_velocities`/`momentum_source`/`interphase_k` 场属性、P10 SolverBackend 门面）+ 工厂 `make_eulerian_multiphase`（别名 eulerian/eulerian_multiphase/ee/euler/euler_euler）；单流体耦合——`PressureSolver(eulerian_model=...,eulerian_inlet_alphas=...,eulerian_enable_lift/virtual_mass/wall_lubrication/population_balance=...)` 惰性 `_ensure_eulerian_model()` + SIMPLE 环尾 `_update_eulerian()`，`rho`/`mu` 属性经 eulerian_model 读混合物性，`monitor_payload` 增 ee_n_phases/ee_alpha_min/max/sum/ee_mom_src 键；纯 numpy 约束范数走 `solver_run._safe_norm`；`tests/test_eulerian_multiphase.py` 18 项全绿 + `self_test.py` P10 锚点，occ 环境 self_test/run_all 全绿无回归） |
| 运动谱系算例：刚体运动+滑移 interface → morphing → DFBI 6DOF → overset + MRF 旋转源（P11） | motion（`motion.py` 运动谱系全链：①刚体运动+滑移 interface——`rotate_points` Rodrigues 旋转 / `rigid_transform` 刚体变换（平移参考系原点→旋转→平移回+平动位移）/ `sliding_interface` 滑移面标识；②morphing——`_build_vertex_adjacency` 顶点邻接+边界顶点集 / `morph_mesh`（边界夹持+内部 Laplacian 松弛）；③DFBI 6DOF——`DfbiBody`（mass/inertia/position/quaternion/`_quat_mul`/`_quat_rotate`/`quat_to_dcm` 方向余弦、`world_inertia`/`inv_inertia`/`advance` 半隐式欧拉推进）；④overset——`overset_donor_weights`（最近 k 个 donor 反权重，硬切换退化）/`overset_interpolate`；⑤MRF 旋转参考系源——`mrf_source`（离心 `S=ρω²r_perp` + 科氏 `S=−2ρ Ω×u`，omega=0 归零）；`MotionSolver`（mode 决定更新行为、P10 SolverBackend 门面 `_initialize_field`/`step`/`residual`/`monitor_payload`、`mrf_source(rho)`/`mesh_displacement()`/`body_kinematics()`）+ 工厂 `make_motion`（别名 motion/rigid/rotating/rotation/mrf/sliding/morph/morphing/deform/dfbi/6dof/sixdof/overset/overlapping/chimera，大小写/连字符/下划线不敏感）；对应 STAR-CCM+ `star.motion` 系（Rigid Body Motion / Sliding Interface / Morphing / DFBI 6DOF / Overset Mesh / Rotating Reference Frame）——原 P9 燃烧顺延、经确认补全；PressureSolver 耦合——`PressureSolver(motion_model=...)` 惰性 `_ensure_motion_model()` + SIMPLE 环尾 `_update_motion()`，`_assemble_momentum` 注入 MRF 旋转源（离心+科氏体源×V），`monitor_payload` 增 motion_mode/motion_rotation_speed/motion_disp_max 键，默认 None 零回归；纯 numpy 约束范数走 `solver_run._safe_norm`；`tests/test_motion.py` 22 项全绿 + `self_test.py` P11 锚点，occ 环境 self_test/全量 pytest 486 passed 27 skipped 无回归） |
| 可压缩/密度基流算例：理想气体 EOS + 压力基可压缩 SIMPLE 密度耦合（P12） | solver（`compressible.py` 可压缩/密度基流全链：①理想气体 EOS `ideal_gas_rho(p,T,W_mix,R)` `ρ=p W/(R T)`（标量/数组 + 零温保护）/ `density_derivative` 等温压缩率 `(∂ρ/∂p)_T=W/(RT)=γ/c²`；②声学诊断 `sound_speed` `c=√(γRT/W)=√(γp/ρ)` / `sound_speed_pr` / `velocity_magnitude` / `mach_number` `M=|u|/c`（零声速保护）/ `dilatation` 体积膨胀率 `∇·u` / `compression_work` 压缩功 `−p∇·u`；③压力基可压缩耦合 `density_correction` 等熵密度修正 `ρ'=p'/c²` / `compressibility_diagonal` 压力修正附加对角项 `V/(c²Δt)` / `density_change_source` 非稳态密度质量源 `(ρ−ρ_prev)V/Δt`；④`CompressibleSolver`（`update` 密度欠松弛 `ρ=(1−relax)ρ_old+relax·ρ_EOS` 返归一化残差、`rho`/`temperature`/`gauge_pressure`/`pressure`（绝对）/`sound_speed`/`mach`/`dilatation`/`compression_work`/`max_mach`、`face_density`/`compressibility_diagonal`/`density_change_source`/`density_correction`、P10 SolverBackend 门面）+ 工厂 `make_compressible`（8 别名不敏感）；对应 STAR-CCM+ Compressible/Coupled Flow + Ideal Gas EOS 谱系——承接 P5 压力基与 P7 能量；PressureSolver 耦合——`PressureSolver(compressible_model=...,compressible_gamma/mw/p_ref/t_ref/dt/relax=...)` 惰性 `_ensure_compressible_model()` + SIMPLE 环尾 `_update_compressible()`，`rho`/`_face_rho` 属性经 compressible_model 读逐单元密度场/面密度，`_assemble_pressure_correction` 注入压缩性对角项与密度变化质量源实现压力-密度耦合，`monitor_payload` 增 cmp_* 键，能量-密度耦合单向读 `energy_model.T`，默认 None 零回归；纯 numpy 约束范数走 `solver_run._safe_norm`；`tests/test_compressible.py` 25 项全绿 + `self_test.py` P12 锚点，occ 环境 self_test ALL CHECKS PASSED 无回归） |
| 后处理：标量/矢量 color-by（V1） | view（`postprocess.py` 色彩映射内核：`DEFAULT_COLORMAP_VALUES` 官方风格蓝→黄→红 4n 断点 + `parse_colormap`（位置降序自动翻转；断点 <8 或非 4 倍数返 `(None,None,None)`）+ `sample_colormap` 重采样 `(n,4)` RGBA + `scalar_range`（含 robust 百分位）+ `map_scalars` 标量→RGBA + `vector_magnitude`（1D 直通）+ `color_by`（scalar/magnitude/component 统一入口，非 2D 场取分量诚实 `ValueError`）；纯 numpy，与 G8 `extract_scene_display` 官方色表断点同源） |
| 后处理：显示器全家桶（V2） | view（`postprocess.py`：`marching_tets` 逐四面体等值面（法向按标量梯度定向）+ `section_plane`/`clip_plane`（`keep="negative"` 保 `(x−p)·n≤0`）+ `extract_surface`（剔内部共享面）+ `threshold_cells`（非单元场诚实 `ValueError`）/`threshold_surface` + `mirror_geometry`（`T[:, ::-1]` 反绕序+merge）+ `vector_glyphs`（stride 抽稀+箭头）+ `streamline`/`streamlines`/`pathline`/`particle_trace`（RK 积分，`_safe_norm` 纯 numpy）；cube_tet_mesh(3) 等值面 1.0/剖面 1.0/半盒裁剪 4.0/全盒外表面 6.0 精确断言） |
| 后处理：派生零件（V3） | view（`postprocess.py`：`locate_cell` 体积坐标重心定位（域外 `(-1,None)`）/`sample_scalar`/`sample_vector`/`cell_to_vertex` + `probe`（points/cells/inside/values）/`line_sample`（+t/length）/`plane_sample`（+u/v/grid_shape/normal/point）/`iso_volume`（等值体积 fraction，接受逐顶点场按单元顶点均值折算）/`threshold_part`（+cells/count/volume/lo/hi）/`plane_box_polygon`/`DerivedCache`（`get_or_compute` 命中未命中计数 + key/clear/`__len__`）） |
| 后处理：绘图数据（V4） | view（`postprocess.py`：`xy_series` + `histogram`（counts/edges/centers/bin_width/n/range/mean/density/[pdf]）+ `cumulative_distribution`（x/cdf/n）+ `decimate_series`（x/y/n/original）+ `MonitorBuffer` 容量滚动缓冲（append/series/clear）；供 `star_gui_plots.py` 渲染，数据面与官方曲线同源） |
| 后处理：注记/图例/色标尺/动画（V5） | view（`postprocess.py`：`colorbar_ticks`/`colorbar_strip` 色标尺几何 + `legend_items` + `annotation` 注记链 + `frame_times`/`frame_indices`/`frame_name`（补零命名 `frame_0007.png`）+ `export_animation` PNG 序列/GIF 真写，mp4 无 ffmpeg 诚实降级 `"unavailable(ffmpeg)"`；动画帧序正确） |
| 后处理：数据写出 CSV/EnSight/CGNS（V6） | view（`postprocess.py`：`write_csv`/`export_csv`（含 cell/x/y/z 几何列）+ `write_ensight`（`.case`/`.geo`/`.dat` 三件套）+ `write_cgns`（CGNS SIDS HDF5，C=cells+1 1-based；无 h5py 诚实 `RuntimeError`）；**门面/工厂** `_solver_fv`（试 `fv`/`fvm`/`_fv`）+ `fields_from_solver`（velocity/speed/pressure/rho/T/mach）+ `PostProcessor`（color/streamline(s)/isosurface/section/clip/threshold/mirror/glyphs/derived/probe/line/plane/iso_volume/xy/histogram/cdf/colorbar/legend/animate/export_csv/export_ensight/export_cgns + `from_solver`）+ `make_postprocessor`（8 别名，未知 `ValueError`）；`tests/test_postprocess.py` 49 项全绿 + `self_test.py` V 波锚点，occ 环境全量 pytest 560 passed 27 skipped 无回归） |
| 保存视图 | persist（`persist_view` 写入 Scene `CurrentView` 对象行，不再只放内存） |
| 3D-CAD 模式外壳 | session（剖面/变换走三角化；草图/拉伸 needs_kernel）+ C1/C6 自研显示级：`surface_polydata` 外部 STL/OBJ→vtkPolyData 重建显示（顶点/面数校验）；`mesh_io` STL(ascii+binary)/OBJ 双向读写往返面数一致；B-Rep 建模（拉伸/布尔/圆角/包裹）与 STEP/IGES/BREP 依赖 OCP/官方桥→受限 |
| 绘图标签（监视器/报告数值） | view（Monitor/Report/Plot/Residual 标量 + 1D 数组折线≤512 点；明确标注非求解器采样；无数据降级为文本） |
| 体网格线框 | view（`extract_volume_mesh` + `volume_mesh_actors`，G3 存储体系精确抽取；数组对不上时禁用并写原因） |
| 场景显示参数（G8） | view（`extract_scene_display`：背景/灯光/显示器/场范围/图例/官方色表 4n 断点/注记链；CLI `--scenes` 同源只读；GUI `lut_from_colormap` 官方 ColorMap→vtkLookupTable 断点插值重采样 256 级真渲染标量着色） |

## G0 复审结论（相对 `star_gui_next.md` E8 口径的变化）

1. **过时口径修正 3 处**：树右键「重命名/删除/复制」session→persist；「文件>宏 disabled」→菜单已移除（宏桥挂在网格菜单）；「网格>生成表面 needs_kernel」→macro（有 starccmw 真跑）。
2. **F3/F4/F5/F6 均已按 star_gui_next.md 落地并经代码验证**：真 STL 读写、变换落盘、框选缩放/测距、Parts 过滤器、边界 FaceTypes 精确高亮、体网格线框（尽力）、1D 曲线与标量着色（有数据才开）。
3. **遗留不精确点**（转入 G/W 波）：
   - 「生成表面网格」宏体实为 `generateVolumeMesh()`，表面/体网格宏模板未细分（W/A 波宏映射表）；
   - 体网格导入只取边界三角化，体单元表未重建（G3）；
   - ~~`Save All`/AutoSave/.simt 模板未实现（X1）~~ → 已随 X1 落地（`star_gui_session.py` + `cmd_save_all` 等）；多仿真文档窗口 + 跨仿真复制粘贴已随 X2 落地（见「客户端体验收尾（X 波）」）；
   - Server/连接保持诚实禁用；求解 Run/Pause/Step/Stop 已随 P10 闭环启用（`solver_run.py` + 残差实时曲线，见菜单栏「求解/连接」）。
