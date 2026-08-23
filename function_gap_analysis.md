# .sim 解析：功能完整性与深度差距分析（GAPS）

> 依据：`startutorialsdata` 全部 21 个 .sim 实测（版本 6.03–10.05 / modeller 240014323–2800151，
> 187KB–13MB）、`doc\client` Javadoc、`doc\en\online` 用户手册、`doc\UserGuide_19.02.pdf`。
> 结论先行：**容器级/结构级解析已覆盖全部 21 个文件（21/21 可解析）**。
> 七项系统改进已全部落地（改进①~⑦，2026-08，均已提交 GitHub）：
> ①语义字典/全量建树/分层 ②网格抽取+STL ③嵌套子块关联 ④二进制数值流扫描
> ⑤版本指纹+ZIP 回退 ⑥状态表长度自校验 ⑦语义层报告。剩余缺口集中在字段级
> 语义解码（T 块文法、数组表字段、体网格）与回写能力。

## 0. 当前状态（v2 解析器实测）

| 能力 | 状态 |
| --- | --- |
| repr 字典头 / 分区遍历（2112~5047 分区） | ✅ 全 21 文件 |
| 状态表 ASCII 编码（80 列折行还原、记录文法） | ✅ 18 文件 |
| 状态表**二进制**编码变体（id=3B 大端、1B flags/version/尾值） | ✅ 3 文件（airfoil / manifold / vibratingPipe），数值流新增 2B 整 + 8B 双精度扫描（改进④），完整文法待解 |
| 数组块（Character1/Unsigned4/Integer4/Float8/**Integer8/Float4**） | ✅ 全部解码为 numpy；**面网格语义已解（改进②）**，其余字段级语义待解 |
| 对象图（2076~10395 对象，id=序号+2，Parent/Keys/NameManager 建树） | ✅ 全部；与官方 API 视图逐项一致（adjointWing 验证） |
| 语义字典/分层/别名表/全量建树（改进①） | ✅ semantic_dict.py：包→语义层、旧名→新名别名、属性引用方向；--layers/--aliases/--validate；游离对象 304→87 |
| 嵌套 TRANSMIT 子块识别（每文件 0~10 个） | ✅ 改进③：多 id 魔数 = 表内分段长度表（已按字节切分验证）；表内嵌 `CD-adapco_STAR-CCM+_ID<对象id>` 标记解析并关联到对象图（如 adjointWing id 15 → ConditionTypeManager）；独立 Character1 数组子块逐块解析 |
| ? 指针（对象 id / 浮点引用 / 嵌套 ?? 链） | ✅ 解析；浮点引用的含义未定 |
| 新格式字母/位图（L、J、TTT、B/P/???? 位图、+- 元素标记） | ✅ 捕获为记录；语义未解 |

## 1. 容器与格式变体（完整性）

已发现的变体与剩余工作：

1. **头部字段变体** ✅（改进⑤）：`--fingerprint` 输出版本指纹
   （banner 版本 ↔ StarVersion ↔ 状态表编码 ↔ 头部字段组合），21 文件指纹表已建
   （version_fingerprint.py）；发现二进制模式与版本无关（10.04/10.05 二进制、
   10.01 ASCII 并存），由内容/特性决定。
2. **魔数块变体**：外层 `CD-adapco_STAR-CCM+_ID`（标准）、**多 id 变体**
   （如 openWaterPropeller：`ID | 25 | 65607 | 65605 | ...`）、**无前缀变体**
   （methaneOnPt、AXMGeometry 直接以 `1 | 16260 | @ | T51...` 开头）。
   ✅ 多 id 语义已解（改进③）：N 后跟 N 个**表内分段长度**（按 banner+表体折行
   原文计字节，切分后尾差=0）；分段 0 = 主块，其余为记录流物理延续。
   表内嵌 `CD-adapco_STAR-CCM+_ID<对象id>` 标记 = 某对象的内嵌状态根，
   id 已关联对象图；分段 ↔ 对象级归属（哪个分段属于哪个对象）仍待细查。
3. **ZIP/压缩变体** ✅（改进⑤）：语料无 `PK` 容器；合成 ZIP 测试通过
   （zip 内单条目自动解包解析，2076 对象/36153 字符与直读一致）。真实压缩 .sim 待遇到再验。
4. **新数组类型** Integer8（methaneOnPt）、Float4（airfoil、directedMeshCAD 的
   MasterArray<Float4>）已支持；若遇 Unsigned8/复杂结构体需扩展 TYPE_SIZE。
5. **多 Character1 数组**（最多 11 个/文件）→ 嵌套 TRANSMIT 子块：已解析；
   **缺口**：子块与主表的关联（哪个数组属于哪个 Part/Scene）、子块内 `A<n>` 数组引用
   指向"文件级数组表"还是"子块私有数组表"尚未验证。
6. **大文件性能**：13MB 全解析 <1.2s（线性扫描），无性能缺口；但导出 100+ 数组时
   需流式写盘（当前一次性内存，可接受）。

## 2. STAR-CORE 状态表（深度）

1. **T/V/S/Z/J/L 块内部语义**（最大的黑盒）：每个格式字母后的数值流是结构化记录
   （面/边/顶点表、位图、指针），目前仅按"数值流"原样保留。
   → 已确证的：`?1` 记录 = 单个 face（18~31 个数值，含与 Float8 顶点表一致的坐标）、
   `+-` = 元素分隔标记、`A<n>` = 数组表引用、`?N`(int) = 对象图引用。
   未确证：`?<float>` 与 `??` 嵌套指针的目标（二进制状态区偏移？）、
   `T` 块开头 20+ 个数值（表头：计数/维度/容差？）、F/T/B/P 位图的含义。
2. **二进制编码状态表的数值流**（改进④部分完成）：id=3B 大端(=×256)/flags/version/
   尾值=1B 已解码；新增 `decode_binary_values()` 尽力切分 2 字节整 + 8 字节双精度
   交替段（已确认二进制流中确实存在 8 字节 LE 双精度，如 -0.1 = bf d9 99...）。
   **新发现**：用 starccmw.exe 重存二进制文件（airfoil/vibratingPipe，见 resave_sim.java）
   仍得到二进制编码 → 二进制模式**不是**老版本遗留，而是内容/特性决定（疑似与
   3D-CAD 几何数据相关）；重存后状态表与原件一致（写入器确定性）。
   **仍开放**：T/Q/G/V 块内 int/double 无类型标记的交替文法未完全还原（配对标例
   未能得到 ASCII 版本）。
3. **状态表尾部校验记录**（`S0 74 4 CI16 ... dCCZ ... 550 460 178 ...`）：含义
   （偏移/校验和）未定；可尝试修改文件重存验证。
4. ~~多 id 魔数与长度自校验~~ ✅（改进⑥）：`check_state_length()`/`--check-length`
   通用规则 "魔数值 = 块总长 - 魔数块长 - 1；多 id 时 sum = banner+表体长"，
   21 文件外层 16/16 通过（5 个无状态表/异形跳过）、嵌套子块 29/29 通过。

## 3. 数组表（深度）

- 已确证：Float8 n%3==0 的数组 = 顶点坐标表（adjointWing 数组 2 = 1412×3）。
- ~~网格抽取模块~~ ✅ 已完成（改进②）：`extract_mesh()`/`--mesh`/`--mesh-export`
  按"part.TriangleCount×3 匹配面索引数组 + 面索引最大值匹配顶点数组规模"的约束抽取
  点/面表，**19 个有网格元数据的文件中 18 个面数精确匹配主 part 且索引自洽**
  （1 个顶点数组为启发式、2 个无网格的纯几何文件正确返回空）；STL 导出验证通过
  （adjointWing：2824 面/1412 顶点）。
- **仍开放**：其余数组的字段级语义（法向、面积、单元/体表、边界映射、
  index_map/node_id_index_map/schema_embedding_map/child 所指的表）未解。
  语料中大量 `N, N, N*2` 式 Float8 三元组与 3 倍数之外的长度（如 cylinderBase 的
  732/744/744）表明除顶点外还有面数据/向量场表。
- 状态表记录（`index_map82 0 A17` 等）与数组块的对应关系可做自动标注：
  在 21 文件上交叉验证"每个 A<n> 引用的数组"与"网格规模自洽性"——待做。

## 4. 对象图（深度）

- 结构解析完备（1286 种 ClassName、63,245 个对象已全部解析）；缺**语义字典**：
  按包分类后（common 3530 / meshing 1113 / vis 784 / cadmodeler 457 / flow 411 /
  material 365 / energy 215 / motion 140 / turbulence 101 / post 57 / coremodule 48 ...），
  可自动归类"几何/CAD、网格、物理、场景可视化、后处理、求解器、自动化、协同仿真"。
  （完整包级语义目录见 doc_javadoc_catalog.md，已覆盖 574 个 star.* 包。）
- **官方机制命名确认**：对象图区 = 官方 Javadoc 中的 **DOM 序列化**
  （`star.common.dom.ObjectModel`：每个对象一个 DOM 元素），对象模型根 =
  `star.base.neo.ClientServerObject`（键 ClientServerObjectKey，属性走 NeoProperty）。
  这印证了"每行一个 repr 字典"的结构，并可指导将来的**回写**设计。
- **类名跨版本漂移**（重要）：本语料（6.x–10.x）含大量旧类名（XyPlot→Cartesian2DPlot、
  View→VisView、PolyhedralMesher→DualAutoMesher、PrismLayerMesher→PrismAutoMesher、
  SurfaceRemesher→ResurfacerAutoMesher...），解析器需要一张**旧名→新名别名表**
  （用 1286 个语料类名 × Javadoc 现名自动比对生成）。
- **缺口**：
  1. ~~属性级语义~~ → ✅ 已建 semantic_dict.py（DOWN/UP 属性方向白名单），全量建树后
     adjointWing 游离对象 304→87；剩余游离多为容器/子模型对象（MasterArray、
     SerializableSurfaceMesh、Export*Vector 等），可继续补充属性方向。
  2. 数值容器对象（MasterArray<Float8>/SerializableVector/Domain/InactiveList 等）与
     "文件级数组块"的对应关系未还原（对象图内的容器数据是否=数组块数据？）——仍开放。
  3. ~~ClassVersions 尾部统计~~ → ✅ --validate 输出诊断比对（类注册表快照：552 类/
     136 精确一致/计数和 1772 vs 图内 2075；语义为写入方运行时类注册表，含未序列化
     类，非严格校验和）。
  4. NameManager 为空对象——名字映射存于何处未解（可能即状态表内 ?N 指针区）——仍开放。

## 5. 语义层能力矩阵（"能导出什么"）

| 层 | 现状 | 差距 |
| --- | --- | --- |
| 文件元信息 | ✅ 版本/时间/对象数 | — |
| CAD 几何（cadmodeler 顶点/边/面/名称引用） | ⚠️ 对象已解析，坐标在状态表 T 块/数组 | 未能重建 B-Rep/几何拓扑 |
| 网格（顶点/面/体表） | ✅ 面网格已抽取+STL（§3） | 体网格（体单元表）待做 |
| Region/Boundary/Interface | ✅ 已关联网格表（改进⑦ --report：Region→Part→三角数） | 边界↔面片映射未做 |
| 场景/视图/Display/注记 | ✅ 已重建摘要（Scene→Displayer→视图，改进⑦） | 显示参数未解码 |
| 绘图/监视器/报告 | ⚠️ 对象已解析 | 未重建曲线数据 |
| 物理模型/材料/场函数 | ✅ 已按 PhysicsContinuum 归组（改进⑦ --report：22-23 模型/continuum） | 模型参数未解码 |
| 求解状态/解数据 | ⚠️ 官方文档确认 .sim 含解数据（restart file） | 未定位解字段快照的位置（可能在状态表 T 块/数组/嵌套子块） |
| 协同仿真链接（cosimulation 包） | ⚠️ 对象已解析 | 未提取链接配置 |
| 回写/修改 .sim | ⚠️ 对象图 repr 行可替换（`sim_writer.py`） | 不能插入新对象、不改数组/状态表；见 `star_gui_next.md` |

## 6. 官方文档支撑（已调研：用户指南 6909 页 + 发行说明；详见 doc_userguide_sim.md）

**文档确认（CONFIRMED）**：
1. .sim = 二进制、自包含（对象+状态+数据）、STAR-CCM+ 专属的 **restart file**；
   唯一关联文件，可被 OS 直接移动/复制/重命名。
2. 内含：几何、网格、物理、region/boundary、**解数据**、场景/绘图/窗口布局、
   Simulation Guide、Auto Save 设置、材料库路径；双精度使文件体积最多 +100%。
3. 保存体系：Save / Save As（自动加 .sim 后缀）/ Save All / Reload / 模板 .simt /
   自动保存副本 intake@N.sim / 备份 intake.sim~ / CHECKPOINT 触发文件。
4. **.simh = 独立的历史解文件（HDF5，可无损/有损压缩，含 states），.sim 内只存其相对路径**；
   压缩不涉 .sim 本身。
5. 并行 I/O（v6.06 起）不改格式；串/并行保存的文件相同；旧文件可恢复。

**文档未涉及（内部格式完全未公开）**：.sim 的块布局/二进制编码/序列化机制、
TRANSMIT（文档中仅指 Parasolid Transmit，与 .sim 内部节无关）、StarVersion、
loadedLibraries —— 即本项目的全部逆向工作无官方背书，需以语料实测为准。

**对本项目的启示**：
- "解数据在 .sim 内" → 状态表 T 块/数组块/嵌套子块中应有解字段快照，需继续定位。
- "6.06 修订过格式" → 解释了 6.x 时代二进制状态表 ↔ 7.x 起 ASCII 状态表的切换
  （语料中 binary 模式恰好都来自 modeller 270x/280x 的老文件）。
- .simh/HDF5 与 .sce/.scd5 是独立格式，不在本项目范围内（可列为后续扩展）。
- Javadoc（doc_javadoc_catalog.md）与用户指南互补：前者给出对象级语义字典
  （574 包），后者给出文件级行为定义；两者均未公开内部序列化格式。

## 6.5 查看器 / 编辑器交付（star_gui）

| 里程碑 | 状态 |
| --- | --- |
| M0–M6 查看器骨架 | 完成 |
| U1–U5 官方树 / 双行工具栏 / 视口 | 完成 |
| E0–E8 编辑器外壳 + 会话命令 + 对象行 Save | 完成（深度见 `star_gui_next.md`） |

> 查看器「打开-浏览-看图」约 65%；编辑器约 18%，可重开复现的主要是已有对象的名称/颜色等标量。
> 解析侧剩余缺口（T 块、数组字段、边界→面片、体网格）不变。下一阶段 **F0–F8** 见 `star_gui_next.md`。

## 7. 建议的下一步（按性价比排序）

GUI / 回写（优先，不依赖解完 T 块）：

1. **F0 加固**：Reload 脏提示、显隐走命令总线、会话改完刷新 3D。
2. **F1 已有对象落盘闭环**：颜色/透明/CurrentView/Keys → Save As → 重开一致。
3. **F2 对象图插入**：复制/新建场景可 Persist。
4. **F3 真 STL + 顶点数组回写**。

解析（可并行，不挡 F0–F2）：

5. 体网格单元表；边界 ↔ FaceTypes。
6. 二进制 T 块文法（同内容双格式配对）。
7. 容器对象 ↔ 文件级数组块；解场/.simh 定位。
