# STAR-CCM+ 官方文档中关于 .sim 文件的所有说明（调研摘要）

> 调研范围：HTML 用户指南（Simcenter STAR-CCM+ 2402, D:/training/starccm/doc/en/online/STARCCMP/, 6909 个页面，通过 WebHelp 全文搜索索引定位）、ReleaseNotes_19.02.012_01.html、InstallationGuide_19.02.012_01.html。
> PDF（UserGuide_19.02.pdf, 159 MB）**无法用本环境工具搜索**（无 pdftotext/mutool；read 工具只能读 UTF-8 文本），本报告完全基于 HTML 版指南 + 发行说明。
> 所用源文件均在 D:/training/starccm/doc/en/online/STARCCMP/ 下，下文只列 GUID 文件名。

---

## 一、.sim 文件是什么（核心定义）

**来源：GUID-FAADED77-5FC2-4B37-BFDA-8AB834B3D10C.html — “Saving Simulations to File”**（最关键的一页）
- 原文：“A simulation file contains the complete set of simulation objects, state, and data for the problem.” —— **.sim 文件包含该问题的完整仿真对象（simulation objects）、状态（state）和数据（data）**。
- “It is the restart file for Simcenter STAR-CCM+ and is stored with the .sim extension.” —— **.sim 就是重启文件（restart file）**。
- “Since it is a binary file, applications other than Simcenter STAR-CCM+ cannot read it.” —— **二进制格式，除 STAR-CCM+ 外任何应用都无法读取**。
- “The file can safely be moved, copied, and renamed using operating system commands as it is the only file that is associated with a simulation.” —— **它是与仿真关联的唯一文件，可用操作系统命令安全地移动/复制/重命名（可移植性声明）**。
- 保存时的行为：覆盖保存或另存时，STAR-CCM+ 会卸载过时的表面网格数据（unloads any outdated surface mesh data），保留每个 part 的最新表面网格。

## 二、保存 / 另存为 / 自动保存 的行为

| 操作 | 说明 | 来源 |
|---|---|---|
| **Save** | 把改动写回磁盘上已有的 .sim；若仿真从未保存过则转为 Save As；只有存在未保存改动时才可用（Ctrl+S）。**打开一个 scene 也算改动**。 | GUID-2C7B9FD5… (Saving Over an Existing Simulation File)；GUID-99CC09ED… (Menu Reference) |
| **Save As…** | 保存到新文件；对话框默认只列 *.sim；**扩展名自动追加**：fred→fred.sim，fred.dat→fred.dat.sim，fred.sim→fred.sim；原文件保留（相当于做一份拷贝）。 | GUID-B4F51A33… (Saving a Copy of a Simulation) |
| **Save All** | 把所有有改动的打开仿真一次性保存。 | GUID-342E5504… (Saving Multiple Simulations) |
| **Reload Simulation** | 放弃未保存改动，回到“最后保存状态”（last saved state）。 | GUID-46D2ED86… (Reverting Unsaved Changes to the Simulation) |
| **Save to Template** | 存成模板仿真文件，扩展名 **.simt**（可包含无网格的 mesh pipeline、physics continua、regions/boundaries、reports/scenes/plots 等）。 | GUID-2D0CAF6E… (Template Simulation Files)；GUID-99CC09ED… |
| **Auto Save…** | 按间隔自动保存；设置项：“**The settings are saved in the simulation database**”（设置保存在仿真数据库中）。 | GUID-A92F332F… (Automatically Saving at Regular Intervals) |

**Auto Save 属性（GUID-E72E56F4… “Auto Save Properties” / GUID-A92F332F…）**：
- Max Autosaved Files：自动保存副本数，默认 0 = 总是覆盖同一文件（等同 Save）；达到上限后删除最旧的副本。
- Autosave Batch Runs / Autosave After Volume Mesh / Autosave After Next Step。
- **文件名**：副本与仿真同名 + 分隔符（默认 **@**）+ 序号，例如 intake@1.sim；Separator 可改，Format Width 控制序号宽度。
- Trigger File：默认名为 **CHECKPOINT** 的触发文件，每 60 秒检查一次，找到后在该步（迭代或时间步）结束时自动保存，之后把触发文件改名为 CHECKPOINT~。→ “Saving with a Checkpoint File”（GUID-E7C71BDC…）。

**备份文件（GUID-791B4549… “Using Backup Files”）**：修改并保存已有仿真时，自动生成上一版本的副本，文件名形如 intake.sim~（扩展名后加波浪号 ~），用于撤销已保存的改动；客户端默认不识别该文件以防误开。

**磁盘满 / 异常（GUID-B2FEB1DF… “What Happens If a Disk Is Full?”）**：磁盘满时保存失败并提示；发行说明还指出某些 NFS 下可能写出 0 字节的 .sim（无报错）。

## 三、.sim 文件里存了什么（文档明确/间接确认）

- **完整对象树**：Geometry parts、mesh、physics continua、regions/boundaries、solution、reports、scenes、plots 等（核心定义页；教程“Loading the Starting Simulation File” GUID-ECECA1B5… 展示 .sim 内含 geometry + mesh operations + continua + regions + simulation operations）。
- **Solution/state**：状态与解数据是保存内容的一部分（见核心定义；Reload 回到 “last saved state”）。
- **窗口布局**：Tool > Options > Visualization > Save layout on sim save 选项把布局随 .sim 保存（GUID-A712BB68… “Preserving the Window Layout”）。
- **Simulation Guide**：可编辑的内嵌文档（格式化文本/表格/图片/超链接）随仿真保存（GUID-4216C87B… “Simulation Guide”）。
- **材料数据库路径**：发行说明 FDF-2632 称“prevent the simulation file from retaining the material database path”——.sim 会记住材料数据库路径。
- **精度影响体积**：双精度版 .sim 比混合精度版大**不超过 100%**（取决于浮点字段占比）——说明 .sim 内存有大量浮点字段（GUID-4DB645E0… “Mixed or Double Precision”）。
- 保存/加载通过服务端完成；可选 **Parallel I/O**（仅 Linux、仅 .sim）：串行/并行 I/O 保存的文件**格式完全相同**，可互相读取；“the format was revised for version 6.06”以支持并行 I/O，旧文件均可恢复（GUID-7B35D443… “Using Parallel I/O”）。

## 四、.simh（Solution History）—— 有明确格式说明的兄弟文件

**来源：GUID-0E3DFBD6-6A84-444B-87B8-A3E6A52ED064.html — “Solution History Reference”**；GUID-1435423A… / GUID-590177D5… / GUID-F0FD2C57…（“Setting Up the Solution History File”）
- .simh = 仿真历史文件（simulation history file），在 **Solution Histories** 节点下创建，用于按指定时间间隔写入所选解数据；是独立于 .sim 的**单独文件**，.sim 内记录其**相对路径**（Path 属性；新建未保存的仿真会存绝对路径）。
- **压缩模式 Compression mode：文件格式为 .hdf5（Hierarchical Data Format / HDF）**，选项 Off / Lossless（无损）/ Lossy（有损，Preserved Digits 1–8 控制保留有效位数）；当前版本仅表面/体积数据（网格信息与场数据）被压缩，顶点坐标不被压缩。
- **State 属性：显示 .simh 中已保存状态的个数**；“states”概念 = 文件里保存的时间/解状态；Auto-Rescan 使表示与文件内 states 同步。
- Auto-record 自动记录；Functions/Inputs/Regions 选择要存的数据；Create Snapshot 手动把当前解快照加入 .simh；Clear 会删除并重建文件。
- 教程页还说明：**保存 .sim 时也保存 .simh 文件**，并创建 recorded solution view / representation。
- 发行说明：**.trn 文件已弃用，改用 Solution Histories（.simh）**。

## 五、相关格式对比（文档原文）

- **.ccmt / .ccmp / .ccmg —— 是 STAR-CD 的 “CCM” 格式，不是 STAR-CCM+ 自有格式**（GUID-1FBA89BE… “Importing STAR-CD Data”；GUID-47B4AB5B… “Importing Volume Meshes”）：
  - .ccmg = pro-STAR 的几何/网格数据；.ccmp = STAR 求解器的 restart/post 解数据；.ccmt = STAR 的瞬态 post 数据。二者都链接到 .ccmg。
  - 导入 .ccmp 时只有第一个 state 可用；把 .ccmt 改名成 .ccmp 可读第一个时间步。
  - 输出示例中出现 “CCM file version 20617” —— 文件头带版本号。
- **.cas/.dat（Fluent）**：STAR-CCM+ 只能导入 Fluent **网格**（不导入 BC/模型/解），读取其若干 section（FLUENT_HEADER_SECTION=1、NODE=10、CELL=12、FACE=13、ZONE=39 等），并将 zone 类型映射到 STAR-CCM+ 边界类型（GUID-28403E55… “Migrating from Fluent”）。
- **.simt**：模板仿真文件（见上）。
- **.sce**：Simcenter STAR-CCM+ Viewer 文件，用于场景/绘图导出与 Teamcenter Share 共享（GUID-8A82BCDE… / GUID-582D3A47…）。
- **.scd5**：Simcenter Data File 导出格式（HDF5 相关，GUID-D82BE280…）。
- **表文件内部格式**：指南里叫 “File Format Reference” 的页面（GUID-840A16C7…）只讲**表数据**的 xyz/radial 内部格式（{'Type':'xyz', 'DataSets':[...], 'X':[...]} 文本结构）、.csv/.tab/.xy——与 .sim 无关。

## 六、关于内部结构的搜索结论（TRANSMIT / StarVersion / loadedLibraries / serialization）

- **TRANSMIT**：全文索引中唯一命中是 “Parasolid Transmit”（Parasolid 几何交换格式，GUID-7DC1507B…）——**与 .sim 内部结构无关，文档从未把 TRANSMIT 描述为 .sim 内的区块**。
- **StarVersion / loadedLibraries / serializ / persist**：在 2402 指南全文索引中 **0 命中**（均不在索引中）。
- 文档对 .sim 内部结构（区块/分块布局、序列化机制、头信息、对象编码）**没有任何描述**；只给出三个外围事实：二进制；串/并行 I/O 格式相同且 6.06 起修订过格式；双精度使体积最多翻倍。→ **内部格式未公开文档化（符合预期）**。

## 七、跨版本/可移植性（文档明确说的）

- .sim 可被 OS 命令移动/复制/重命名，且是唯一关联文件（见第一节）。
- 并行 I/O 页：旧版（6.06 之前）保存的 .sim 也能被新版恢复（backward compatibility）。
- 客户端与服务端版本必须匹配才能连接（GUID-8ADD069B… “Client-Server Connections”）；未找到 “.sim 可被任意新版打开” 的笼统保证。
- 发行说明（19.02.012）补充：修了“带 CGNS 外部链接保存后再恢复”的错误（CAEI-6605）；材料数据库路径会残留在 .sim 中（FDF-2632）；.trn 弃用改 .simh。

## 八、文档 CONFIRM vs 未文档化

**文档已确认（CONFIRMED）**：
1. .sim 是二进制、自包含（对象+状态+数据）、STAR-CCM+ 专属、是 restart file。
2. 保存/另存/保存全部/自动保存/保存为模板/备份 ~ 文件/CHECKPOINT 触发文件的行为与文件命名规则。
3. 内含：几何、网格、物理、region/boundary、解数据、场景/绘图/布局、Simulation Guide、Auto Save 设置、材料库路径；双精度使文件体积最多 +100%。
4. .simh 是独立文件，HDF5，可无损/有损压缩，含“states”；.sim 里存相对路径。
5. 并行 I/O 不影响格式；格式在 6.06 修订过；旧文件可恢复。
6. .ccmt/.ccmp/.ccmg 是 STAR-CD CCM 格式；.cas 导入仅网格。

**文档未涉及（NOT documented —— 内部格式）**：
- .sim 的内部结构、chunk/块布局、二进制编码、序列化/持久化机制——**完全没有**。
- TRANSMIT（作为 .sim 内部节）、StarVersion、loadedLibraries 等内部标识符——**完全没有**。
- .sim 自身的压缩选项/压缩算法——**没有**（压缩仅存在于 .simh 的 HDF5）。
- 官方未声明 .sim 的“文件格式规范/版本号清单”，仅旁证“格式在 6.06 修订”与 CCM 格式的 “file version 20617” 示例。

---

*调研方法备注：6909 个 HTML 页无法逐页 grep（超时），改用 WebHelp 自带全文搜索索引（oxygen-webhelp/app/search/index/index-1/2/3.js + htmlFileInfoList.js）以 Node 程序按词查询定位页面，再对目标页做 HTML→文本提取核读；发行说明/安装指南直接全文检索。*