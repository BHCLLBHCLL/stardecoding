# stardecoding — STAR-CCM+ .sim (TRANSMIT) 文件解析

解析 Siemens STAR-CCM+ 的 `.sim` 项目文件（内部称 "TRANSMIT FILE"）。纯 Python 标准库实现，
**不需要安装或启动 STAR-CCM+**；numpy 可选（用于数组导出）。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `sim_parser.py` | 主解析器（CLI + 可导入模块） |
| `README.md` | 本说明（格式逆向文档） |
| `function_gap_analysis.md` | **功能完整性与深度差距分析**（21 文件实测，按节推进改进） |
| `starccm_dump_tree.java` | 用官方 Java API 交叉验证对象图的宏（需要 license） |
| `adjointWing_start.sim` | 官方算例（本解析器的主要测试对象） |

> 已用 `D:\training\starccm\startutorialsdata` 全部 **21 个 .sim** 实测（6.03–10.05 版本，
> 187KB–13MB）：容器/分区/数组/对象图全部可解析；状态表含 ASCII 与二进制两种编码变体，
> 且存在嵌套 TRANSMIT 子块——详见 function_gap_analysis.md。

## 快速上手

```bat
python sim_parser.py adjointWing_start.sim --summary   :: 总体概览
python sim_parser.py adjointWing_start.sim --arrays    :: 数组表
python sim_parser.py adjointWing_start.sim --state     :: STAR-CORE 状态表记录流
python sim_parser.py adjointWing_start.sim --objects   :: 对象图
python sim_parser.py adjointWing_start.sim --tree      :: 对象树
python sim_parser.py adjointWing_start.sim --export out :: 导出 .npy/.csv/JSON
```

```python
# 作为模块使用
from sim_parser import SimFile
sim = SimFile("adjointWing_start.sim")
print(sim.summary())
verts = sim.array_data(2)      # 1412x3 Float8 顶点坐标 (numpy)
print(sim.object_by_id(2))     # <SimObject id=2 star.common.Simulation name='adjointWing_start'>
```

## .sim 文件结构（逆向结果）

`.sim` 是 modeller（STAR-CCM+ 前后处理，Python 编写）序列化的对象流，**线性**结构：

```
偏移 0      {'Binary': '\x07\x08\x02\x07', 'ClassName': 'STAR',
             'StatePosition': 221991, 'Version': 2}          ← 文件头（repr 字典）
            （空行）
            {'ClassName': 'Array', 'Type': 'Character1',
             'nElements': 36153, 'sizeof<T>': 1}              ← 字符数组描述
            + 36153 字节 ASCII 状态表（STAR-CORE 状态序列化）
            + 32 个类型化数组块（repr 描述行 + little-endian 二进制载荷）
偏移 N      {'ClassName': 'StarVersion', 'BuildArch': 'win64', ...}
            {'ClassName': 'loadedLibraries', ...}
            + modeller 对象图：每行一个 repr 字典（约 2000+ 对象）
```

- 头部/对象均为 **Python 2 风格 repr 字典**：字符串用单引号、整数带 `L` 后缀、布尔为
  Java 风格 `true/false`，可能含嵌套 dict/list/tuple。末尾一行是 `ClassVersions`
  （类名→数量统计，非引用）。
- `StatePosition` 指向 `StarVersion` 对象的绝对偏移；文件按偏移线性读取即可，无需跳转。

### 1) STAR-CORE 状态表（Character1 数组内容）

```
CD-adapco_STAR-CCM+_ID        ← 魔数块（4 行：ID / 1 / 36120 / @）
1                             ←  其中 36120 = 状态表长度 - 33
36120
@
T51 : TRANSMIT FILE created by modeller version 250020723 SCH_2500207_25001_1300
6189 0 12 29 CCCCCCCCCCCCCDI5 owner1040 0 CCCCCCCCCA16 index_map_offset0 0 1 dA9
 index_map82 0 A17 node_id_index_map82 0 A20 schema_embedding_map82 0 A5 child12
 0 A14 lowest_node_id0 0 1 dZ1 1132 2 3 0 0 ...
```

- 记录表按 **80 列折行**，折行会切断单词（甚至数字中间），换行还会**吞掉空格**。
  `sim_parser` 按逐字节推得的规则还原（见 `unwrap_table()` 的注释）：
  - 行尾字母 + 下一行以数字/`_` 开头 → 补空格（如 `"T"|"1"`、`"SDL/TYSA"|"_COLOUR79"`）
  - 行尾完整 `?数字` 指针 + 下一行以数字开头 → 补空格
  - 行尾数字 + 下一行以字母开头（且不是 `e±N` 续写）→ 补空格
  - 其余情况直接拼接（折行切词或空格保留在下一行行首）
- 记录文法（`<名字><id> <flags> [<version>] <格式字母><尾值> [数值流...]`）：
  - 名字后紧跟十进制 id（如 `owner1040` = 名字 `owner`、id `1040`）；
  - `flags`、可选 `version`（整数，后跟格式 token 时判定为 version）；
  - 格式字母与尾值**粘连**（如 `dA9` = 格式 `dA`、值 9）；
  - 随后的纯数值/`+-` 元素标记直到下一个非数值 token 为止，归入该记录的 values。
- 格式字母（经验表）：`C` 字符、`I` 整数、`D` 双精度、`d` 内联双精度、`A` 数组表引用、
  `u` 无符号、`l` 长整、`Z`/`V`/`S`/`T` 结构化标记（其后为数值流）、`?N` 对象图引用
  （N = 对象 id）、`F`/`T` 位图、`+-` 元素分隔标记。
- 典型记录（本算例）：
  - `6189 0 12 29` 表头（6189 ≈ 表中整数值个数）；
  - `index_map82 0 A17` → 数组表第 17 块；`node_id_index_map82 0 A20` → 第 20 块；
    `schema_embedding_map82 0 A5` → 第 5 块；`child12 0 A14` → 第 14 块；
  - `lowest_node_id0 0 1 dZ1 1132 2 3 ...` → 容差/索引参数（43 个数值）;
  - `T` 记录 + 长数值流 = 边界几何定义；`?1` 记录 = 单个 face（18~31 个数值，
    含顶点坐标，坐标与数组块 2 的 Float8 顶点表一致）；
  - 尾部 `S0 74 4 CI16 index_map_offset0 0 1 dCCZ20 14 3 0 0 550 460 178 ...` 为
    索引表重建/校验记录。

### 2) 数组块（32 块，little-endian）

| idx | 类型 | 数量 | 解读（本算例） |
| --- | --- | --- | --- |
| 1 | Unsigned4 | 1412 | 顶点序号表 |
| 2 | Float8 | 4236 | **1412×3 顶点坐标 (x,y,z)**（已与状态表数值交叉验证） |
| 5 | Unsigned4 | 2532 | 844×3 面顶点索引 |
| 6/7 | Float8 | 2532 | 面法向/面积等 |
| ... | ... | ... | 其余为索引映射、单元表、boundary 表等 |

### 3) modeller 对象图

- 每行一个对象 `{'ClassName': ..., attr: value, ...}`；对象显示名存在
  `PresentationName` 属性（顶层 Simulation 用 `name`）；
- **对象 id = 图中的序号 + 2**（Simulation=2、ManagerManager=3、NameManager=4...，
  由 `ManagerManager`/`NameManager`/`Parent` 引用自洽验证）；
- 引用关系：`Parent`（父对象）、`Keys`（管理器持有的子对象列表）、
  `NameManager`（对象的名字管理器）；其余整型属性（`Simulation`、`System`、
  `Dimensions`、`Units`...）是语义引用，方向不一，`--tree` 只按
  Parent/Keys/NameManager 建树，其余列为"游离对象"。

## 官方 API 交叉验证

本机装有带 license 的 STAR-CCM+（`starccmw.exe`）。已用官方 API 加载同一算例，
官方视图（Region=Fluid Domain、Scene=Mesh Scene 1、Continuum=Physics 1、
Plot=Residuals 及 9 个 Monitor）与本解析器对象图逐项一致。可自行运行：

```bat
starccmw.exe -batch starccm_dump_tree.java adjointWing_start.sim
```

## 已知边界

- 本实现以 `startutorialsdata` 全部 21 个 .sim（TRANSMIT 格式 v2，2011–2015 保存，
  含 ASCII/二进制两种状态表编码、嵌套 TRANSMIT 子块、多 id 魔数等变体）实测，
  **21/21 可解析**。
- STAR-CORE 状态表部分格式字母的完整语义（`T`/`V`/`S`/`Z` 块内部结构、
  `?1` face 记录各字段含义、二进制变体的数值流文法）仍为经验结论——见 function_gap_analysis.md。
- 若文件以 `PK` 开头（ZIP 容器），解析器会自动解包取主条目后按相同格式解析。
- 中文输出依赖 UTF-8 终端（脚本已自动 reconfigure）。
