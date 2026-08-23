# star_gui 对标 STAR-CCM+ 20.02（除求解运算）

能力：`view` 已实现查看 · `session` 会话可改（重开丢失） · `persist` 可写回已有对象行 · `disabled` 灰显 · `needs_kernel` 需网格/CAD 内核 · `stub` 菜单在、不改真实数据

下一阶段见 `star_gui_next.md`（完整度复盘 + F0–F8）。本表按代码审计，不按菜单存在与否乐观估计。

## 菜单栏

| 条目 | 能力 |
| --- | --- |
| 文件>新建/打开/关闭/最近/退出 | view |
| 文件>重新加载 | session（脏提示与关闭/新建对齐） |
| 文件>保存/另存为 | persist |
| 文件>全部保存/自动保存/模板 | disabled |
| 文件>导入 CAD/表面/体网格 | persist（真读 STL/OBJ → ImportedVertices/Faces）；体网格 needs_kernel |
| 文件>导出 STL/摘要/报告 | view |
| 文件>宏 | disabled |
| 编辑>撤销/重做 | session |
| 编辑>复制/粘贴/删除/重命名 | persist（复制插入对象图行；删除改 Keys） |
| 编辑>上一选择/下一选择/按名称搜索 | session |
| 网格>生成表面/体网格 | needs_kernel（尝试 STAR 宏，否则禁用说明） |
| 网格>清除网格/缩放/诊断/转 2D | 诊断=view；缩放=session actor；其余 needs_kernel |
| 场景>（本客户端保留，官方在 Vis 工具栏） | view |
| 求解/连接 | disabled |
| 工具>诊断（指纹/长度/ClassVersions） | view |
| 工具>选项 | session |
| 窗口>树/属性/输出/绘图/工具栏 | view |
| 帮助>关于 | view |

## 工具栏

| 组 | 能力 |
| --- | --- |
| 系统：新建/打开/保存 | persist（保存） |
| 编辑：复制/粘贴/选择历史 | session |
| 网格生成：导入/修复/生成 | session / needs_kernel |
| 求解 | disabled |
| Vis：适配/视图/透明/网格开闭/派生零件 | view + session |
| 选择：框选 | view |
| 3D-CAD | session 外壳 |

## 树 / 3D 右键

| 条目 | 能力 |
| --- | --- |
| 显隐/仅显示/高亮 | session（勾选/仅显示走 CommandBus，可撤销） |
| 重命名/删除/复制 | session |
| 变换（平移/旋转/缩放） | persist（ImportedVertices 写回）；已有计算网格仍为 actor 预览 |
| 指定到区域 | session→persist（已有 PartGroup.Keys） |
| 3D：适配/视图/表示/网格开闭/复制图像 | view |
| 新建自动网格/布尔 | needs_kernel |

## 属性 / CAD / 绘图

| 条目 | 能力 |
| --- | --- |
| 类型化属性编辑 | persist（名称/颜色/透明/Keys/CurrentView） |
| 3D-CAD 模式外壳 | session |
| 绘图标签（监视器/报告数值） | view |
