# STAR-CCM+ Java API 语义目录（Semantic Catalog of .sim Object Graph Classes）

> 来源：STAR-CCM+ 官方 Java API Javadoc（D:/training/starccm/doc/client/html，共 574 个 star.* 包）。
> 用途：为 .sim 文件解析器提供 ClassName → 语义 映射参考。.sim 中每个对象是 star.base.neo.ClientServerObject 的实例，ClassName 即其 Java 类全名（如 star.common.Simulation、star.vis.Scene）。
> 版本差异（重要）：本版文档不存在 star.solvermodels 包（求解模型分散在 star.flow、star.energy、star.segregatedflow、star.coupledflow 等物理包中）；不存在 XyPlot / View / PolyhedralMesher 等旧类名（见各节“命名差异”）。

---

## 一、包总览（按 .sim 对象图角色分组，各一行）

| 包 | 一句话作用 |
|---|---|
| star.base.neo | 对象模型内核：ClientServerObject、对象键、管理器基类、属性（NeoProperty）、序列化枚举、日志 |
| star.base.report | 报告/监视器体系：Report、Monitor、ReportManager、MonitorManager、场均值/积分等派生数据 |
| star.base.query | 查询/谓词体系：Query、Predicate（按名称/属性/注释筛选对象的动态查询） |
| star.base.generic | 通用对象：GenericObject / GenericObjectManager（用户自建通用对象） |
| star.common | 仿真主干：Simulation、Region/Boundary/Interface/Continuum、Part 体系、单位/量纲、场函数、求解器/模型管理器、表格、绘图 |
| star.common.dom | 对象模型与 DOM 的映射（ObjectModel），是理解 .sim 序列化格式的关键 |
| star.vis | 可视化：Scene、Displayer（Part/Scalar/Vector/Glyph）、VisView、Light、LookupTable、ColorMap、切面/等值面/流线、硬拷贝 |
| star.vis.dom | 可视化 DOM 写入器（ColorMapWriter 等） |
| star.post | 解历史/后处理：SolutionHistory、SolutionRepresentation、SolutionViewManager、解动画 |
| star.meshing | 几何与网格主包：Part（CadPart/SimpleBlockPart/MeshPart）、MeshOperation、AutoMesher、接触、印刻、布尔运算、网格管线 |
| star.meshing.geometryrepair | 几何修复工具（GeometryRepairWidget 系列） |
| star.trimmer | 切割体网格器（TrimmerAutoMesher）及尺寸/生长率设置 |
| star.prismmesher | 棱柱层网格器（PrismAutoMesher）与层参数 |
| star.delaunaymesher | Delaunay 体网格器（DelaunayAutoMesher） |
| star.dualmesher | 多面体/四面体网格器（DualAutoMesher，本版替代 Polyhedral/Tetrahedral Mesher） |
| star.solidmesher | 薄体网格器（ThinAutoMesher / ThinSolidAutoMesher） |
| star.sweptmesher | 扫掠/定向网格器（DirectedMesher、DirectedMeshOperation） |
| star.extruder | 拉伸网格（ExtruderMesher、Surface/VolumeExtruderOperation） |
| star.twodmesher | 2D 网格（AutoMesher2d、DualAutoMesher2d 等） |
| star.bodyfittedmesher | 贴体网格器（AdvancingLayerAutoMesher） |
| star.surfacewrapper | 面包裹（Wrapper）设置：间隙闭合、泄漏检测、接触防止、偏置厚度 |
| star.resurfacer | 表面重构（ResurfacerAutoMesher / AutomaticSurfaceRepairAutoMesher） |
| star.cadmodeler | 3D-CAD 建模器：CadModel/Body/Feature/Sketch/约束，CAD 特征树 |
| star.material | 材料体系：Material、Mixture、Gas/Liquid/Solid、MaterialProperty + PropertyMethod |
| star.flow | 流动物理：重力、参考压力/密度、进出口剖面、力/力矩报告、多孔介质、风扇曲线、用户源项 |
| star.energy | 能量物理：温度、热通量/换热系数、多部件/多层热容、换热器、压力降报告 |
| star.turbulence | 湍流模型基类：RANS/LES/DES、湍动能/耗散剖面与壁函数 |
| star.segregatedflow | 分离求解器：SegregatedFlowModel/Solver、压力/速度求解器、声学 CFL 限制 |
| star.coupledflow | 耦合求解器：CoupledFlowModel/Solver、Courant 数控制、收敛加速、伴随流 |
| star.lagrangian | 拉格朗日相：LagrangianPhase、Injector、粒子模型、尺寸分布、阻力/传热模型 |
| star.lagrangian.spray | 喷雾：雾化、破碎、蒸发、液膜撞击（子包） |
| star.lagrangian.tracks | 粒子轨迹文件（TrackFileModel、BoundaryTrackSamples） |
| star.lagrangian.dem | DEM 离散元（碰撞/接触，子包） |
| star.multiphase | 欧拉多相/相间作用：EulerianPhase、PhaseInteraction、曳力/升力/虚拟质量/壁面润滑、聚并/破碎/成核 |
| star.vof | VOF 自由表面：VofWave 系列、表面张力、锐化格式（HRIC） |
| star.eulerianmultiphasemasstransfer | 欧拉多相传质：沸腾/蒸发/冷凝、气泡成核、相间传热传质 |
| star.mixturemultiphase | 混合物多相（MMP）模型 |
| star.dmp | 离散多相（DispersedMultiphase，气泡/液滴相） |
| star.radiation.common | 辐射：灰体/多带/S2S/PMC/球谐模型、发射率/反射率剖面、太阳载荷 |
| star.species | 组分输运：质量/摩尔分数剖面、组分源项、热扩散 |
| star.fea.common.models | 结构有限元：时间积分方法、矩阵更新策略 |
| star.motion | 运动/参考系：Motion、SixDofMotion、TrajectoryMotion、MorphingMotion、旋转/平移规范 |
| star.sixdof | DFBI 六自由度体：Body、SixDofBodyMotion、约束/耦合、外力（重力/阻尼/弹簧） |
| star.cosimulation（api/link/common） | 协同仿真：CoSimulation、CoSimulationType（Abaqus/Amesim/FMI/GT-POWER/Nastran/RBA…）、耦合区与场映射 |
| star.coremodule | 客户端内核/UI：Application、Session、StarBeanNode、ActionRegistry |
| star.coremodule.ui.layout | 界面布局：Layout、LayoutObject、Mode（场景/绘图窗口布局，.sim 中可保存） |
| star.automation | 仿真自动化：AutomationBlock/Chain/Workflow、SimDriverWorkflow、循环/条件块 |
| star.assistant | 仿真助理：SimulationAssistant、Task、Condition/ConditionTrigger |
| star.solvermeshing | 求解器网格：GeometryRefinementModel（几何细化） |
| star.combustion | 燃烧：CombustionModel、火焰类型（预混/非预混/PPDF）、层流火焰速度/厚度、点火器 |
| star.battery | 电池：BatteryModel/Cell/Module/Pack、电路、电化学（子包） |
| star.casting | 铸造：CastingModel、凝固时间/枝晶间距、Niyama 判据 |
| star.overset | 重叠网格：Overset 守恒模型、自适应棱柱层 |
| star.morpher | 网格变形：MeshDeformationModel、位移规范、设计点 |
| star.stabilization | 求解稳定化：FsiAddedMassParameters、Anderson 参数 |
| star.mdx | 设计管理器：DesignSet、参数/响应、优化研究（.sim 中保存设计表） |

---

## 二、重点包详解

### 1. star.base.neo —— 对象模型内核（.sim 一切的根基）
- ClientServerObject：所有服务端对象的客户端代理基类；.sim 对象图里每个对象都是它（或其子类）的实例
- ClientServerObjectKey / ClientServerObjectManager：对象的唯一引用键 / 对象构造管理器
- NamedObject、NamedClientServerObjectManager：带名字的对象与命名管理器
- Instantiator：实例化 STAR-CCM+ 对象（解析 .sim 时按 ClassName 创建对象的入口）
- NeoProperty / NeoPropertyArg：属性系统（对象属性→值/引用/数组的传输载体）
- ObjectRegistry / GroupableObjectManager / NeoObjectManager：对象注册表与分组管理
- ClientServerObjectGroup：对象引用组（如边界组）
- SerializableEnum：可序列化枚举（枚举选项属性）
- VersionInfo：版本比较；NeoJournal：宏日志系统
- Plane：几何平面（原点+法向，切面等的基础）

### 2. star.common —— 仿真主干（.sim 顶层节点几乎全在这里）
- Simulation：根对象（Server simulation proxy），拥有全部管理器
- SimulationIterator：迭代/时间步控制（迭代、物理时间）
- Region / RegionManager：区域（流体/固体/多孔/粒子/液膜区域）
- Boundary / BoundaryManager：边界；BoundaryInterface：边界-界面关联
- Interface / InterfaceManager：界面（内部界面/接触界面/周期界面）
- Continuum / PhysicsContinuum / ContinuumManager：连续体（物理连续体持有模型/初始条件/参考值）
- Part / GeometryPart / GeometryPartManager / PartManager：几何部件与管理器
- PartSurface / PartCurve / PartPoint：部件表面/曲线/点
- PartContact / PartContactManager：部件接触；PartGroup / PartGrouping：部件组/分组
- Units / UnitsManager / Dimensions：单位与量纲（每个带单位数值属性都挂 Dimensions + Units）
- FieldFunction / FieldFunctionManager / PrimitiveFieldFunction / UserFieldFunction：场函数
- Table / TableManager / InternalTable / FileTable；DataSource / DataSourceManager：数据表与数据源
- GlobalParameterManager / ScalarGlobalParameter / VectorGlobalParameter：全局参数（输入参数）
- Solution / SolutionView；Solver / SolverManager：解与求解器
- Model / ModelManager / PhysicsModel / MethodManager：模型与方法（物理连续体的 Model 子节点）
- SolverStoppingCriterion：停止准则（MonitorIterationStoppingCriterion、PhysicalTimeStoppingCriterion）
- ReferenceFrame / LabCoordinateSystem / LocalCoordinateSystem / CoordinateSystemManager：参考系/坐标系
- InitialConditionManager / ContinuumConditionManager / RegionInitialConditionManager：初始/边界条件管理器
- MeshManager / MeshImporter / MeshPipelineSolver：网格工具代理
- PartitioningModel / PartitioningSolver / PartitionConfigManager：并行分区
- ParticleTracks / ParticleTracksManager：粒子轨迹
- AutoSave / AutoExport：自动保存/导出设置
- 绘图（本版在 star.common）：PlotManager、MonitorPlot、ResidualPlot、Cartesian2DPlot（原 XyPlot）、Cartesian3DPlot、GeneralizedPlot、AnalysisPlot、HistogramPlot、PiePlot
- 命名差异：无 XyPlot / View 类；XyPlot→Cartesian2DPlot；视图类在 star.vis（VisView）

### 3. star.common.dom —— 对象模型 ↔ DOM（.sim 格式的关键）
- ObjectModel：将 ClientServerObject（含属性与依赖）映射到 DOM 元素的稀疏数据结构 —— .sim 序列化/反序列化核心
- 一系列 *Writer（DoubleWriter、DimensionsWriter、ProfileWriter、PolynomialInputWriter 等）：各类属性值的 DOM 写入器

### 4. star.vis —— 可视化（场景/显示器/视图）
- Scene / SceneManager：场景及其管理器
- Displayer / DisplayerManager：显示器基类与管理器
- PartDisplayer；BoundaryActor / PartActor / RegionActor / GlyphActor：场景演员
- ScalarDisplayer / ScalarDisplayQuantity、VectorDisplayer / VectorDisplayQuantity：标量/矢量显示
- VisView / ViewManager / ViewAngle：视图（相机）与管理器（旧 View 类并入 VisView）
- Light / LightManager：光源
- LookupTable / LookupTableManager / ColorMap / ColorPalette / UserLookupTable / UserColorPalette：色标与配色
- Legend / TextAnnotation / LogoAnnotation / SceneAnnotation / ReportAnnotation / AnnotationManager：图例与标注
- ClipPlane / PlaneManager、PlaneSection、IsoPart / IsoCreator / IsoValue、ThresholdPart / ThresholdCreator：切面/等值面/阈值
- StreamlineCreator、ScalarWarpSurface：流线/变形面
- PartDataSource / BoundaryDataSource / ThresholdDataSource / IsoDataSource / GridDataDataSource：场景数据源
- VolumeRenderingSettings、RayTrace*Material（材质）、SceneExportSettings：体渲染/光线追踪/导出
- VisualizationSolver：可视化求解器
- 命名差异：无 SurfaceScene/ScalarScene 类；场景 = Scene + Displayer + VisView

### 5. star.post —— 解历史/后处理
- SolutionHistory / SolutionHistoryManager：按用户触发或周期性记录的解数据快照
- SolutionRepresentation：解表示（CurrentSolutionView、RecordedSolutionView）
- SolutionViewManager、SolutionAnimation / SolutionAnimationSettings / SolutionAnimationDirector：解视图与动画
- FvRecordedObject 系列（FvRecordedPart/Surface/Volume/PointCloud/ParcelCloud）：记录数据的有限体积对象

### 6. star.base.report —— 报告与监视器
- Report / ReportManager：报告基类与管理器；Monitor / MonitorManager：监视器基类与管理器
- ReportMonitor：报告转监视器；ScalarMonitor：标量监视器
- FieldMeanMonitor / FieldMaxMonitor / FieldMinMonitor / FieldRMSMonitor / FieldSumMonitor / FieldVarianceMonitor / FieldCoVarianceMonitor：场统计监视器
- AreaAverageReport / SurfaceIntegralReport / VolumeAverageReport / VolumeIntegralReport / SumReport / ExpressionReport / StatisticsReport：常用报告
- MonitorData / MonitorDataSet / TableData：监视器采样数据/数据集
- DerivedData / DerivedDataManager：派生数据（统计/滤波/FFT）
- ReportValue / ReportValueManager、ReportAnnotation3D：报告值显示
- 注：力/力矩/质量流量报告在 star.flow（ForceReport、MomentReport、MassFlowReport）

### 7. star.base.query —— 查询/动态选择
- AbstractQuery / DynamicQuery：可求值的查询
- Predicate 体系：NamePredicate、CommentPredicate、CsoPropertyPredicate、CompoundPredicate、IdentityPredicate 等
- CsoSet：有序对象集合；ManagerWithQuery：带查询的管理器

### 8. star.meshing —— 几何与网格主包
- CadPart：CAD 部件（导入的 CAD 几何，.sim 常见）；ExtractedPart：抽取部件
- MeshPart / LeafMeshPart / MeshPartDescription：网格部件及其描述
- SimpleBlockPart / SimpleCylinderPart / SimpleSpherePart / SimpleConePart / SimpleShapePart：基本体素部件
- MeshOperation / MeshOperationManager：网格操作（管线节点）
- AutoMeshOperation / AutoMeshOperationShell / AutoMesher / AutoMesherManager：自动网格操作
- BooleanPartsOperation / SubtractPartsOperation / IntersectPartsOperation / CombinePartsOperation：布尔运算
- TransformPartsOperation / TransformControl：变换操作
- ImprintPartsOperation：印刻操作；ContactOperation / ContactObject：接触操作
- FillHolesOperation、PrepareFor2dOperation、SurfaceMorphOperation：补洞/2D 准备/面变形
- PartCustomMeshControl / SurfaceCustomMeshControl / CurveCustomMeshControl：自定义网格控制
- SurfaceSize / SurfaceSizeMethod / SurfaceProximity / SurfaceCurvature：表面尺寸/邻近/曲率
- MeshPipelineSolver / MeshPipelineController：网格管线（部件→网格操作→体网格）
- MeshPartImporter / PartImportManager：部件导入
- 命名差异：无 SurfaceWrapper/WrappingPart 类（面包裹在 star.surfacewrapper 与 LocalWrappingExtent）

### 9. 网格器子包（网格管线中的 Mesher 节点）
- star.trimmer：TrimmerAutoMesher（切割体）+ 生长率/各向异性尺寸/小单元截止
- star.prismmesher：PrismAutoMesher + NumPrismLayers/PrismLayerStretching/厚度比/自定义棱柱
- star.delaunaymesher：DelaunayAutoMesher
- star.dualmesher：DualAutoMesher（本版的多面体/四面体网格器）
- star.solidmesher：ThinAutoMesher / ThinSolidAutoMesher（薄体网格）
- star.sweptmesher：DirectedMesher / DirectedMeshOperation + 源/目标面/层数分布
- star.extruder：ExtruderMesher、SurfaceExtruderOperation / VolumeExtruderOperation
- star.twodmesher：AutoMesher2d、DualAutoMesher2d、QuadAutoMesher2d
- star.bodyfittedmesher：AdvancingLayerAutoMesher（贴体棱柱）
- star.surfacewrapper：面包裹设置（GapClosure*、LeakDetector*、PartialWrapSet、PartsContactPreventionSet）
- star.resurfacer：ResurfacerAutoMesher / AutomaticSurfaceRepairAutoMesher（表面重构）
- star.meshing.geometryrepair：GeometryRepairWidget*（修复 GUI，多为客户端对象）

### 10. star.cadmodeler —— 3D-CAD 建模器（CAD 特征树）
- CadModel / CadObject / CadObjectKey / CadObjectManager / CadObjectRepository：CAD 模型与对象仓库
- Body / BodyManager / BodyGroup：体与体管理
- Feature / FeatureManager：特征基类与特征管理器
- Sketch / Sketch3D / SketchPlane / SketchPrimitive* / SketchController：草图与草图基元
- BodyFeature 系列：Extrude/Revolve/Sweep/Loft/Fillet/Chamfer/ShellFeature/ThickenSheetBodies/MirrorBodyFeature/Pattern*
- 参考几何：ReferencePlane* / ReferenceAxis* / ReferencePoint* / ReferenceCoordinateSystem*
- SolidModelPart / SolidModelCompositePart / SolidModelManager：CAD 部件到几何部件的桥接
- ImportCadFileFeature / ImportedCurvesCreator：CAD 导入
- Constraint / ConstraintManager、CadQuery / CadPredicate：约束与 CAD 查询

### 11. star.material —— 材料
- Material / MaterialDataBase / MaterialDataBaseManager：材料与材料库
- Mixture / Gas / Liquid / Solid / GasMixture / LiquidMixture / LiquidSolidGasMixture：材料类型
- MaterialProperty / MaterialPropertyManager / MaterialPropertyMethod：材料属性与方法（Polynomial/Table）
- MolecularWeightProperty、PhaseInteractionMaterial：分子量、相间材料

### 12. 流动/求解物理包
- star.flow：Gravity/GravityModel、ReferencePressure/ReferenceDensity/ReferenceAltitude、PressureReferencePoint、ForceReport/MomentReport/MassFlowReport/ForceCoefficientReport/MomentCoefficientReport/ThrustReport、PorousViscousResistance/PorousInertialResistance、FanCurveSpecification、各 Profile（VelocityProfile/TotalPressureProfile 等）、用户源项（MassUserSource/MomentumUserSource）
- star.segregatedflow：SegregatedFlowModel/Solver、PressureSolver、VelocitySolver、ContinuityInitializer、LimitAcousticCFL*（本版无 star.solvermodels，分离求解器在此包）
- star.coupledflow：CoupledFlowModel/Solver、CoupledImplicitSolver/CoupledExplicitSolver、CourantNumberControl/ConstantCourantNumberControl/AutomaticCourantNumberControl、ConvergenceAccelerator、AdjointFlowModel、CoupledEnergyModel/CoupledSolidEnergyModel
- star.energy：Energy 物理与温度相关类、HeatTransferReport、PressureDropReport、HeatExchanger*（换热器）、ConstantSpecificHeat/PolynomialSpecificHeat、ThermalConductivityProperty、TemperatureProfile 系列
- star.turbulence：TurbulentModel/RansTurbulenceModel/LesTurbulenceModel/DesTurbulenceModel、TurbSolver/TurbViscositySolver、壁函数（WallFunctionCondition/StdWallFunctionCondition/BlendedWallFunctionCondition）、湍动能/耗散剖面
- star.species：组分（SpeciesUserSource、MassFractionProfile/MoleFractionProfile 等）
- star.radiation.common：RadiationModel（GrayThermalRadiationModel/MultiBandThermalRadiationModel）、S2sSpectrumModel、SolarLoadsModel/SolarCalculator、EmissivityProfile/ReflectivityProfile 系列、SpectralBand/SpectralGroup
- star.fea.common.models：TimeIntegrationMethod（Newmark/GeneralizedAlpha/BDF2）、MatrixUpdateStrategy

### 13. 多相物理包
- star.lagrangian：LagrangianPhase/LagrangianPhaseManager、Injector/InjectorManager/PointInjector/SolidConeInjector/VolumeStrippingInjector/FilmStrippingInjector、ParticleModel/MaterialParticleModel/MasslessParticleModel/ParcelModel/PointParcelModel、RosinRammlerParticleSizeDistribution、NukiyamaTanazawaParticleSizeDistribution、DragForceModel/SchillerNaumannDragCoefficientMethod、TurbulentDispersionModel、ParticleResidenceTimeModel、ParticleTracker、ParticleMassFlowReport、ParcelsTable；spray 子包：雾化/破碎/液膜；tracks 子包：TrackFileModel/BoundaryTrackSamples
- star.multiphase：EulerianPhase 体系（EulerianPhaseInteractionModel、Drag/Lift/VirtualMass/WallLubrication 力模型）、Coalescence*/Breakup*/Nucleation*（聚并/破碎/成核）、InteractionAreaDensityModel、ContactAngleProfile、HRICSchemeParameters、PhaseMassFlowReport/PhaseHeatTransferReport
- star.vof：VofWave 系列（FirstOrderVofWave/FifthOrderVofWave/IrregularVofWave/CnoidalVofWave）、FilmVofInteractionModel、FreeSurface*
- star.eulerianmultiphasemasstransfer：BoilingMassTransferRateModel、BubbleNucleationModel、蒸发/冷凝模型
- star.mixturemultiphase：Mmp* 系列（混合物多相）；star.dmp：DispersedMultiphaseModel/Solver、DispersedPhaseManager

### 14. star.motion / star.sixdof —— 运动与 DFBI
- star.motion：Motion/MotionManager、SixDofMotion、SixDofEmbeddedMotion、SixDofMorphingMotion、SixDofPlusRotatingMotion、TrajectoryMotion、MorphingMotion、EventsMotion、UserDefinedVertexMotion、MotionRotationAxis、RotationSpecification/RotationRate/RotationAngle/RotationOrigin、ReferenceFrameBase/ReferenceFrameManager
- star.sixdof：Body/BodyManager/BodyBase、SixDofBodyMotion、SixDofBodyConstraint、SixDofBodyCoupling、SixDofSolver、BodyFreeMotion/BodyEquilibriumMotion、CenterOfMass、MomentOfInertia、InitialValueManager、ExternalForceAndMomentManager/ExternalGravityForce/GravityForce/DampingForce、SpringDamperCoupling、ContactConstraint/ConstraintSolver、BodyForceReport/BodyMomentReport/TranslationReport

### 15. star.cosimulation —— 协同仿真
- star.cosimulation.api：ExternalCodeModel/Solver（外部代码耦合计时）
- star.cosimulation.link.common：CoSimulation/CoSimulationManager、CoSimulationType（Abaqus/Amesim/GtPower/Nastran/Rba/ReactingChannel/Starccmplus/Fmi/Cgns/Hdf5/SimcenterDataFile）、CoSimulationPartner、CoSimulationZone/CoSimulationZoneManager（点/线/面/体/壳耦合区）、LinkModel/LinkModelManager、CoSimUrfBase（收敛加速）
- star.cosimulation.common：CoSimulationValue/ValueSpecification（导入导出场/值规范）、FieldMapper/MapperManager、FieldTreatment（场处理）、CoSimBoundary、外部代码单位（ExternalCodeUnit 系列）

### 16. star.coremodule —— 客户端内核/UI（少量进 .sim）
- Application/ApplicationManager、Session、StarBeanNode/StarChildren（UI 节点树）、ActionRegistry
- star.coremodule.ui.layout：Layout/LayoutManager/LayoutObject/Mode（保存的界面布局，.sim 中可出现）
- 说明：此类多为客户端瞬态对象，.sim 中主要出现 Layout 与少数设置节点

### 17. star.automation / star.assistant —— 自动化
- star.automation：AutomationBlock/AutomationBlockManager/AutomationChain/AutomationWorkflow、LoopAutomationBlock/ConditionAutomationBlock/MeshAutomationBlock/InitializeSolutionAutomationBlock/ClearSolutionAutomationBlock/StopWorkflowAutomationBlock、SimDriverWorkflow/SimDriverWorkflowManager
- star.assistant：SimulationAssistant/SimulationAssistantLibrary、Task、Condition/ConditionTrigger（仿真助理，.sim 可保存）

### 18. 其它物理/行业包（.sim 中按激活物理出现）
- star.combustion：CombustionModel、PremixedCombustionFlameTypeModel/NonPremixedPpdfModel/PartiallyPremixedPpdfModel、LaminarFlameSpeed*、TurbulentFlameSpeed*、SparkIgnitor/TemperatureIgnitor/ProgressVariableIgnitor、EddyContactModel
- star.battery：BatteryModel/BatteryCell/BatteryModule/BatteryPack、BatterySolver、ElectricalMesh、BlockBatteryCell/CylindricalBatteryCell（电化学在 star.battery.cellproperty 子包）
- star.casting：CastingModel、SolidificationTimeModel、Primary/SecondaryDendriteArmSpacingModel、LiquidResidenceTimeModel、DimensionlessNiyamaModel
- star.overset：OversetConservationModel、自适应棱柱层
- star.morpher：MeshDeformationModel、DisplacementSpecification、ControlPointList、DesignPoint*
- star.solvermeshing：GeometryRefinementModel/Specification（求解器网格细化）
- star.stabilization：FsiAddedMassParameters（FSI 稳定化）
- star.mdx：Mdx*（Design Manager 设计集/参数/响应，.sim 中保存设计表）

---

## 三、对 .sim 解析的启示

1. **根与身份**：.sim 的根是 star.common.Simulation；图中每个对象都是 star.base.neo.ClientServerObject 子类，用 ClientServerObjectKey 唯一标识。解析器可把 ClassName + 对象键 作为节点身份，用 Instantiator/ObjectRegistry 语义理解“哪个管理器持有哪个对象”。
2. **对象图 = 管理器森林**：顶层结构几乎一一对应 star.common 的管理器：RegionManager（区域）、BoundaryManager（边界）、InterfaceManager（界面）、ContinuumManager（连续体）、GeometryPartManager/PartManager（几何部件）、MeshOperationManager（网格操作）、FieldFunctionManager（场函数）、TableManager（表格）、PlotManager（绘图）、SolverManager（求解器）、ModelManager/MethodManager（模型/方法）、CoordinateSystemManager（坐标系）、GlobalParameterManager（参数）、SimulationIterator（时间步）。
3. **DOM 序列化**：star.common.dom.ObjectModel 表明 .sim 就是“对象图 → DOM 元素”的序列化结果：每个对象一个 DOM 节点，属性（含 Dimensions/Units 包裹的量值、Profile、Polynomial）由对应 Writer 写出。解析时按 ClassName 决定节点结构。
4. **ClassName 版本漂移**：不同 STAR-CCM+ 版本类名会变（如 XyPlot→Cartesian2DPlot、PolyhedralMesher/TetrahedralMesher→DualAutoMesher、PrismLayerMesher→PrismAutoMesher、SurfaceRemesher→ResurfacerAutoMesher、View→VisView、无 star.solvermodels 包）。解析器应维护“旧名→新名”映射，不能硬编码单版本类名。
5. **包 ↔ .sim 区块对照**：
   - 几何：star.meshing（CadPart/MeshPart/SimpleBlockPart）、star.common（Part/GeometryPart）、star.cadmodeler（3D-CAD 模型，仅当使用 3D-CAD）
   - 网格管线：star.meshing（MeshOperation/AutoMeshOperation）+ 各网格器包（TrimmerAutoMesher/PrismAutoMesher/DualAutoMesher/ThinAutoMesher/DirectedMesher/ExtruderMesher/SurfaceWrapper 设置/ResurfacerAutoMesher）
   - 物理设置：PhysicsContinuum 之下是各物理包的 Model/Option（star.flow、star.energy、star.turbulence、star.segregatedflow、star.coupledflow、star.lagrangian、star.multiphase、star.vof、star.radiation.common、star.species、star.fea.common.models、star.combustion、star.battery、star.casting…）
   - 求解：star.common（Solver/SolverStoppingCriterion/Solution）+ star.segregatedflow/star.coupledflow（具体求解器）
   - 后处理：star.base.report（Report/Monitor）、star.common（Plot 类）、star.post（SolutionHistory/RecordedSolutionView）
   - 显示：star.vis（Scene/Displayer/VisView/Light/LookupTable/切面/等值面）
   - 运动：star.motion（Motion/参考系）、star.sixdof（DFBI Body）
   - 协同仿真：star.cosimulation.link.common（CoSimulation/Zone/Type）
   - 材料：star.material（Material/Property/Method）
   - 自动化/助理/布局：star.automation（AutomationBlock）、star.assistant（SimulationAssistant）、star.coremodule.ui.layout（Layout）
   - 单位与量纲：star.common（Units/UnitsManager/Dimensions）——几乎所有数值属性都挂 Dimensions + Units，解析量值必须成对读取。
6. **报告/监视器数据节点**：MonitorDataSet/TableData/DerivedData 表明 .sim 中监视器会带采样数据子节点；解数据（SolutionHistory）在 star.post。
7. **动态选择**：star.base.query 的 Query/Predicate 常作为筛选器子节点挂在边界/界面/显示器上（如按名称选边界），解析时需识别。
