# FTLPU-CMODEL 全仓周期语义与状态更新机制审计

> 调查对象：`D:\LPU\FTLPU-CMODEL-main` 当前本地文件快照  
> 调查日期：2026-11-13（Asia/Shanghai）  
> 性质：只读源码审计；未编译、未运行测试、未运行仿真、未修改正式源码  
> 结论标记：**CONFIRMED**=源码直接支持；**PARTIAL**=只确认局部/单元级行为；**UNKNOWN**=现有证据不足

## 1. 调查范围与仓库版本

### 1.1 版本边界

- **UNKNOWN — Git 分支、HEAD、最近提交和未提交状态均无法从本目录确定。** `D:\LPU\FTLPU-CMODEL-main` 当前没有 `.git` 元数据，因此题目给出的 `git status`、`git rev-parse HEAD`、`git log -1` 在该目录不具备可解释结果。不能借用 `D:\LPU` 下其他 VMODEL 仓库的 Git 信息。
- 本报告只描述本地可读文件的实际内容，不声称它等于 GitHub main 或任一 commit。
- `clock_experiment/cpp_timing_modeling.hpp` 与 `clock_experiment/CPP_TIMING_MODELING.md` 均存在。后者明确说明 Stage 1～4 的方法已在独立实验验证，但正式全系统周期精确性尚未覆盖（`clock_experiment/CPP_TIMING_MODELING.md:1-12,45-64`）。
- 正式默认硬件配置由顶层 CMake 指向 `config/ftlpu-lpu32.json`（`CMakeLists.txt:9-17`；`README.md:94-97`）。配置可以被 `FTLPU_HARDWARE_CONFIG` 覆盖，因此本文中的具体容量均注明“默认 profile”，不能视为所有 build 的固定值。

### 1.2 方法与覆盖

已逐层追踪：顶层 `TspSliceSystem`、系统 ICU、所有 `DistributedIcuQueue` 前端、Macro Decoder、QCTL、Stream Fabric、MEM、MXM、VXM、SXM、C2C Endpoint/Link/DMA/DDR、ProgramLoader，以及独立 AXI/InstructionFetchDMA。还检查了辅助的 FU3D/RUN2D 时间字段、SXM Matrix helper 和遗留 `IcuUnit`。

本报告不把以下内容混为一谈：

1. 正式系统调用链中的实现；
2. 已实现但只在独立测试中使用的组件；
3. 编译器/编码字段中的“周期”；
4. 统计计数器；
5. Stage 1～4 独立 RTL Golden 的参考方法。

## 2. 核心结论

1. **CONFIRMED — 当前正式 CModel 至少存在八类时序建模变体，而不是一个统一时钟内核：**顶层固定 Phase 调度、显式 Current/Next、局部 next-container、倒序数组移位、带倒计时的事件队列、立即更新状态机、逻辑/物理双时间域、以及纯功能存储/软件容器。
2. **CONFIRMED — `TspSliceSystem::CyclePhase` 不是 Stage 4 的四相时钟。** 它是一次 `tick()` 内的模块执行顺序保护；唯一系统级统一 commit 是 `StreamRegisterFabric::commit_cycle()`，并不统一提交所有 FU 内部状态（`include/ftlpu/system/tsp_slice_system.hpp:367-381,487-615`）。
3. **CONFIRMED — Stream Fabric 是正式模型中最完整的显式同步边界。** 所有读取针对 `current_`，写入暂存到 `next_`，最后 swap，并检查同周期多生产者冲突（`include/ftlpu/core/stream_fabric.hpp:130-179,210-251,286-410`）。
4. **CONFIRMED — ICU 同时存在系统物理周期、屏障逻辑周期、Queue 局部逻辑周期、QCTL 物理服务周期以及在途 Fetch 物理推进。** HOLD/WAIT 时不能用“所有 `cycle_` 同步加一”概括：逻辑周期可冻结，而已发出的 iMEM/QCTL/transport 响应继续返回。
5. **CONFIRMED — 多数 FU 依赖固定 C++ 调用顺序取得正确周期语义。** MEM、MXM Control、VXM Slice、SXM Slice 在单个 `evaluate/tick` 中完成“队列出队→执行→移位/清理”；它们不是全局 Current/Next 提交模型。
6. **CONFIRMED — 多数 `std::deque` 不是同步 FIFO。** 许多 MEM/MXM/C2C 指令队列没有显式容量或读写指针；它们是软件排队容器。VXM Superlane 的 3-entry queue、ICU IQ、DMA FIFO、DDR request queue 等才具有明确有限容量，但其接口语义也不完全相同。
7. **CONFIRMED — VXM 主要依赖静态调度，而非动态 valid/ready。** 输入 Bundle 缺失、Superlane FIFO 溢出等被视为调度错误并抛异常；不能把它描述成会向 ICU 自动反压的流水线。
8. **PARTIAL — 正式模块有大量 C++ 周期断言测试，但未发现正式全系统与 RTL Golden 的逐周期对照。** 按题目 M0～M5 口径，Stage 1～4 参考实验可评 M5；正式模块最高只能评 M4。
9. **CONFIRMED — `InstructionFetchDma`、`Axi4MemorySlave`、`SimpleDramBackend` 已实现且有独立测试，但未接入 `TspSliceSystem`/`C2cDmaSystem` 正式取指链路。** 它们不能作为“全系统已经完成 DRAM→AXI→iMEM”证据。

## 3. 顶层全局时钟与真实调用链

### 3.1 `TspSliceSystem::tick()`

```text
TspSliceSystem::tick()                           system cycle N
  begin_cycle_phase()
    east/west StreamRegisterFabric.begin_cycle()
  dispatch_phase()
    barrier safety check
    InstructionControlUnit.dispatch(...)
  mxm_phase()
    MxmControlSlice.tick()
    Mxm.tick_datapath()
  vxm_phase()
    VxmSlice.prepare_cycle()
    MEM→VXM capture
    VxmSlice.tick()
    VXM→MEM stage
    passive VXM bridge stage
  mem_sxm_commit_phase()
    MemArrayModel.evaluate()
    SxmSlice.evaluate()
    C2cEndpoint.evaluate()
    passive routes stage
    StreamRegisterFabric.commit_cycle()          唯一系统级共享 commit
  end_cycle_phase()
    C2cDma.tick()
    ICU advance_barrier_events()
    ++TspSliceSystem::cycle_
```

证据：`include/ftlpu/system/tsp_slice_system.hpp:367-381,487-638,730-916`。

### 3.2 Phase 的实际含义

| System Phase | 读/写对象 | 是否等价四相时钟 |
|---|---|---|
| BeginCycle | 打开两个 Fabric 的 staged-write 窗口 | 否；只是事务窗口开始 |
| Dispatch | ICU 状态机推进并把指令送入各 FU 局部队列 | 否；大量局部状态立即变化 |
| Mxm | 控制 slice 再 datapath，存在明确调用先后 | 否；是同一 tick 内拓扑顺序 |
| Vxm | 准备输入、执行、输出和桥接按顺序发生 | 否；混合组合读取与局部寄存状态推进 |
| MemSxmCommit | MEM/SXM/C2C stage 写，然后一次 Fabric commit | 部分相似 Phase0+Phase1，但只覆盖 Fabric |
| EndCycle | DMA、屏障事件、全局计数推进 | 否；是周期收尾 |

**调用顺序依赖：** MXM Control 先产生本周期 pulse，随后 MXM datapath 使用它；VXM 输入 capture 必须先于 VXM tick；MEM/SXM/C2C 必须在 Fabric commit 前完成 staged write。改变这些顺序可能改变同周期可见性，不是任意拓扑的统一 evaluate/commit。

## 4. 全局与局部周期计数器清单

| 类/变量 | 定义与更新证据 | 初值/更新频率 | 暂停与时间含义 | 与系统周期关系 |
|---|---|---|---|---|
| `TspSliceSystem::cycle_` | `tsp_slice_system.hpp:617-638,1306` | reset=0；每次完整 `tick` 末尾 +1 | 不因 FU idle/WAIT 暂停；芯片物理周期 | 主系统时间坐标 |
| `StreamRegisterFabric::cycle_` | `stream_fabric.hpp:58-73,383-410,641` | reset=0；每次 `commit_cycle` +1 | cycle 打开期间不变 | 正常由每次系统 tick 唯一 commit，与系统周期同速 |
| `InstructionControlUnit::cycle_` | `system/icu.hpp:1653-1889,2356` | reset=0；每次 `dispatch` 末尾 +1 | 顶层每 tick 调一次；主要用于 ICU 物理 trace | 通常同速，但不是 common barrier 的逻辑周期 |
| `IcuCommonBarrierControl::physical_cycle_` | `common_barrier.hpp:104-139,173` | 每次 `end_cycle` +1 | Hold 也前进 | ICU 物理时间 |
| `IcuCommonBarrierControl::logical_cycle_` | 同上 | 仅 `begin_cycle` 允许 compute 时 +1 | 等待事件时冻结；`wait_cycles_` 增加 | 可落后物理周期；测试直接验证 2 logical/6 physical（`common_barrier_test.cpp:54-74`） |
| `DistributedIcuQueue::cycle_` | `distributed_queue.hpp:1653-1748,1757-1784,2666-2716,4379` | 正常 tick/prefetch 可推进；HOLD/transport-only 不推进 | 代表队列调度逻辑周期；WAIT/QCTL Hold 冻结 | 不能一一映射系统周期，需借 trace 的物理时间/调用点 |
| `queue_control_physical_cycles_` | `distributed_queue.hpp:1018-1019,2666-2716,4320-4321` | raw Macro 每个物理服务周期推进 | QCTL wait 时仍推进 | 专用于 QCTL/队列物理占用统计 |
| `raw_compatibility_cycle_` | `distributed_queue.hpp:2549,4377-4384` | 编译/兼容计算维护 | 非正式物理 Queue clock | 不应映射为系统物理周期 |
| `MacroBitstreamDecoder` | `macro_bitstream_decoder.hpp:172-213,854-980` | 无独立 `cycle_`；由调用者传 `current_cycle`，统计 run tick | `hold()` 只推进 fetch response，不推进 parser/DDB/admission统计周期 | 共享 Queue 逻辑周期，但物理 fetch 可单独前进 |
| `IcuQueueControlPool::cycle_` | `queue_control_pool.hpp:66-105,148-149` | 每次 `end_cycle` +1 | shared one-port 仲裁持续推进 | ICU 物理服务域 |
| Private QCTL frontend | `queue_control.hpp:43-58,87-93` | 无绝对 cycle；pending.remaining 每周期递减 | due head 等待时仍可缓存返回 | 相对延迟事件域 |
| `MemArrayModel::cycle_` | `mem_array.hpp:418-450,869-898` | reset=0；每次 evaluate +1 | 无 idle gate | 顶层每 tick 调一次，正常同速 |
| `MxmControlSlice::cycle_` | `mxm/control_slice.hpp:651-673,1100-1125` | 每次 tick +1 | 无 idle gate | 顶层每 tick 每实例调用，正常同速 |
| `Mxm` datapath | `mxm/mxm.hpp:149-258,722-850,1057-1098` | 无独立公开 `cycle_`；每次 `tick_datapath` 推进显式 pipeline | 无独立 HOLD | 借顶层/MXM control 周期解释 trace |
| `VxmSlice::cycle_` | `vxm/slice.hpp:203-229,543` | 每次 tick +1 | 无 idle gate | 顶层每 tick 调一次 |
| `VxmSuperlane::cycle_` | `vxm/superlane.hpp:490-579,647` | 每次 superlane tick +1 | 配置不 ready 会失败/无 issue，而非全局 stall | 由 VxmSlice 每 tick 驱动 |
| `VxmLane::cycle_` | `vxm/lane.hpp:442-726,1249` | 每次 lane tick +1 | 与 superlane lockstep | 正常与 VXM slice 同速 |
| `SxmUnitGroup::cycle_` | `sxm/unit_group.hpp:49-101,288` | `complete_cycle` +1 | evaluate 后必须 complete | `SxmSlice::cycle()` 以 UnitGroup 周期为基准（`sxm/slice.hpp:96-126`） |
| `SxmMatrixCore::cycle_` | `sxm/matrix.hpp:69-84,112-167` | 独立 helper tick +1 | 可 synchronize_idle 到 MEM cycle | 非 `TspSliceSystem` 主 SXM slice 核心路径 |
| `C2cDma::cycle_` | `c2c/dma.hpp:188-208,324-340` | 每次 DMA tick +1 | 无 idle gate | `TspSliceSystem::end_cycle_phase` 每 tick 驱动 |
| `C2cLink::cycle_` | `c2c/link.hpp:79-137` | 每次 link tick +1 | serialization/flight 每 tick老事件先 age | DualChip 系统时间域 |
| `DualChipC2cSystem::cycle_` | `system/dual_chip_c2c_system.hpp:60-85` | chip0、chip1、links 后 +1 | 不暂停 | 两个 chip 在 link 前都完成，避免先调用 chip 获得额外跨片优势 |
| `Ddr4Model::cycle_` | `c2c/ddr4.hpp:265-314,331,457` | 每次 tick +1 | 请求 latency/bandwidth均推进 | C2cDmaSystem 每 tick 驱动 |
| `C2cDmaSystem::cycle_` | `system/c2c_dma_system.hpp:59-97` | loader→chip→DDR 后 +1 | 不暂停 | 外层系统物理周期 |
| `ProgramLoader` | `system/program_loader.hpp:235-358,387-401` | 无自己的 free-running `cycle_`；接收外部 cycle | route/write commit 使用外部物理周期与 `next_write_cycle_` | 由 C2cDmaSystem 传入统一外层 cycle |
| `InstructionFetchDma::cycle_` | `icu/instruction_fetch_dma.hpp:223-345,494` | start 清零；每次独立 tick +1 | valid/ready backpressure 下仍推进控制周期 | **未接入正式顶层** |
| `Axi4MemorySlave::cycle_` | `axi/axi4_memory_slave.hpp:170-253,392` | 每次 tick +1 | R stall 时保持输出并统计 | **未接入正式顶层** |
| `SimpleDramBackend::cycle_` | `axi/memory_backend.hpp:69-92,145-205,257,314` | 每次 tick +1 | pending latency 与带宽服务推进 | **未接入正式顶层** |
| 遗留 `IcuUnit::cycle_` | `icu/icu.hpp:13-50,91-220` | Running/Nop/Repeat 条件推进 | 非每次调用都推进 | 未找到正式实例化，归类 legacy/dormant |

### 4.1 可否统一映射到全局物理周期

- **可直接映射：** Tsp、Fabric、MEM、MXM Control、VXM Slice/Superlane/Lane、SXM UnitGroup、片内 C2C DMA，在正常 `TspSliceSystem::tick` 调用链中均每物理 tick 一次。
- **必须带域标签才能映射：** ICU Queue `cycle_`、common barrier logical cycle、Macro `next_issue_cycle`、FU3D `wait_cycle/cycle_strides`。这些是逻辑调度坐标或指令局部 offset，不是系统绝对 cycle。
- **独立系统映射：** C2cDmaSystem/DDR、DualChip/Link、AXI/InstructionFetchDMA 各有自己的 top tick；只有被同一外层逐周期驱动时才可比较 cycle 数值。
- **无法仅靠局部计数反推全局周期：** Queue HOLD/WAIT 会让同一个 queue logical cycle 覆盖多个 physical cycles。必须保留 physical trace timestamp 或显式传入 global cycle。

## 5. 状态更新机制变体

### 5.1 已确认的八种变体

| 变体 | 代表模块 | 语义与风险 |
|---|---|---|
| 1. 顶层固定 Phase 顺序 | TspSliceSystem | 用函数调用顺序表达数据依赖；不是统一 edge scheduler |
| 2. 完整 Current/Next + 统一 commit | StreamRegisterFabric | 最接近同步寄存器阵列；同周期写冲突显式检查 |
| 3. 局部 next-container 后整体替换 | Mxm datapath、VxmLane、C2C TX | 在单个函数内部正确采样旧状态，但没有跨模块统一 commit |
| 4. 倒序数组移位 | MEM/VXM/SXM instruction rows、C2C RX、Special ALU | 若先读后倒序移位可等价同步 pipeline；必须逐个函数检查顺序 |
| 5. Pending event + remaining/ready_cycle | iMEM Fetch、QCTL、MEM R1、DDR/DRAM、C2C link | 是延迟事件模型，不是 FIFO 指针模型 |
| 6. tick 内立即更新 FSM/queue | ICU、MEM/MXM control queues、SXM buffers | 后续同函数/后续模块可能看到新状态，依赖调用顺序 |
| 7. 逻辑/物理分离 | Common Barrier、Distributed Queue/QCTL | HOLD 冻结程序逻辑时间，但捕获物理响应/transport |
| 8. 纯功能状态 | SRAM、编码器/解码 helper | 直接读写，无自身周期；时序由上层 wrapper 决定 |

### 5.2 明确 Current/Next 的位置

- **Stream Fabric（强）：** `begin_cycle()` 打开写窗口；读 API只看 `current_`；`stage_write()`只改 `next_`；`commit_cycle()` swap 并清除已消费/陈旧 cell（`stream_fabric.hpp:130-179,210-251,286-410`）。
- **MXM datapath（局部）：** 从旧 `east_pipeline_` 计算，构造 `next_pipeline` 后一次赋值；decode pipeline 同样使用 current/next（`mxm/mxm.hpp:176-258,722-850`）。
- **VXM Superlane config（局部）：** `queues_ → decoding_ → next_ → current_`，`consume()`只在物理执行接受后减少 repeat（`vxm/superlane.hpp:64-170,341-344`）。
- **VXM Lane（局部）：** 每次 tick 构造 `next_inputs`，完成 ALU 路由后 `stage_inputs_=next_inputs`（`vxm/lane.hpp:442-719`）。
- **C2C TX（局部）：** 用 `next_pipeline` 处理 downstream ready 后整体替换（`c2c/slice.hpp:137-204`）。
- **不应误称 Current/Next：** Macro reservoir/DDB、MEM RegisteredR1 response deque、DDR pending request、QCTL pending response均是事件/缓存容器。

### 5.3 立即更新与调用顺序风险

- ICU `dispatch()`依次 tick 不同 queue，立即消费/产生 queue 状态，末尾才增加 ICU cycle；没有“全部 queue 先 evaluate、再统一 commit”（`system/icu.hpp:1653-1889`）。
- MEM `evaluate()`在一次调用中先 stage 上周期 R1 响应，再把 deque 头放入 row0，执行当前行，最后移位并增加周期（`mem_array.hpp:418-450,527-604,785-797`）。新入 row0 的命令可在本次 evaluate 被执行，属于约定好的软件顺序。
- MXM Control 同一 tick 依次 dispatch、准备 weight input、execute、advance rows（`mxm/control_slice.hpp:651-673,775-802,1087-1098`）。
- VXM Slice 同一 tick dispatch row0、执行 instruction rows、驱动 superlane、最后移位（`vxm/slice.hpp:203-219,289-455`）。
- SXM Slice 同一 evaluate 内完成 due output、UnitGroup evaluate、Permute/Transpose capture、complete/shift（`sxm/slice.hpp:203-247,358-610`）。
- C2C RX 的共享路径和 unindexed 路径使用不同的推进写法；一个先移位再接收，一个移位/row0/输出/清尾串行进行（`c2c/slice.hpp:384-509`）。迁移时不能只做机械 API 改名。

## 6. 数组移位与流水线清单

| 流水结构 | 级数/容量 | 推进方式 | stall/hold | 证据与实际延迟边界 |
|---|---:|---|---|---|
| MEM `instruction_rows_` | `kTileRows` | evaluate末尾倒序移位 | 无通用 stall；输入按队列/资源检查 | `mem_array.hpp:527-577,785-797`；RegisteredR1 另加 response queue |
| MXM control load/compute rows | `kMxmSupercellsPerPlane` | dispatch row0、execute、末尾 advance | 缺 weight stream 抛错 | `mxm/control_slice.hpp:651-673,775-802,1087-1098` |
| MXM east datapath | 本地 MAC stages + supercell传播 | 显式 `next_pipeline` | 无动态 backpressure | `mxm/mxm.hpp:149-258` |
| MXM decode pipeline | 固定数组 | current→next→output shift | 无动态 backpressure | `mxm/mxm.hpp:722-850` |
| VXM instruction rows | `kTileRows` | dispatch/execute后倒序移位 | 缺静态输入报错，不重试 | `vxm/slice.hpp:289-455` |
| VXM basic ALU | 默认1 cycle；乘法2 cycle | 请求 pipeline | fully pipelined | `vxm/alu.hpp:33-120` |
| VXM special ALU | 5 stages | downstream-to-upstream in-place shift | 无全局 stall | `vxm/special_alu.hpp:70-74,227-317` |
| VXM lane stage inputs | 16 ALU slots | 本地 `next_inputs` 替换 | 依配置/feedback规则 | `vxm/lane.hpp:442-719,1224-1250` |
| VXM static quantizer | 1 pending vector | tick移交 | 无 ready/replay | `vxm/static_quantizer.hpp:13-66` |
| SXM transpose instruction rows | `kTileRows` | evaluate末尾移位 | source slot占用时报满/错 | `sxm/slice.hpp:203-247,448-454,692-707` |
| SXM Permute | 单一 one-cycle slot | `ready_cycle < current cycle` 才可输出 | 无队列化覆盖 | `sxm/slice.hpp:506-610` |
| WideTranspose | jobs + input/output beat deques | ready-cycle/job推进 | 有 initiation interval/容量检查 | `sxm/wide_transpose.hpp:10-22,36-129` |
| C2C TX/RX | `kTileRows` | TX next_pipeline；RX倒序移位 | link credit/下游空位 | `c2c/slice.hpp:137-279,384-509` |
| C2C Link | serialization + flight | 老 in-flight 先 age，新 flight 当周期不 age | RX credit | `c2c/link.hpp:34-105` |

**结论：** 倒序移位本身不是问题；真正风险是“row0何时装入、当周期是否立即执行、输出何时被 Fabric commit”。迁移必须保留每个模块当前的读写顺序和测试所见边界。

## 7. FIFO、Queue、Buffer 与存储清单

### 7.1 ICU 前端（默认 `ftlpu-lpu32`）

| 结构 | 默认容量/类型 | 硬件语义分类 | 满载/延迟/反压 |
|---|---|---|---|
| Queue-local iMEM | MEM 131072×96b；MXM 65536×128b；VXM/SXM 2048×96b；C2C 32768×96b | 有限存储镜像 | launch 后禁止写；超深度拒绝。配置：`config/ftlpu-lpu32.json:4-24` |
| Queue IQ | 各类16 entries | 有限 IQ，但实现用容器而非显式指针 RTL | fetch latency默认1；IQ credit限制新 fetch（同上:7,11,15,19,23,27） |
| Pending native fetch | 单个 optional `{pc,entry,remaining}` | 延迟事件 | completion优先于新 request；HOLD仍捕获（`distributed_queue.hpp:2736-2814`） |
| Macro reservoir | 3 physical words | 有限 bit reservoir | deque word + head bit offset，不整体移位（`macro_bitstream_decoder.hpp:423-501`） |
| Macro DDB | 8 descriptors；run capacity 8 | 有限 descriptor buffer | admission/lookahead均8；满载统计 stall（配置:30-34；decoder:1592-1703） |
| Active Context | MEM32、MXM40、VXM/SXM/C2C各4 | 有限 context RAM | admission按 occupancy/release cycle；独立于 fetch IQ（配置:8,12,16,20,24） |
| Private QCTL RAM/cache | RAM16、cache4、read latency2 | 有限控制存储+pending response | begin提交 ready response；end最多发一读（`queue_control.hpp:17-93`） |
| Shared QCTL Pool | pool512、每queue limit16、cache4、1 read port、II=1 | 共享有限 RAM/cache | 每物理周期 round-robin 最多一 grant（`queue_control_pool.hpp:34-150`；配置:38-45） |
| Common Barrier table | 1024 | 有限控制表 | 逻辑 cycle Hold；物理 cycle继续（配置:35-36；`common_barrier.hpp:35-175`） |

参数桥接证据：`include/ftlpu/core/hardware_params.hpp:12-21,143-180`。该文件引用生成的 `config::*`，不是自身写死数值。

### 7.2 FU 与互连

| Buffer/Queue | 容量与容器 | 分类 | 行为 |
|---|---|---|---|
| MEM per slice/bank/port instruction queue | 多维 `std::deque`，未见显式容量 | 软件排队容器 | enqueue选择read/write port；不是 full/empty FIFO（`mem_array.hpp:256-271,869-898`） |
| MEM instruction rows | `kTileRows` optional array | 固定寄存流水 | 每 tick移位 |
| MEM RegisteredR1 responses | deque | 延迟事件队列 | 上周期 snapshot 在本周期 stage 到 Fabric（`mem_array.hpp:579-604`） |
| SRAM | sparse storage map/直接 API | 功能存储 | 无自身 tick、端口冲突或读延迟（`mem/sram.hpp:15-44`; `src/mem/sram.cpp:27-65`） |
| MXM load/dequant/compute queues | 3个 `std::deque`，无显式容量 | 软件指令队列 | `issue` push；tick dispatch（`mxm/control_slice.hpp:529-545,1102-1104`） |
| MXM control/data pipelines | fixed optional arrays | 寄存流水近似 | 无通用 ready/valid；资源缺失报错 |
| VXM Slice compact queues | 每ALU `std::deque`，未见容量 | 软件指令队列 | 向 Superlane 下发前依静态资源状态 |
| VXM Superlane FIFO | 8 logical stage queues，每个深度3 | 有限硬件队列近似 | 溢出直接 throw，不 stall 上游（`vxm/superlane.hpp:26-33,113-170,341-344`） |
| VXM InputBuffer | Empty/Collecting/Ready 单 bundle | 收集寄存状态 | group可跨周期到达；重复/缺失报错；成功 issue后 release；无 retry/backpressure（`vxm/input_buffer.hpp:32-122`） |
| VXM feedback/output/quantizer | fixed optional arrays | 局部寄存/事件状态 | 由 Lane tick统一推进（`vxm/lane.hpp:1224-1250`） |
| SXM transpose rows/bank | fixed optional rows + bank state | 固定流水/局部buffer | incomplete block保持busy；满则拒绝/异常 |
| SXM Permute slot | 1 instruction | 单寄存槽 | 强制至少一完整周期隔离 |
| WideTranspose jobs/input/output | deques + fixed matrix | 专用延迟模型 | 有 job II 与 payload capacity；不是通用 FIFO |
| C2C Endpoint TX/RX command queues | 多个 `std::deque`，部分无显式容量 | 软件queue + fixed pipeline | fixed tile-row pipeline；link credit控制末端 |
| C2C Link RX FIFO | 默认4 vectors | 有限 credit FIFO | `can_send`受 outstanding 与 RX depth 限制（`c2c/types.hpp:19-34`; `c2c/link.hpp:34-105`） |
| C2C DMA inbound/outbound/pending | 默认每stream 16 vectors | 有限 FIFO/在途事件 | pending+inbound共同占 credit；动态 backpressure到DDR request（`c2c/dma.hpp:33-41,141,223-296,325-337`；配置:240-250） |
| DDR request queue | 默认256 | 有限请求队列 | latency后按channel/带宽服务；read/write completion deques（`c2c/ddr4.hpp:23-34,156,265-314,415-457`） |
| ProgramLoader route/write queues | 配置化 bus32B/cycle、route1、write II1、commit1 | 弹性route+延迟提交 | credit/target busy会等待（`program_loader.hpp:13-16,235-401`; 配置:252-256） |
| AXI Slave R FIFO | 配置化 | 有限 response queue | RVALID/RREADY stall保持 beat（`axi/axi4_memory_slave.hpp:170-253`） |
| InstructionFetchDMA assembler/word output | 配置化 byte assembler + 单输出保持 | valid/ready组件 | 下游不ready时 word和AXI控制保持；独立测试覆盖（`instruction_fetch_dma.hpp:262-340,373-498`） |

### 7.3 三类概念判定

- **A 真正/近似硬件有限 FIFO：** ICU IQ、Macro Reservoir/DDB/Active Context、QCTL cache/pool、VXM Superlane FIFO、C2C Link/DMA/DDR queues、AXI response FIFO。
- **B 延迟事件队列：** pending iMEM fetch、QCTL pending read、MEM R1 response、Macro parser/DDB residency、C2C serialization/flight、DDR/DRAM request、ProgramLoader route/write commit。
- **C 软件容器：** MEM/MXM/Slice command deques、部分 C2C endpoint command queues、编码/程序图像 vectors。除非补充物理容量与握手契约，不应直接用于 SRAM/FIFO PPA 推断。

## 8. 握手、反压、静态调度与异常

| 类别 | 模块 | 未接受事务行为 |
|---|---|---|
| 标准 valid/ready | AXI Slave ↔ InstructionFetchDMA | AR/R handshake；R/word 在 backpressure 下保持，统计 stall |
| Credit/容量反压 | C2C Link、C2C DMA、DDR、ProgramLoader | 不发新请求或等待 credit；已发请求继续 age/返回 |
| staged producer collision | Stream Fabric | 同cell多producer抛异常；consumer标记抑制被消费segment转发 |
| Queue admission | ICU IQ/Macro/QCTL | 满时不发 fetch/admit；HOLD仍吸收已到 response |
| 静态调度契约 | VXM、MXM多处、SXM local block | 缺输入、溢出或碰撞直接异常；不是动态反压 |
| 无显式反压 | MEM/MXM若干 instruction deques | push可增长；上游容量真实性未建模 |

VXM 特别结论：`VxmInputBuffer` 收集既定 bundle，`VxmSlice` 只有在所有 required streams 到齐时执行；未到齐通常是 static schedule error，不会保持一个 ready/valid request向上游反压（`vxm/slice.hpp:366-445,490-525`; `vxm/input_buffer.hpp:47-122`）。Superlane FIFO 深度3且 overflow throw 的行为由测试明确说明（`tests/unit/vxm/superlane_test.cpp:52-76`）。

## 9. HOLD、WAIT 与物理/逻辑周期分离

### 9.1 Native queue HOLD

`DistributedIcuQueue::hold_cycle()`会接收已经 ready 的 fetch response并 age pending fetch，但冻结 decoder FSM、loop/repeat/NOP和 queue `cycle_`（`distributed_queue.hpp:1653-1673`）。`tick_transport_only()`允许 synchronized transport继续推进，又不推进 queue逻辑周期（`:1757-1784`）。

### 9.2 Raw Macro/QCTL HOLD

`tick_raw_macro()`先检查当前 queue logical cycle 是否有QCTL due。WAIT或shared transport hold时，Macro decoder不运行，queue `cycle_`不变，但 `queue_control_physical_cycles_`继续增加；正常路径先尝试从已有 Active Context issue，再 tick decoder，因此新 context最早下一逻辑周期可 issue（`distributed_queue.hpp:2666-2716`）。

### 9.3 Common Barrier

`begin_cycle()`根据 barrier event决定 Run/Hold；`end_cycle()`无论是否 compute-enabled 都推进 physical cycle，只在 Run推进 logical cycle（`common_barrier.hpp:98-139`）。系统 `dispatch()`在 barrier禁止计算时对所有 compute queue调用 hold，但 transport与事件路径仍可工作（`system/icu.hpp:1653-1889`）。

### 9.4 在途响应原则

- 已发 iMEM/QCTL read不能因为逻辑WAIT丢失，必须进入cache/reservoir/IQ或保持 pending。
- DDR/C2C/ProgramLoader等物理 transport不应绑定某条 compute queue的逻辑 cycle。
- 后续统一时钟模型必须把“是否推进物理事件”和“是否推进程序逻辑状态”拆成独立 enable，不能只给模块一个 `tick_enabled`。

## 10. 模块级时序语义矩阵与成熟度

评级采用题目定义：M0功能状态；M1时间戳；M2模块周期；M3 Current/Next或等价同步更新；M4模块间周期协作/握手/反压；M5 RTL Golden逐周期验证。

| 模块 | 周期来源 | 状态更新 | 延迟/握手 | 评级 | 依据与限制 |
|---|---|---|---|---:|---|
| TspSliceSystem | global `cycle_` | 固定Phase调用 | 全局协调 | M4 | 有跨模块phase和guard；无正式RTL逐周期Golden |
| InstructionControlUnit | 每dispatch物理周期 + barrier logical | tick内顺序更新 | WAIT/HOLD/event | M4 | 跨多queue与FU协调；非统一commit |
| DistributedIcuQueue | queue logical + frontend physical | FSM/容器立即更新 | IQ credit、fetch/QCTL delay | M4 | 物理/逻辑分离较完整；接口复杂 |
| Macro Decoder | 调用者current_cycle | reservoir/parser/DDB顺序更新 | finite admission/fetch | M3～M4 | 局部多级时序明确；无独立统一edge |
| QCTL private frontend | 相对remaining | begin/end split | cache credit | M3 | 明确响应边界，局部使用 |
| QCTL shared pool | pool physical cycle | begin complete/end arbitrate | one-port RR反压 | M4 | 多queue共享端口协作明确 |
| StreamRegisterFabric | fabric cycle | Current/Next+commit | consume/collision | M4 | 正式最强同步边界；无RTL Golden |
| MemArrayModel | local每evaluate | queue→row→execute→shift | R1 pending response | M3～M4 | 与Fabric协作，但内部立即更新 |
| SRAM | 无 | 直接map读写 | 无时序/端口模型 | M0 | 时序由MemArray wrapper提供 |
| MxmControlSlice | local每tick | deques+row shift | 静态资源检查 | M2～M3 | 周期明确，缺通用backpressure |
| Mxm datapath | 顶层调用 | 局部next pipeline | 静态 | M3 | 显式旧/新pipeline，跨模块commit未统一 |
| VxmSlice | local每tick | dispatch/execute/shift | 静态bundle | M2～M3 | 缺输入报错，不是动态握手 |
| VxmSuperlane | local每tick | FIFO/decoding/next/current | depth3 overflow throw | M3 | 当前/下一级配置边界清楚 |
| VxmLane | superlane lockstep | `next_inputs`+内部pipelines | 无通用stall | M3 | 局部同步语义较强 |
| SxmSlice | UnitGroup cycle | evaluate+complete+shift | slot/bank资源检查 | M2～M3 | 专用ready_cycle与buffer语义 |
| SxmUnitGroup | local complete_cycle | issued集evaluate后clear | 无持续队列 | M2 | 明确周期，但定义为stateless executor |
| WideTranspose | owner cycle | jobs/beat事件 | II/capacity | M3 | 应保留专用延迟模型 |
| C2C Endpoint | 系统tick | fixed pipelines + deques | link ready/credit | M3～M4 | TX/RX更新风格不一致 |
| C2C Link | link cycle | serialization/flight events | credit RX FIFO | M4 | 动态反压与在途保持明确 |
| C2C DMA | DMA cycle | active+pending+FIFO | DDR容量反压 | M4 | 多stream在途/完成通知明确 |
| DDR4 Model | DDR cycle | request events | queue/channel/bandwidth | M4 | 专用外存延迟模型，应保留 |
| ProgramLoader | 外部physical cycle | elastic route/write events | credit/commit latency | M4 | 系统装载跨模块协作完整 |
| InstructionFetchDMA | 独立cycle | handshake FSM/assembler | AXI valid/ready | M4（独立） | 有强单测，但未接正式系统 |
| AXI4 Memory Slave | 独立cycle | evaluate+tick | AXI valid/ready | M4（独立） | 未接正式系统 |
| SimpleDramBackend | 独立cycle | pending request events | bandwidth/latency | M2～M3 | 明确声明非bank/row/refresh模型 |
| legacy IcuUnit | 条件local cycle | FSM立即更新 | NOP/Repeat | M1～M2 | 未找到正式调用，勿作为现状基线 |
| Stage 1～4 reference | FourPhaseClock | Current/Next统一commit | FIFO/backpressure | M5（仅实验） | 不能外推到正式模块 |

## 11. 现有测试与证据强度

本调查只阅读测试源码，**未执行**。因此“测试存在”不等于“当前快照已实际通过”。

| 模块/测试 | 验证行为 | 中间/逐周期 | 独立RTL Golden | 强度 |
|---|---|---|---|---|
| `tests/unit/stream/stream_fabric_test.cpp:12-63` | commit后可见、每周期一列传播、consume、冲突 | 是/是 | 否 | 强（C++） |
| `tests/unit/icu/raw_macro_decoder_test.cpp:106-178` | priming、fetch latency/II、上下文解码周期 | 是/是 | 否 | 强（单元） |
| `tests/unit/icu/queue_control_test.cpp:59-66,180-202,291-305` | WAIT不停止独立计算、cache pending、shared one-port/latency | 是/是 | 否 | 强（单元） |
| `tests/unit/icu/common_barrier_test.cpp:54-74,140-179,207-226` | logical/physical分离、HOLD保留Macro/fetch | 是/是 | 否 | 强（单元/系统） |
| `tests/unit/mem/mem_sync_read_test.cpp:20-61` | RegisteredR1 snapshot、tail drain、Fabric commit边界 | 是/是 | 否 | 强（单元） |
| `tests/unit/mxm/control_slice_test.cpp:89-188,235-261` | tile wave、并行load/compute、缺输入错误 | 是/部分 | 否 | 中～强 |
| `tests/unit/vxm/input_buffer_test.cpp:14-61` | Collecting→Ready→release、重复group拒绝 | 是/状态序列 | 否 | 中 |
| `tests/unit/vxm/superlane_test.cpp:52-148,218-332` | depth3、decode-start、连续issue、feedback边界 | 是/是 | 否 | 强（单元） |
| `tests/unit/sxm/local_blocks_timing_test.cpp:49-93,114-151` | 禁止同周期T→P bypass、partial busy、source release | 是/是 | 否 | 强（单元） |
| `tests/subsystem/c2c/dual_chip_test.cpp:179-188,259-263` | credit、serialization/flight、跨chip同步 | 是/是 | 否 | 强（子系统） |
| `tests/subsystem/c2c/dma_ddr4_test.cpp:63-86,225-270,305-391` | latency/beat、transport-only逻辑冻结、相对age | 是/是 | 否 | 强（子系统） |
| `tests/unit/icu/instruction_fetch_dma_test.cpp:185-248,327-471` | AXI握手保持、burst、queue/assembler backpressure | 是/是 | 否 | 强（独立组件） |
| `tests/unit/axi/axi4_read_test.cpp` | R channel stall、FIFO、backend latency/bandwidth | 是/是 | 否 | 强（独立组件） |
| `tests/subsystem/system/program_loader_test.cpp:178-198,217-265` | route/write参数影响、drain、外层周期一致 | 是/是 | 否 | 强（子系统） |
| `clock_experiment` Stage 1～4 | C++/RTL逐半周期/逐阶段一致 | 是/是 | 是 | M5但仅参考实验 |

**缺口：** 未发现把正式 `TspSliceSystem` 的全部 FU state/handshake 与 RTL 逐周期比较的 Golden；也未发现统一捕获所有模块 pre-edge/post-edge 状态的正式 trace schema。

## 12. 与候选四阶段 ClockPhase 的映射

参考契约：Phase0只读旧Current并准备Next；Phase1统一commit；Phase2只重算更新后Current的纯组合输出；Phase3保持正沿寄存状态（`clock_experiment/CPP_TIMING_MODELING.md:26-43`; `cpp_timing_modeling.hpp:10-76`）。

| 模块 | Phase 0 候选 | Phase 1 候选 | Phase 2 是否需要 | Phase 3 | 迁移判断 |
|---|---|---|---|---|---|
| TspSliceSystem | 调用所有 evaluate/prepare | 统一调所有commit | 仅组合可见输出/接口重算 | trace/TB驱动 | **必须重构调度器语义**；现有CyclePhase不可直接改名复用 |
| Stream Fabric | begin+read current+stage write | `commit_cycle` | 若同一高电平需观测更新后组合route/ready则需要；当前系统可下一tick读 | 无状态改动 | **最容易接入，主体可保留** |
| ICU/Queues | 计算issue/fire/Next FSM、物理response acceptance | 提交queue/FSM/counter | 通常不需要；除非暴露组合ready/due | 只做TB/trace | **优先拆分**，尤其逻辑enable与物理event enable |
| Macro/QCTL | 解析/admission/arbitration，计算pending next | 统一提交reservoir/DDB/cache | due/ready输出可选重算 | 无 | **保留专用延迟结构，统一时间/提交边界** |
| MEM | 从旧rows/queues/SRAM生成执行和Fabric staged writes | 提交rows、response、queue state | SRAM若定义异步组合读才需要；当前R1不需要 | 无 | **需要明确row0当周期执行契约** |
| SRAM | 纯组合/存储访问由wrapper定义 | 写提交若要拟真同步SRAM | 取决于最终SRAM端口模型 | 无 | **当前M0，不能直接四相包装解决端口时序** |
| MXM Control/Data | control/data分别由旧状态算Next | 同一edge提交两者 | datapath组合输出若硬件当高电平可见才需要 | 无 | **优先消除control先调用给datapath的隐式同周期可见性** |
| VXM Slice/Superlane/Lane | 旧config/input/feedback算Next | 同时提交instruction/config/lane pipelines | 更新后Current驱动的纯组合feedback/ready可能需要 | 无 | **分层迁移，保留ALU专用pipeline** |
| SXM | 旧slot/bank/job求输出/Next | 提交slot/bank/row | 如果P组合输出由新T寄存器直接驱动则需要；当前明确禁止same-cycle bypass | 无 | **大多不需要Phase2状态变化** |
| C2C Endpoint/Link/DMA | 旧credit/pipeline/request算Next | 提交所有链路状态 | ready/valid外部输出建议重算 | 无 | **适合统一handshake fire模型** |
| DDR/DRAM | age/仲裁/带宽服务算Next | 提交request/completion | ready可重算，memory物理细节无需门级settle | 无 | **保留事件队列模型** |
| AXI/InstructionFetchDMA | `evaluate()`生成输出和fire | `tick`状态提交 | **需要**重算commit后的 ARREADY/RVALID/word valid 供观察 | 无 | **最接近四阶段外部接口，但尚未系统集成** |

### 12.1 Post-edge Settle 结论

- 不应给每个模块无条件增加一个“Phase2更新状态”函数；Phase2不得更新Next或推进event。
- **明确受益者：** AXI/ready-valid接口、未来对外暴露的FIFO full/empty/ready、以及需要记录正沿后组合观察值的统一trace。
- **通常不需要单独Phase2：** RegisteredR1 MEM、固定寄存流水、Macro/QCTL延迟事件、DDR latency queue；它们可在下一Phase0读新Current。
- Stream Fabric在当前正式系统中 commit后没有同周期状态消费者，因此不加Phase2不会改变现状；若未来希望模拟“正沿后组合route/ready立即变化”，再只重算观察输出。

## 13. 主要缺口与潜在风险

1. **跨模块提交不统一。** Fabric以外的FU状态在各自函数中立即变化，顶层顺序本身成为时序规范。
2. **cycle命名未携带域。** Queue logical、ICU physical、system global、指令local offset都可叫cycle，trace合并容易误读。
3. **若干指令deque无物理容量。** 这会隐藏真实full/ready和上游反压，不能用于FIFO深度/PPA结论。
4. **静态调度错误与动态反压混杂。** VXM/SXM/MXM用throw表达“硬件上不应发生”，而AXI/C2C用保持/credit；统一接口必须保留差异。
5. **SRAM仍是功能存储。** 无端口、同步读写冲突、写优先级与真实macro延迟；仅有上层MEM timing wrapper。
6. **Post-edge可见性没有全系统规范。** 现有测试多按每次tick观察，没有统一 pre-edge/fire/post-edge字段。
7. **正式系统没有M5证据。** Stage 1～4证明方法可行，不证明所有正式FU满足该方法。
8. **独立AXI取指未集成。** InstructionFetchDMA的周期语义暂时不能证明它与DistributedIcuQueue HOLD/credit组合后仍正确。

## 14. 建议的统一顺序（仅迁移规划，不是接口设计）

### 优先级 P0：定义时间域和观察契约

1. 先冻结术语：`global_physical_cycle`、`queue_logical_cycle`、`transport_cycle`、`instruction_local_offset`。
2. 给统一trace定义 pre-edge input、fire、post-edge state、post-edge combinational observation；不先改模块。
3. 建立正式 CModel 的小范围 RTL Golden，至少覆盖 Fabric + 两级有限queue + HOLD中的在途response。

### 优先级 P1：先接入已有明确边界的模块

1. StreamRegisterFabric：保留Current/Next主体，只由统一scheduler调用。
2. AXI/InstructionFetchDMA：保持valid/ready语义，作为四阶段外部接口样板，但先保持独立，不直接宣布全系统集成。
3. C2C Link/DMA/DDR：统一时间戳和commit边界，保留serialization、credit、latency/bandwidth专用模型。
4. QCTL frontend/pool：保留begin/end和shared one-port仲裁，明确response acceptance属于物理域。

### 优先级 P2：拆分调用顺序依赖

1. ICU Queue：把物理response capture、逻辑issue、状态提交拆清；保留HOLD语义。
2. MEM/MXM/VXM/SXM：逐个把“读取旧状态/产生side effect/推进pipeline”列成契约，再决定局部Current/Next粒度。
3. MXM Control→datapath与VXM input→superlane是最高风险的同周期可见路径，必须用现有测试锁定后再迁移。

### 优先级 P3：补物理容量，而不是机械包装

对MEM/MXM/VXM Slice/C2C Endpoint的无界deque，先由架构确定物理容量、满载行为和是否允许同周期push/pop；在这些事实未确定前，不应把它们统一替换成一个通用FIFO模板。

### 建议保留的专用机制

- Macro reservoir/parser/DDB/Active Context 的多级前端；
- QCTL shared one-port仲裁与逻辑/物理周期分离；
- MEM RegisteredR1 snapshot；
- MXM/VXM/SXM专用流水与静态调度检查；
- WideTranspose、C2C serialization、DDR/DRAM latency+bandwidth事件模型；
- ProgramLoader弹性route/write commit模型。

“统一”应统一时间坐标、edge契约、fire/commit和trace，不应消除这些专用延迟结构。

## 15. 尚未确认的问题

1. **UNKNOWN — 当前实际构建使用的 profile。** 默认是 `ftlpu-lpu32.json`，但本目录无可验证的生成 `hardware_config.hpp` 或 Git/build provenance，不能证明某个既有可执行文件没有用 `-DFTLPU_HARDWARE_CONFIG` 覆盖。
2. **UNKNOWN — 正式 RTL 对各FU内部queue/pipeline的逐周期Golden。** 现有独立 Stage 1～4不能替代模块级RTL映射。
3. **PARTIAL — 无界deque的硬件身份。** 源码能确认它们当前没有容量反压，不能确认目标硬件应有的深度。
4. **UNKNOWN — 真实SRAM端口数、read-during-write规则和macro延迟。** 当前 `Sram` 类不表达这些属性。
5. **PARTIAL — Mxm datapath/Control跨模块同edge的最终硬件采样定义。** CModel调用顺序明确，但仍需RTL或规格确认它是否完全等价。
6. **PARTIAL — VXM缺bundle时应当是编译错误还是未来动态stall。** 当前实现明确采用错误/静态调度；未来架构意图需另行确认。
7. **UNKNOWN — AXI InstructionFetchDMA接入DistributedIcuQueue后，HOLD、IQ credit、burst backpressure与程序切换的组合语义。** 当前只有独立组件测试。
8. **UNKNOWN — 正式统一模型是否需要多时钟域。** Stage 4明确未覆盖多时钟、异步FIFO、负沿触发器、latch和门级延迟（`CPP_TIMING_MODELING.md:55-64`）。

## 最终判定

当前 CModel 已经具备大量局部周期语义，尤其是 Stream Fabric、ICU HOLD/QCTL、C2C/DDR和VXM局部pipeline；它并非“纯功能模型”。但它也尚不是统一正沿提交的 RTL 前仿模型。最合理的下一步不是立刻重写全部 `tick()`，而是先建立统一时间域与trace契约，用正式模块的小范围RTL Golden锁定边界，再按“显式边界优先、调用顺序风险其次、物理容量最后”的顺序迁移。

