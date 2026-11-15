# FTLPU-CMODEL-clock 统一时序架构

## 当前冻结范围

本文件冻结截至 Checkpoint 7B 已经实现并复验的正式架构。正式基础设施位于
`include/ftlpu/core/timing.hpp`，实验目录中的早期调度器仅作为历史参考，不能覆盖
这里的调用权划分。

## 四阶段物理时钟

全局阶段严格按照以下顺序推进：

`LOW -> POSEDGE -> HIGH -> NEGEDGE -> LOW`

`phase_count` 单调递增，`cycle = phase_count / 4`。Clock 维护物理时间和阶段合法性，
所有正式接入的正沿状态共享一个提交边界。

| 阶段 | 唯一负责人 | 行为 |
|---|---|---|
| LOW | SystemPhase | 按显式硬件依赖顺序读取旧 Current、传播组合逻辑并准备 Next |
| POSEDGE | Clock | 先验证全部 ClockedModule，再统一提交；提交期间不计算组合逻辑 |
| HIGH | SystemPhase | 读取全部更新后的 Current，只计算需要的组合观察输出 |
| NEGEDGE | Clock | 只执行真实负沿行为；正沿状态保持 |

Clock 不调用模块的 `low()` 或 `high()`。SystemPhase 不调用
`validate_posedge()`、`posedge()` 或 `negedge()`。Clock 注册顺序只定义提交集合，
不能替代 SystemPhase 的组合拓扑顺序。

## 正式基础设施

- `ClockPhase`：四阶段枚举。
- `ClockParticipant` / `ClockedModule`：时钟参与接口和正沿状态模块接口。
- `Clock`：注册、阶段检查、全局验证/提交屏障、物理周期和 faulted 状态。
- `SystemPhase`：命名的 LOW/HIGH 有序执行节点。
- `ClockedState<T>`：Current/Next、prepare、validate、commit。
- `Register<T>` / `RegisterBank<T,N>`：同步复位、enable、hold 和正沿提交。

## StreamRegisterFabric 试点

StreamRegisterFabric 已接入 `ClockedModule`，但保留专用稀疏
`current_`/`next_` 双缓冲，避免每周期复制完整 Fabric。LOW 打开 staging 窗口，
生产者通过 `stage_write()` 修改 Next；POSEDGE 前检查几何和周期状态，随后交换
Current/Next。East/West 两个 Fabric 由 TspSliceSystem 的同一个 Clock 注册，只有
两个 Fabric 都通过验证后才开始提交。

正式 Checkpoint 1 接口是 `low()`、`validate_posedge()`、`posedge()`。任务说明中
出现的 `begin_cycle()`、`commit_cycle()` 是早期接口名称；本工作区不恢复旧 API。

## TspSliceSystem 当前拓扑

LOW 显式顺序为：

1. Stream Register LOW 准备。
2. ICU dispatch。
3. MXM control 和 datapath。
4. VXM capture、execute 和 route。
5. MEM、SXM、C2C 和 passive-route staging。
6. C2C data DMA、completion notify、共享 DDR 和 ICU barrier progression。

POSEDGE 由 Clock 对 ICU、两个 Fabric、两个 MEM、各 MXM Control/Datapath、VXM Slice、
两个 SXM Slice、已连接 C2C Endpoint、两个 data DMA 和共享 DDR 先统一验证，再开始提交。
HIGH 目前只执行提交后观察和日志。

## 验证边界

Checkpoint 1～6、7A 与 7B 证明了基础框架、StreamRegisterFabric、MEM、当前集成 MXM、
VXM、主系统 `SxmSlice`、ICU、C2C Endpoint/Link 和单芯片 data DMA/DDR 的已定义行为
契约。它没有证明跨芯片共同 barrier 或完整系统已经与 RTL 逐周期等价。

## Checkpoint 2：MEM 接入

`MemArrayModel` 已作为 `ClockedModule` 加入与 East/West
`StreamRegisterFabric` 相同的全局 Clock。SystemPhase 仍按既有拓扑在 LOW 中先打开
Fabric，再执行 ICU/MXM/VXM，随后执行 MEM/SXM/C2C；MEM 的 `evaluate()` 只准备
Next 和局部 SRAM 写集合，Clock 在所有已注册模块验证通过后统一提交。

### MEM 正式边界

| 状态/事件 | LOW | POSEDGE | HIGH |
|---|---|---|---|
| 指令入口 | 查看 Queue front，填入 row0 的 Next 并执行本周期 tile0 行为 | pop 已接受 front；提交移位后的流水 Current | 只读新 Current |
| 指令流水 | 全部级从旧 Current 派生，统一向下一 tile 生成 Next | 整体提交 | 保持 |
| SRAM Read/Main | 读取正沿前 Memory，写 Fabric Next | Fabric 与 MEM 统一提交，输出变为 Current | 可观察 |
| SRAM Read/R1 | 读取正沿前 Memory，写 R1 Next；同时将旧 R1 Current 写 Fabric Next | R1 与 Fabric 各自提交 | 可观察新 R1/Fabric Current |
| SRAM Write | 采样地址、字节、tag 和 transport owner，只进入局部 pending write | 原地更新目标 8-byte segment，并更新 commit counter | 可观察新 Memory |
| MEM cycle | 计算 `Next = Current + 1` | 提交 | 只读 |

因此 `Main` 在请求 LOW 后的同一正沿把数据提交到 Fabric；`RegisteredR1` 先在该正沿
提交 R1，再由下一个 LOW 把旧 R1 Current 送入 Fabric，并在下一正沿可见。两种模式
没有被机械统一。若同一行为周期同时读写相同地址，当前模型读取旧 Memory、写在正沿
提交，即 read-before-write。

`Sram` 继续采用按页稀疏分配，不放入通用 `ClockedState`，避免每周期复制完整 Memory。
现有模型允许每个 bank 的读/写指令端口同时活动，但尚无可靠 RTL 或 SRAM Macro 规格
证明实际端口数、冲突策略、write mask 或物理读延迟；Checkpoint 2 只冻结并验证现有
行为契约，不把它宣称为已完成 RTL 对照的周期 Golden。

## Checkpoint 3：MXM 接入

当前系统顶层实际集成的是 VMODEL 的 legacy `lpu_mxm_slice` 控制结构：Load、Dequant、
Compute 分别进入四级 row pipeline，row0 也是正沿寄存状态。独立存在但尚未接入顶层的
`lpu_mxm_core` 使用另一套更深的 M1～A3/Backend 契约，本 Checkpoint 没有把它的延迟
机械套入当前 CModel。

### MXM 正式边界

| 状态/事件 | LOW | POSEDGE | HIGH |
|---|---|---|---|
| Control Queue front | 只读取 front，把被接受条目写入 row0 Next | pop 已接受 front；row0 变为 Current | 只读 |
| Load/Dequant/Compute row pipeline | 只执行旧 Current；所有 row 从旧 Current 派生下一 row Next | 四级整体提交 | 保持 |
| Supercell Weight/valid | IW 写 Weight/column-valid/buffer-valid Next | 权重和有效位同时可见 | 只读 |
| Supercell Activation pipeline | 组合 MAC 读取旧权重和旧 stage；只向下一 stage Next 推进一次 | stage 整体提交 | 只读 |
| Datapath pipeline/control state | 从 Control Current pulse 和旧 Datapath Current 计算 Next | Datapath Current 统一更新 | 只读 |
| Accumulator | 本 LOW 接受局部 accumulate/clear 操作；同 LOW 的后续 read 看见顺序投影值 | 原地按接受顺序更新目标 segment | 只读新值 |

`MxmControlSlice` 拥有并提交其 `MxmArray`/Supercell 的边沿状态；`Mxm` 单独拥有 Datapath
和 Accumulator 状态。两者都注册到 `TspSliceSystem::Clock`，但 LOW 仍由 SystemPhase
明确按照 `ICU dispatch -> MXM control -> MXM datapath` 执行，因此 Clock 注册顺序不会
替代组合依赖顺序。

Queue 新条目不再在同一个 LOW 中穿过 row0。静态测试调度中的“目标执行周期”因此在
ICU Queue 中提前一个物理周期发射；MEM 数据运输时序保持由 Stream Fabric 的寄存列数
决定。该变化消除了旧 C++ `dispatch -> execute` 零拍旁路，并与当前集成 legacy RTL 的
row0 寄存边界一致。

Accumulator 保留大数组本体，只暂存本 LOW 实际命中的 segment 操作，不复制整个 SRAM。
同 LOW 多次 accumulate/clear 的读取投影与正沿提交顺序一致。当前行为仍是 CModel 行为
契约；VMODEL 只提供了 legacy control/compute 的局部静态证据，尚未建立逐周期 RTL Trace
对照，也不宣称未集成的新 MXM core 延迟已经验证。

## Checkpoint 4：VXM 接入

`VxmSlice` 已作为一个层次化 `ClockedModule` 注册到 TSP 全局 Clock。Slice 在 LOW 开始时
从已提交 Current 建立事务化 Next，并把当拍 ICU 投递的 8 条逻辑控制流、Input Bundle
捕获、Superlane Decoder/Config、Lane ALU/LUT/量化流水和反馈寄存器全部推进到 Next。
只有所有模块通过全局验证屏障后，VXM 才以一次不可抛异常的指针交换提交完整层次状态。

### VXM 正式边界

| 状态/事件 | LOW | POSEDGE | HIGH |
|---|---|---|---|
| ICU local issue | 先进入待投递容器；`prepare_cycle()` 合并到 Slice Next Queue | 与完整 Slice Next 一并成为 Current | 不重复入队 |
| Slice row pipeline | 只从旧 Current/当拍入口派生一次；每拍向下一 tile 推进一级 | 整体提交 | 只读 |
| Decoder/Next/Current Config | Superlane 在事务副本中完成 FIFO→Decoder→Next→Current 和 repeat consume | 作为 Slice 层次状态整体提交 | 不重复 decode/consume |
| Input/Bundle Buffer | 从 Fabric Current 捕获到 VXM Next；静态调度检查在 LOW 完成 | Empty/Collecting/Ready/Release 结果整体提交 | 不重复 capture/release |
| Lane datapath | 旧 token、ALU/LUT/quantizer pipeline、accumulator、feedback 计算下一状态 | 整体提交 | 只读提交后状态 |
| VXM 输出 | LOW 由旧流水完成结果产生并写 Fabric Next | VXM 与 Fabric 在同一全局屏障提交 | 可观察新 Current |
| VXM cycle | LOW 准备 `Next = Current + 1` | 提交 | 保持 |

VXM 仍采用编译器静态调度：Bundle 不完整、配置未及时到达或寄存冲突继续抛出 schedule
错误；本批没有把这些断言改成动态 valid/ready 重试。`VxmLane::Statistics` 是软件观测
状态，事务快照共享它以避免复制不断增长的 timeline；它不被描述成硬件触发器。

本地 VMODEL 提供了局部寄存边界证据：`lpu_vxm_control.sv` 的 8 条 local pipeline 与
global config wave、`lpu_vxm_slice.sv` 的 `tile_execute_wave_q` 一拍寄存，以及
`lpu_vxm_tile_execution.sv` 的 resident config、pending valid、input token、accumulator、
feedback 和 output pending `always_ff` 状态。CModel 保留自身 8 logical Queue、4 tile 和
Superlane compact decoder 组织，没有宣称两者已经建立逐周期等价 Trace。

## Checkpoint 5：SXM 接入

`TspSliceSystem` 中 East/West 两个 `SxmSlice` 已加入同一全局 Clock。ICU 当拍投递先进入
`pending_issues_`；SXM LOW 才从已提交 Current 建立完整 Next、合并投递，并且只在这个
Next 上推进 Transpose/Permute/Distribute/ShiftSelect 和 Wide Transpose 状态。全局验证
屏障通过后，`posedge()` 以一次指针交换提交整个 Slice；HIGH 不再重复消费 Stream、
捕获行、产生扇出或推进 cycle。

### SXM 正式边界

| 状态/事件 | LOW | POSEDGE | HIGH |
|---|---|---|---|
| ICU SXM issue | LOW 前进入 pending；prepare 后拒绝追加 | 随 Slice Next 一并可见 | 不重复 issue |
| Transpose instruction wave | 旧 Current/当拍入口派生四个 tile row Next | 整体推进一格 | 只读 |
| Local Transpose Bank | 从 Fabric Current 捕获到 Bank Next；旧 ready block 可同拍被 Permute 读取/释放 | block、mask、dst、ready cycle 同时提交 | 不重复 capture/release |
| Permute | 读取旧 ready block，在 LOW 写 Fabric Next；释放只写 Bank Next | 释放状态与 Fabric 结果统一提交 | 不重复路由 |
| Fanout delay queue | LOW 发送本周期到期项并生成后续到期项 Next | 延迟队列整体提交 | 不重复发送 |
| UnitGroup/cycle | Distribute/ShiftSelect 等只在事务副本执行一次 | issue 状态清空且 cycle + 1 可见 | 保持 |
| Wide Transpose | job、alignment FIFO、双 Bank SRAM 有效槽和 issue interval 在 Next 中推进 | 整体提交 | 不重复读写 |

本地 VMODEL 的 `lpu_sxm_control.sv` 证明 Permute 是一拍寄存控制，`lpu_sxm_slice.sv`
证明 Transpose Bank、tile ready、pass-through 边界寄存器均在 `always_ff` 更新；Permute
数据选择本身是组合路径。CModel 还支持行式捕获、扇出延迟和 WideStreaming 试验路径，
这些不是与 integrated RTL 逐对象同构的声明。

独立 `SxmMatrixCore/SxmMatrixMemReference` 是三角矩阵/WriteTap 日历参考路径，不是
`TspSliceSystem` 当前实例化的 `SxmSlice`。它继续使用自己的独立 Clock/SystemPhase
包装并纳入 SXM 回归；本 Checkpoint 没有为追求形式统一而改变其固定 P/WriteTap 日历。

## Checkpoint 6：ICU 接入

`InstructionControlUnit` 作为一个层次 `ClockedModule` 注册到 TSP 全局 Clock。SystemPhase
仍在既定的 ICU dispatch 节点调用一次组合/时序准备逻辑；该调用从 Current 建立完整 Next，
在 Next 上推进所有 local Queue、fetch/decode、Macro 展开、QCTL、Barrier 和周期状态。
Clock 先验证全部模块，再以 ICU 状态交换提交；HIGH 不重新 fetch、decode、issue 或推进控制器。

| 状态/事件 | LOW | POSEDGE | HIGH |
|---|---|---|---|
| local iMEM | 作为只读 Memory 被 Current/Next 共享；配置写才 Copy-on-Write | 不逐拍复制程序映像 | 只读 |
| IQ / pending fetch | 从旧 Current 处理响应、发请求并生成 Next | 与 Queue 状态整体提交 | 不重复 fetch/pop |
| Macro frontend | reservoir、parser、DDB、active context 只在 Next 展开一次 | decoder/context 状态整体提交 | 不重复展开 |
| QCTL / Sequencer | pool/cache/frontend 与 Queue 控制状态按显式依赖执行一次 | pool 与全部 Queue 原子可见 | 不重复 retire/issue |
| 外部通知 | LOW 前进入 Pending-event；LOW 中应用到 Next | 事件效果成为 Current | 不重复消费 |
| Barrier/Completion 查询 | 对 Pending-event 做无副作用投影，供 due/ready 组合检查 | 实际 controller 状态提交 | 只读 |
| FU issue | ICU Next 消费 Queue；目标 FU 仍按各自 pending/Next 接口接收 | ICU 与 FU 同一全局验证屏障 | 不重复发射 |

程序映像采用 Copy-on-Write backing，因为 local iMEM 是 Memory 而不是成千上万个 DFF；
Macro DDB 和 active context 属于会随执行改变的硬件状态，因此使用深拷贝事务快照。QCTL
frontend 保存的 shared-pool 指针、MEM transport executor 保存的 ICU 回调，在复制与提交后
都必须重绑定，避免快照继续引用旧对象。

周期外到达的 completion/notify 不允许直接改 Current。为保持原有组合契约，
`common_barrier_due()` 和 `queue_completion_ready()` 可以在不写 Current 的前提下预览
Pending-event；这既保留同拍安全检查，也不破坏统一正沿提交。

instruction-fetch DMA 是外部程序装载路径，未被当作 local Queue fetch；C2C data DMA 与
DDR 也未在本检查点迁移。Checkpoint 6 只证明当前 CModel ICU 的事务边界和回归一致性，
不声明已经建立 RTL fetch/decode/issue 的逐周期 Golden。

## Checkpoint 7A：C2C Endpoint 与 Link 接入

片上 C2C Endpoint 和芯片间 Link 已分别建立 Current/Next。Endpoint 的 TX gather、RX
replay、命令 Queue、active receive 和完成保持状态只在 LOW 准备；TSP 中两个永久
`C2cEndpointClockSlot` 引用可选 Endpoint，在全局 POSEDGE 验证后统一提交。Clock 不调用
Endpoint LOW，组合顺序仍由 `mem_sxm_stage_phase()` 中的显式 `evaluate_c2c()` 决定。

| 状态/事件 | LOW | POSEDGE | HIGH |
|---|---|---|---|
| TX/RX issue | LOW 前进入 pending；prepare 时合并到 Next | Queue Current 可见 | 不重复入队 |
| TX gather / RX replay | 读取 Fabric Current 和旧 Slice Current，推进 Next 一次 | 四 tile pipeline 整体提交 | 保持 |
| Stream consume/write | 只在 `evaluate_c2c()` 执行一次，目标为 Fabric Next | Endpoint/Fabric 同一 TSP barrier 提交 | 不重复副作用 |
| Link send/pop | 记录本周期请求，Link LOW 应用到 Next | serializer/flight/RX FIFO/credit/cycle 原子提交 | 保持 |

`C2cLink::tick()` 仅作为现有测试和 `DualChipC2cSystem` 的兼容包装，内部仍执行
LOW→validate→commit。当前 Dual-chip 顶层尚未把两个 TSP Clock 和四条 Link 纳入一个共同
验证屏障，因此本检查点只证明各对象内部不提前暴露 Next，不声称完成跨芯片原子提交。

`C2cDmaEngine`、`Ddr4Model` 仍使用旧 `tick()` 边界。Checkpoint 6 已使 completion 到
ICU 的通知进入 Pending-event/Next，但这不等于 DMA/DDR 内部状态已经迁移。instruction-load
DMA 与这里的 C2C 数据 DMA 是不同路径，也未借用前者的 RTL 或接口宣称后者已经完成。

## Checkpoint 7B：C2C Data DMA 与 DDR4 接入

`C2cDmaEngine` 与 `Ddr4Model` 现在都是 `ClockedModule`。它们保留独立 `tick()` 作为单元
测试兼容包装；在 `C2cDmaSystem` 中则由 TSP 的永久 Clock slot 统一验证和提交，禁止再在
芯片周期之后额外推进 DDR。

| 状态/事件 | LOW | POSEDGE | HIGH |
|---|---|---|---|
| Endpoint→DMA send/pop | Endpoint evaluate 暂存输入，不改 DMA Current | 随 DMA Next 可见 | 不重复握手 |
| DMA descriptor/active/FIFO | 从 Current 和 pending input 生成 Next，处理旧 DDR completion | 与 ICU/FU/DDR 一起提交 | 保持 |
| DMA completion→ICU | 从 DMA Next 取 completion token，通知已准备的 ICU Next | DMA completion 和 ICU event 同时可见 | 不重复 notify |
| DDR request/pop | ProgramLoader/data DMA 在 DDR LOW 前暂存 | request/completion FIFO 状态提交 | 保持 |
| DDR beat/latency/bandwidth | 在 DDR Next 中服务一次，局部记录 Memory write | counter/completion 与目标 bytes 同时提交 | 只读新状态 |
| DDR 稀疏 Memory | 读取已提交页；同 LOW 后续读可按顺序看到 staged write 投影 | 只写本拍命中的 byte，不复制完整 Memory | 可观察新值 |

SystemPhase 的依赖顺序是 Endpoint interaction→East/West data DMA LOW→DMA completion
通知 ICU Next→共享 DDR LOW→ICU barrier advance。Clock 的 slot 注册顺序只定义共同提交
集合，不替代上述组合依赖。ProgramLoader 内部 instruction-load DMA 仍是独立软件路径，
但它在 TSP 之前产生的 DDR 请求与 data DMA 请求由同一个 DDR LOW 和同一个 POSEDGE 接受。

当前边界仍止于单芯片 TSP。`DualChipC2cSystem` 的两个 TSP 与 Link 尚未合并为一个全局
验证/提交屏障，这是 Checkpoint 7C 的范围。

