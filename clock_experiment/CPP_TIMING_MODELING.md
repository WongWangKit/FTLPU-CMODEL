# C++ 硬件时序行为建模参考

本文件记录 `clock_experiment` 中已经由独立 RTL Golden 实验验证的方法，以及尚未覆盖的边界。长期参考实现位于 `cpp_timing_modeling.hpp`。

## Current / Next 与统一提交

- 组合求值只读取 Current，并在正沿前准备 Next。
- Phase 1 只执行统一 Commit；Commit 不重新计算组合逻辑，也不读取其他模块刚更新的 Current。
- 模块调用顺序可以改变，但只要 evaluate 全部读取旧 Current、commit 只复制本模块 Next，最终 Trace 就不应依赖调用顺序。
- `CurrentNext<State>` 是最小参考容器；正式模型可以使用模块自己的 Current/Next 类型，不要求继承该类。

这些原则已经在 Stage 1、Stage 2、Stage 3 中通过 C++/RTL 对照验证。

## 组合逻辑级联建模

组合逻辑节点不是寄存器。`MUL0 → ADD0 → MUL1 → ADD1 → MUL2 → ADD2` 必须在一次函数调用内按数据依赖顺序求值：

```cpp
const auto mac = ftlpu::timing::evaluate_mac(input);
```

每一级函数读取前一级的实际返回值，不保存跨周期状态，不修改 Current/Next，也不推进时钟。组合级数只影响真实硬件的门级传播路径；只有显式寄存器边界才增加周期延迟。

Stage 4 使用 16-bit 无符号中间值，并把 `result16` 的低 8 bit 写入下游 FIFO。MAC 输入来自 FIFO A 的 Current `rd_data`，而不是 FIFO A 的 Next。

## 四阶段时钟模型

`FourPhaseClock` 使用单调递增的 `phase_count`：

| Phase | 时钟 | 允许的行为 |
|---:|---:|---|
| 0 LOW_EVALUATE | 0 | 读取 Current、求组合逻辑、准备所有 Next、计算正沿事件 |
| 1 POSEDGE_COMMIT | 1 | 所有正沿状态统一 Commit，不重新求值 |
| 2 POST_EDGE_SETTLE | 1 | 读取更新后的 Current，重新求纯组合输出；不得修改 Next |
| 3 NEGEDGE | 0 | DUT 正沿寄存器保持；允许 Testbench Producer 驱动器完成已接受事务的指针推进 |

`cycle = phase_count >> 2`。调用方应使用 `require()` / `advance_from()` 拒绝跳相、重复 Commit 或在错误阶段准备 Next。

### Pre-edge 与 Post-edge

Phase 0 的 MAC 输出基于正沿前 Current。Phase 1 完成寄存器提交后，Phase 2 再读取新的 FIFO A `rd_data/rd_data_valid` 并重新求 MAC。Phase 2 的求值只产生观察信号，不调用 FIFO evaluate、prepare-next 或 commit。

同步 FIFO A 在 Cycle N 正沿产生读响应时，Phase 2 已能看到新的组合结果；FIFO B 仍在 Cycle N+1 正沿采样该结果。因此 A→B 是 1 cycle，而 MAC 本身增加 0 cycle。

## 已验证与未覆盖范围

已经由 RTL Golden 验证：

- 两级流水线、同步 FIFO、双 FIFO反压与 Producer valid/ready。
- 四种 FIFO evaluate/commit 调用顺序不改变结果。
- Stage 4 的四阶段顺序、Post-edge 组合重算、Next 保持、Negedge 状态保持。
- MAC 256 种 8-bit 输入的所有中间节点与最终输出。
- 双 FIFO + MAC 场景的逐阶段 C++/RTL 一致性和端到端顺序。

设计建议但尚未由本实验覆盖：

- 将同一调度接口扩展到完整 ICU/FU 模型。
- 为大规模模型构建集中式模块注册和统一时钟调度器。

尚未覆盖：

- 多时钟域、异步 FIFO、负沿触发寄存器和 Latch。
- 门级传播延迟、建立保持时间、X 传播的完整物理语义。
- 正式 FTLPU-CMODEL 全系统的周期精确性。

## 独立实验中的面向对象时序框架

`ftlpu::timing::framework` 在保留上述 API 的前提下新增以下职责分层：

- `Clock`：非 owning 注册时钟参与者，按 LOW、POSEDGE、HIGH、NEGEDGE 调度；仿真开始后冻结注册表，参与者抛出异常后永久进入 faulted 状态。
- `ClockParticipant`：各阶段均有默认空实现的虚接口。
- `ClockedModule`：要求时序模块实现 LOW、提交前检查与 POSEDGE。
- `ClockedState<T>`：保存 Current/Next；`prepare()` 默认保持，`next_value()` 只开放 Next，`validate_commit()` 构成提交屏障，`commit()` 每次只允许一次。
- `Register<T>` / `RegisterBank<T,N>`：同步高电平复位、enable 保持与正沿提交。
- `SyncFifo<T,Depth>`：复用 Stage 2 的 Empty/Full 同周期读写、同步读输出及 Memory 保持契约。
- `SyncSram<T,Depth>`：单端口同步读写；Memory 独立保存，仅输出寄存器使用 `ClockedState`，避免每周期复制完整 SRAM。

`Clock` 在 POSEDGE 先遍历所有参与者执行 `validate_posedge()`，全部通过后才进入提交遍历。这只能避免“可预见的未准备 Next”导致部分提交；若类型 `T` 的复制赋值本身会抛出异常，Clock 会 fault，但已经完成的提交无法通用回滚。因此框架状态类型应优先使用 `noexcept` 赋值类型，正式模型也不应把异常当作正常控制流。

### 组合拓扑与提交顺序

LOW/HIGH 的调用顺序表达组合逻辑拓扑，POSEDGE 的调用顺序只提交已准备好的状态，两者语义不同。验证顶层显式采用 `Source Current → evaluate_mac() → Destination Next`，并用正序、逆序两套 Commit 顺序执行同一激励。该做法保持调度可审计，没有引入通用图求解器。

Phase 2 只允许根据新 Current 更新普通组合观察值；验证会在进入 HIGH 前保存全部 Current、Next 和 Memory，并检查 HIGH/NEGEDGE 后这些时序状态完全不变。

### SRAM 行为边界

当前 `SyncSram` 只代表一个基础单端口、同步读、同步写、寄存器读输出模型。无请求或写请求时读数据保持、valid 清零；复位清理读输出但不清 Memory/initialized。Read-during-write、写掩码、Byte enable、多端口、可配置读延迟和多时钟均未建模，后续应通过明确策略扩展，而不是塞进普通 Register 模板。

## 框架验证状态

由本次独立 RTL Golden 对照覆盖（以 `clocked_module_framework` 的比较日志为准）：

- Register、RegisterBank、无寄存器 MAC 级联、同步 FIFO、单端口同步 SRAM。
- 四阶段计数与采样边界、正沿提交、Post-edge 组合重算、Phase 2/3 时序状态保持。
- FIFO Full/Empty、同周期请求、指针回绕、同步读、复位和 SRAM 未初始化/保持行为。
- 正序与逆序 Commit 得到相同状态。

仅由 C++ 自测覆盖：

- 重复注册、启动后注册、非法阶段、参与者异常后的 faulted Clock。
- 未准备或重复 `ClockedState` Commit 的拒绝。

设计建议、尚未验证：

- 正式 ICU/FU 模块的迁移策略、模块分组与更大规模拓扑调度。
- 真实 SRAM Macro 的端口策略以及大型模型的性能数据。

该 PASS 只说明当前框架在本目录的 Register/FIFO/SRAM/组合逻辑场景满足定义的周期契约，不代表正式 FTLPU-CMODEL 已完成统一时序化。

## 正式 CModel Checkpoint 1（2026-11-14）

正式接口位于 `include/ftlpu/core/timing.hpp`，命名空间为
`ftlpu::timing`。正式实现没有把独立实验中的调度方式原样复制到系统中，
而是把物理时钟边界和组合逻辑拓扑明确拆成两套并行机制：

| 阶段 | 唯一调用者 | 正式职责 |
|---|---|---|
| LOW | `SystemPhase` | 按显式 A→B→C 拓扑读取旧 Current、传播组合逻辑、准备 Next，以及暂时承载尚未迁移模块的一次性旧行为 |
| POSEDGE | `Clock` | 先对全部已注册 `ClockedModule` 执行 `validate_posedge()`，全部通过后再统一 `posedge()` Commit |
| HIGH | `SystemPhase` | 按显式拓扑读取全部更新后的 Current，只更新组合观察结果 |
| NEGEDGE | `Clock` | 只调用真实下降沿行为；没有负沿状态的模块保持 |

`Clock` 不调用模块的 `low()` 或 `high()`；`SystemPhase` 也不执行
`validate_posedge()`、`posedge()` 或 `negedge()`。同一硬件对象可以同时出现在
Clock 的非 owning 注册表和 SystemPhase 的有序节点中，但每个阶段行为只能有一个
调用者。这一约束避免把 Clock 注册顺序误当成组合拓扑，也避免重复入队、出队、
写 SRAM 或推进状态机。

正式框架当前提供：

- `ClockPhase`、`ClockParticipant`、`ClockedModule`、`Clock` 和 `SystemPhase`。
- `ClockedState<T>`、`Register<T>` 与 `RegisterBank<T,N>`。
- 重复注册、启动后注册、非法阶段、未准备/重复 Commit 和 faulted 状态检查。
- 全局 `phase_count` 与 `cycle = phase_count / 4`；一个完整 `tick()` 严格经过
  LOW→POSEDGE→HIGH→NEGEDGE。

### StreamRegisterFabric 试点

`StreamRegisterFabric` 是第一个接入正式框架的系统模块：

- 实现 `ClockedModule`，LOW 打开一次 staging 窗口，POSEDGE 提交。
- 保留原有稀疏 `current_` / `next_` 双缓冲，避免把完整 SR Fabric 套入通用
  `ClockedState<T>` 后产生无意义的大对象复制。
- 两个 Hemisphere Fabric 都由 `TspSliceSystem` 的同一个 Clock 注册。
- POSEDGE 先验证两个 Hemisphere，再统一提交，因此不再存在一个 Hemisphere 已提交、
  另一个在可预见验证错误上失败的部分更新。
- Fabric 自身的 `cycle()` 只是诊断提交计数；在正式系统中由全局 Clock 唯一提交，
  因而与物理 cycle 保持同步。

`TspSliceSystem` 的 LOW 拓扑当前显式固定为：Stream Register 准备、ICU dispatch、
MXM control/datapath、VXM capture/execute/route、MEM/SXM/C2C staging、未迁移模块的
一次性周期末推进。HIGH 目前只做提交后观察和日志，不准备 Next，也不重复执行 FU、
队列或存储副作用。

### Checkpoint 1 验证证据和边界

新增专项测试覆盖：

- Source Register→Combinational A→Combinational B→Destination Register。
- LOW 按 A→B 执行，Destination 从旧 Current 派生 Next。
- POSEDGE 统一提交，HIGH 重新读取新 Current 且不修改 Next。
- 正序/逆序 Clock 注册得到相同结果，SystemPhase 拓扑不受注册顺序影响。
- RegisterBank enable/同步复位和一次性 LOW 副作用。
- 双 Hemisphere Stream Fabric 的统一验证/提交屏障及提交失败前无部分更新。
- `TspSliceSystem` 每个 tick 的四阶段计数及两个 Fabric 的 cycle 锁步。

当前正式范围只包括基础框架、Register/RegisterBank、StreamRegisterFabric 试点和
`TspSliceSystem` 顶层阶段接线。MEM、MXM、VXM、SXM、ICU 内部状态尚未逐模块改写为
Current/Next；它们的旧一次性周期行为暂时位于 SystemPhase LOW。此检查点通过不等于
全部 FU 已完成统一时序重构，也不证明所有模块可以不经模块契约审计而机械批量迁移。

## 正式 CModel Checkpoint 2：MEM（2026-11-14）

MEM 正式迁移继续复用本文件定义的 Clock/SystemPhase 分工，没有增加第二套调度器。
`MemArrayModel` 现在是 `ClockedModule`：指令流水、R1 读响应和周期计数使用
`ClockedState`；大型 SRAM 仍是稀疏专用 Memory，只在 LOW 暂存实际被接受的局部写，
在统一 POSEDGE 原地更新目标 segment。

正式行为契约为：

- Queue front 在 LOW 被接受并执行 row0 行为，但只有 POSEDGE 才 pop；流水 Next 在同一
  正沿提交到下一 tile，避免同一 LOW 穿越多级。
- Main Read 在 LOW 读取旧 SRAM 并写 Fabric Next；RegisteredR1 在 LOW 写 R1 Next，
  下一 LOW 才把旧 R1 Current 写入 Fabric Next。
- Write 在 LOW 只采样输入和校验地址，SRAM、transport commit counter 均在 POSEDGE
  更新；同周期 Read/Write 因此表现为 read-before-write。
- HIGH 不准备 Next、不写 SRAM、不 pop Queue，也不重复发射。

新增专项测试以独立的逐周期期望检查指令只推进一格、写入只在边沿可见、R1 的两阶段
响应、Clock 注册顺序无关，以及另一模块验证失败时 MEM/Fabric 都不发生部分提交。
MEM 8 个单元测试和相关系统回归通过；没有可靠 RTL 或 Macro 规格，因此真实 SRAM
端口数、冲突规则和物理读延迟仍是待确认项，当前 PASS 不等价于 RTL 周期 Golden。

## 正式 CModel Checkpoint 3：MXM（2026-11-14）

MXM 继续采用同一套 Clock/SystemPhase，不增加局部第二时钟。当前集成 legacy MXM 的
Control row0 和后续三行被建模为正沿寄存器：LOW 只执行旧 Current，新 Queue front
写 row0 Next，统一 POSEDGE 后才可见和 pop。Control 拥有的 Supercell Weight/valid 与
Activation stage 同样只在正沿提交。

Datapath LOW 读取 Control Current pulse 和旧 Datapath Current，计算 WS/IS pipeline、
cursor、valid/owner 和输出 Next。Accumulator 不用通用大对象复制，而是在 LOW 保存
实际命中的 segment accumulate/clear 序列；同 LOW read 看到顺序投影，POSEDGE 才更新
物理数组。MXM Control 和 Datapath 都注册到 Tsp 的全局 Clock，SystemPhase 仍显式固定
`ICU dispatch -> MXM control -> MXM datapath` 的组合调用顺序。

专项测试验证了 row0 不提前可见、四级每正沿只推进一级、Weight 与 Accumulator 的边沿
提交、正反 Clock 注册顺序等价，以及后续模块验证失败前不发生部分提交。MXM 9/9 和
fast 排除已知 RMSNorm 后 61/61 通过。当前证据对应已集成 legacy `lpu_mxm_slice`；未
集成 `lpu_mxm_core` 的深流水/Backend 契约未被宣称为已验证，完整 RTL 逐周期 Golden
仍然缺失。

## 正式 CModel Checkpoint 4：VXM（2026-11-14）

VXM 继续复用唯一的 Clock/SystemPhase。`VxmSlice` 在 LOW 开始时建立完整层次 Next：
当拍 ICU local issue、4-row control wave、Superlane FIFO/Decoder/Next/Current Config、
repeat、Input Bundle，以及 Lane 的 ALU/LUT/quantizer/accumulator/feedback 都在这个事务
副本上求值。LOW 失败只丢弃/保留未提交 Next，不改变已提交 Current；全局验证通过后
以一次状态交换完成 POSEDGE 提交。

SystemPhase 的拓扑仍为 ICU dispatch→VXM prepare/capture/evaluate→VXM output route，VXM
输出只写 Stream Fabric Next。HIGH 不再调用旧 `tick()`，因此不会重复 capture、decode、
execute、Bundle release 或 Queue 消费。直接 VXM 单元测试仍可用兼容 `tick()`，其内部
严格执行 prepare/evaluate/validate/commit 一个完整边沿。

VXM 保留编译器静态调度契约：Bundle 未准备、配置未及时到达和固定寄存冲突仍是错误，
不是动态 backpressure。软件 Statistics timeline 不作为触发器状态复制。专项测试验证了
边沿可见性、Clock 注册顺序无关、验证失败无半提交和失败 LOW 不污染 Current。VXM
14/14、fast 排除冻结 RMSNorm 62/62 通过；RMSNorm 仍为既有 active-bundle 错误。

本地 VMODEL 只用于确认寄存状态种类和 config/execute wave 边界。CModel Superlane compact
decoder 与 integrated RTL resident-config 结构并非完全同构，所以当前结论仍是行为契约
验证，不是逐周期 RTL Golden。

## 正式 CModel Checkpoint 5：SXM（2026-11-14）

TSP 主系统的两个 `SxmSlice` 已加入唯一的全局 Clock。ICU 投递先进入待处理入口，SXM
LOW 才从 Current 建立完整 Next 并合并本拍命令；Local Transpose instruction wave、
Transpose Bank 与占用元数据、Permute、fanout 延迟项、UnitGroup cycle，以及 Wide
Transpose 的 job/FIFO/双 Bank payload 全部只在这个事务副本上推进。

这种边界保留了关键的同拍语义：Permute 读取旧的 ready block，同时 Transpose capture
可以在 Next 中复用被释放的 Bank；两者不会因为 C++ 调用顺序而提前改写 Current。
Stream consume 和输出 staging 只在 LOW 执行一次，所有模块验证通过后才一起提交；HIGH
不再重复捕获、释放、路由或推进 cycle。

专项测试证明 SXM 和 Fabric 在正沿前后可见性正确、Clock 注册顺序不影响结果、HIGH
不重复推进，且后续模块验证失败时两者都不会部分提交。SXM 标签 10/10、fast 排除冻结
RMSNorm 63/63 通过；RMSNorm 与 slow 失败类别均保持 Checkpoint 4 基线。

独立 `SxmMatrixCore/SxmMatrixMemReference` 仍是三角矩阵/WriteTap 日历参考路径，不是
`TspSliceSystem` 当前使用的主 Slice。本批保留其独立 Clock/SystemPhase 和固定日历，
相关专项测试继续通过；没有为追求表面统一而改动它的执行语义。
