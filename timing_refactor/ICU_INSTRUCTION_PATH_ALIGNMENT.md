# FTLPU-CMODEL ICU 指令链路与本地 VMODEL 对齐

## 1. 结论与范围

本轮以 `D:\LPU\FTLPU-VMODEL-main` 当前磁盘工作树为 RTL
依据，不以 GitHub main 或 HEAD 的干净内容替代未提交/未跟踪文件。冻结版本：

- branch：`rtl-handshake-naming`
- HEAD：`2cbaa104090c6df099894559794c0f981d4aafda`
- status：116 modified、10 deleted、55 untracked

VMODEL 顶层有两种互斥的 ICU 输入模式：

1. 默认 `ENABLE_DISTRIBUTED_PROGRAM_FRONTEND=0`：138 条 legacy
   Schedule Queue，由 `lpu_icu/lpu_icu_queue` 执行基础指令、NOP 和
   Repeat。
2. 测试显式打开 `ENABLE_DISTRIBUTED_PROGRAM_FRONTEND=1`：
   ProgramLoad→分发→FU-local iMEM→Macro Frontend→timed issue。

本轮 CModel 的主要同步范围是第二条 **MEM/MXM distributed
ProgramLoad/Macro 路径**。VXM/SXM 的 Program boundary 虽已在当前 RTL
顶层实例化，但 `ready=0`，standalone Macro frontend 没有接到该路径。
CModel 原有 VXM/SXM、Native Queue、NOP/Repeat、QCTL 和 FU datapath
功能均保留，不因 RTL 未贯通而删除。

## 2. VMODEL 当前真实指令链路

### 2.1 Legacy Schedule 模式

```text
schedule_valid/queue/is_instruction/command/payload
                  |
                  v
       lpu_icu (138 queues)
                  |
       lpu_icu_queue depth 16
       instruction / NOP / Repeat
       MEM Repeat address stride
                  |
                  v
       legacy_dispatch_valid/payload
                  |
                  v
       lpu_fu_subsystem_cluster external issue adapters
```

证据：

- 模式选择及 legacy ICU 实例：
  `rtl/lpu_top.sv:1-10,178-220`。
- Queue 异构宽度与 138 个实例：
  `rtl/icu/lpu_icu.sv:1-130`。
- NOP、Repeat、interval、MEM stride、运行期 refill 和满队列同拍替换：
  `rtl/icu/lpu_icu_queue.sv:1-190`。

该模式不是 ProgramLoad/local-iMEM/Macro 路径。分布式模式打开时，legacy
`run` 和 Schedule enqueue 被显式关断。

### 2.2 Distributed ProgramLoad 模式

```text
Host command writes / start
          |
          v
command RAM 32 + ProgramLoad controller
          |
          v
config FIFO 16 + Instruction DMA
outstanding ID FIFO 16 + RX FIFO 16
          |
          v
64-byte reassembler
          |
          v
Program Word FIFO 8
          |
          v
PRIME FIFO 32 / write arbitration
          |
          v
distribution ingress FIFO 2
          |
          +--> VXM direct boundary (currently ready=0)
          |
          +--> East relay chain: 7 stages x depth 2
          |        +--> MEM local endpoints
          |        `--> R6 MXM/SXM boundary
          |
          `--> West relay chain: 7 stages x depth 2
                   +--> MEM local endpoints
                   `--> R6 MXM/SXM boundary
```

关键事实：

- `lpu_top` 只在 distributed 参数打开时实例化
  `lpu_program_load_system`，并把 record 接入
  `lpu_fu_subsystem_cluster`：
  `rtl/lpu_top.sv:275-430`。
- 当前实际 filelist 是
  `sim/filelist_program_load_system.f`；ProgramLoad 替换文件大多是
  当前脏工作树中的 untracked 文件。
- Descriptor validation 当前只接受 MEM、MXM Load、MXM Dequant、
  MXM Compute：`rtl/program_load/lpu_program_load_system.sv:183-238`。
- BOTH scope 在 distributor 中维护 East/West 独立完成状态；
  relay 的 ready 只看 registered occupancy，切断跨级组合 ready：
  `rtl/program_load/lpu_program_distributor.sv:94-160`，
  `rtl/program_load/lpu_program_distribute_relay.sv:149-250`。

### 2.3 MEM local frontend

```text
Program endpoint
  -> local iMEM 131072 x 96
  -> two-stage fetch
  -> response FIFO 4
  -> word FIFO 4
  -> 3-word reservoir + dictionary 7
  -> context FIFO 4
  -> scheduler ingress + active contexts 4
  -> timed bank queue 16
  -> MEM issue
```

每个 physical MEM bank 一套，共 208 套。证据：

- `rtl/mem/lpu_mem_subsystem.sv:130-184`
- `rtl/mem/lpu_mem_local_imem.sv:35-117`
- `rtl/mem/lpu_mem_macro_frontend.sv:76-178`
- `rtl/mem/lpu_mem_macro_scheduler.sv:65-88,299-340`
- `rtl/mem/lpu_mem_bank_queue.sv:19-65`

### 2.4 MXM local frontends

每个 MXM 有 Load、Dequant、Compute 三条独立 stream，共 4×3=12 套：

```text
per-stream Program endpoint
  -> local iMEM 65536 x 128
  -> two-stage fetch + response FIFO 4
  -> word FIFO 4
  -> stream-specific decoder, 3-word reservoir, dictionary 7
  -> context FIFO 4
  -> active contexts 4
  -> timed queue 16
  -> local coordinator
  -> MXM FU boundary
```

三类 context/issue 编码不同，但存储级数和深度相同。local coordinator 负责
Load/Dequant 配对和 Compute 顺序约束。证据：

- `rtl/mxm/lpu_mxm_subsystem.sv:131-290`
- `rtl/mxm/lpu_mxm_local_imem.sv:35-117`
- `rtl/mxm/lpu_mxm_*_macro_frontend.sv`
- `rtl/mxm/lpu_mxm_local_coordinator.sv`

## 3. CModel 修改后的实际链路

```text
ProgramImage list
      |
      v
ProgramLoader ClockedModule
      |
      v
C2cDmaEngine + shared DDR model
32-byte physical vectors, finite credits
      |
      v
64-byte reassembly
      |
      v
word FIFO 8
      |
      v
ingress FIFO 2
      |
      v
selected hemisphere relay[0..6], depth 2 each
one registered hop per physical cycle
      |
      v
staged ICU program write -> POSEDGE commit
      |
      v
FU-local iMEM
      |
      v
two-cycle request/return + response FIFO 4
      |
      v
word FIFO 4 -> generic Macro parser/reservoir
      |
      v
CModel packed DDB 8 (CModel-only internal boundary)
      |
      v
decoded context FIFO 4
      |
      v
active contexts 4
      |
      v
timed issue FIFO 16, C+2 lookahead
      |
      v
existing MEM/MXM FU accept path
```

### 3.1 ProgramLoader 和统一时钟

- `ProgramLoader` 已成为 `ClockedModule`，状态使用 Snapshot
  Current/Next；Program reset、generation、load、seal、frontend tick 等
  操作先进入 staged ICU operation，不在调用点直接修改 ICU Current。
- ProgramLoader control LOW、instruction DMA LOW、data DMA LOW、DDR LOW
  都在 `TspSliceSystem` 的显式 SystemPhase 顺序中执行；所有参与模块在
  同一个 Clock validate/commit barrier 提交。
- ProgramLoader 不再在 `C2cDmaSystem::tick` 中私自推进第二次。
- 空任务可以在合法的 C+1 边沿开始并完成，但不会出现 LOW 中提前可见。

证据：

- `include/ftlpu/system/program_loader.hpp:123-203,491-555`
- `include/ftlpu/system/tsp_slice_system.hpp`
- `include/ftlpu/system/c2c_dma_system.hpp`
- `include/ftlpu/system/icu.hpp`

### 3.2 ProgramLoad distribution

- CModel 新增显式 8-entry word FIFO、2-entry ingress FIFO、每半区
  7×2 relay buffers。
- relay ready 使用 LOW 开始时 registered occupancy；满时即使本拍 dequeue
  也不 replacement enqueue，与 RTL relay 的切断 ready 链语义一致。
- 每周期每条记录最多前进一个 relay；MEM 按 bank group 停在近端 relay，
  MXM 走到 R6/tail。
- 所有 FIFO 有 peak/full/write/read 统计，并由 ProgramLoader test 检查
  word、ingress、relay 的写读守恒。

证据：`include/ftlpu/system/program_loader.hpp:47-53,568-651`。

限制：

- ProgramImage 仍按 image 串行，不实现 RTL command RAM/PC。
- 没有 BOTH record 的两分支复制与独立完成。
- 没有 endpoint round-robin、VXM direct、PRIME FIFO 的逐对象模型。
- 当前 instruction load 继续复用 C2C DMA 行为模型，不等同于 RTL
  `lpu_instruction_dma` 的 ID/RX FIFO 协议。

### 3.3 local iMEM / Macro / timed issue

- fetch latency 从 1 修正为 2；加入 response FIFO 4 和 word FIFO 4，
  空 FIFO 不做写到读旁路。
- MEM/MXM 新增 decoded context FIFO 4；新 decoder context 先进入 FIFO，
  下一周期才可 admission 到 active context。
- MEM/MXM active context 都改为 4。
- 新增 timed issue FIFO 16。scheduler 在目标周期前 2 拍形成 entry，
  只有到 due cycle 才可向 FU issue。
- PRIME 需要覆盖 fetch、context admission 和 timed-issue lookahead，当前
  raw Macro 测试的实际 prime 为 20 cycles。
- VXM/SXM 没有强制套用 MEM/MXM timed queue；保持原 CModel 功能并标为
  CModel-only / RTL Not Integrated。

证据：

- `config/ftlpu-lpu32.json:4-39`
- `include/ftlpu/icu/macro_bitstream_decoder.hpp:922-1054`
- `include/ftlpu/icu/distributed_queue.hpp:1972-2050,2730-2792`

## 4. 模块映射

| 功能阶段 | 本地 VMODEL RTL | CModel C++ | 映射 | 当前判断 |
|---|---|---|---|---|
| Program API | command RAM + controller | `ProgramImage` + `ProgramLoader::submit` | Merged/Different | 保留高层 API；没有命令 PC/WAIT/JUMP |
| Instruction DMA | `lpu_instruction_dma` | `C2cDmaEngine` | Partial/Different | 都有限 credit；协议/ID/RX 结构不同 |
| Reassembly | 64 B reassembler | 64 B reassembly | Partial | 容量对应，握手细节不同 |
| Word FIFO | 8×164b | bounded 8-entry semantic FIFO | Partial | 深度和反压对应 |
| Distribution ingress | 2×165b elastic | bounded 2-entry semantic FIFO | Partial/Direct | 逐拍边界对应 |
| Scope/BOTH | distributor branch done | 无 | Missing | 尚未对齐 |
| Relay chain | East/West 7×depth2 | East/West 7×depth2 | Partial | hop/ready/depth对应；无多源仲裁 |
| Endpoint arbitration | RR + commit-valid | single active image endpoint | Missing/Merged | 尚未对齐 |
| local iMEM | MEM96/MXM128 finite SRAM | finite local program backing | Partial | 容量/宽度对应 |
| fetch response | two-stage + FIFO4 | two-cycle + FIFO4 | Direct/Partial | 共同事件时序已对齐 |
| decoder word FIFO | depth4 | depth4 | Direct/Partial | 有限反压对应 |
| Macro decoder | FU-specific RTL | generic decoder + FU codec | Merged/Partial | 语义保留；内部状态不完全同构 |
| context FIFO | depth4 | depth4 | Direct/Partial | MEM/MXM common path |
| active contexts | 4 | 4 | Direct/Partial | MEM/MXM common path |
| timed issue | depth16 | depth16 | Direct/Partial | due-cycle/occupancy对应 |
| legacy NOP/Repeat | `lpu_icu_queue` | Native queue controls | Different/Compatibility | 功能保留，未纳入 distributed Golden |
| VXM Program Macro | boundary ready=0；standalone source | Native queue可用，Macro image拒绝 | Not Integrated/Different | 不纳入 Golden |
| SXM Program Macro | boundary ready=0；standalone source | CModel Macro可用 | RTL Not Integrated/CModel-only | 不纳入 Golden |
| QCTL | shared pool/controller/fabric | shared pool/frontend | Partial | 保留 CP6 功能，本轮未重写 |

## 5. 周期契约

### 5.1 通用 FIFO

`RtlSynchronousFifo<T>` 复现 `lpu_queue_fifo`：

- output/head 是 Current 的组合视图；
- empty 时 enqueue 数据不能同拍 dequeue；
- 通用 FIFO 满时若 Current head 同拍成功 dequeue，可 replacement enqueue；
- 成功读写同时发生时两指针都前进，occupancy 不变；
- 所有写、指针和 occupancy 只在共同 POSEDGE 后成为 Current。

relay 使用 `full_replacement=false`：ready 只看 registered occupancy，满时
同拍 dequeue 不能接受新 entry。

### 5.2 ClockPhase / SystemPhase

- LOW：SystemPhase 按 ProgramLoader control→DMA→DDR→ICU→FU 的依赖顺序
  准备 Next。
- POSEDGE：Clock 先 validate_all，再 commit_all；失败时不允许部分提交。
- HIGH：只观察新 Current，不重复 enqueue/dequeue/issue/SRAM write。
- NEGEDGE：本轮 ICU 新结构无额外负沿状态。

## 6. 验证结果

### 6.1 CModel

定向 Release 测试全部 PASS：

- `unit.icu.rtl_fifo`
- `unit.icu.macro_schedule`
- `unit.icu.raw_macro_decoder`
- `subsystem.program.loader`
- `unit.icu.clocked`
- `subsystem.system.timing`

关键结果：

- raw Macro：`prime_cycles=20`，MEM context stall 被实际覆盖。
- configured ProgramLoad：169 cycles。
- Program word/ingress/relay write/read 守恒检查通过。
- 全量 CTest：70/77 PASS。
- 7 个失败与 Checkpoint 4～7B 冻结基线相同：4 个 MXM accumulator
  配置容量、1 个 MEM SRAM 配置容量、2 个 VXM active-bundle。
  ICU、ProgramLoader、C2C、MEM/MXM/SXM/VXM 单元和 timing 测试没有新增失败。

### 6.2 RTL 与跨模型 Golden

直接使用本地 VMODEL 源码、ModelSim 10.1c，输出只写入 CModel
`out/icu_rtl_alignment/rtl_golden`：

| 验证 | 结果 |
|---|---|
| `lpu_queue_fifo_tb` | PASS |
| 相同激励 FIFO RTL/CModel，12 cycles×17 fields | PASS，逐字段一致 |
| legacy ICU Queue NOP/Repeat/MEM stride | PASS |
| Program config FIFO | PASS，5 checks |
| Program reassembler | PASS，7 checks |
| Program elastic stage | PASS |
| 7-stage relay chain | PASS，80 checks |
| MEM Macro frontend | PASS，2 scenarios |
| MXM three Macro frontends | PASS |

跨模型 FIFO trace 明确区分：

- 正沿前 input/ready/valid/fire/head/level/pointer；
- 正沿后 level/pointer/valid/head。

文件：

- `timing_refactor/rtl_golden/queue_fifo_trace_tb.sv`
- `tests/unit/icu/rtl_fifo_golden_trace.cpp`
- `timing_refactor/rtl_golden/compare_queue_fifo_traces.ps1`

### 6.3 完整 RTL 集群限制

`sim/filelist_frontend_cluster_top.f` 的完整编译已实际尝试。ModelSim
10.1c 无法解析当前完整 core 使用的较新 SystemVerilog 语法，因此没有获得
ProgramLoad→cluster 的完整运行 Trace。单独的当前 ProgramLoad、relay、
MEM/MXM Macro frontend 模块均能编译并通过自检。

因此：

- FIFO primitive：CONFIRMED cycle-equal。
- ProgramLoad 分级组件：CONFIRMED RTL self-test + CModel component tests，
  但不是同一完整 trace。
- 完整 distributed ProgramLoad→FU：PARTIAL，集成 cycle Golden 未完成。
- legacy NOP/Repeat：CONFIRMED RTL self-test；CModel 功能测试通过，但两条
  架构的完整 cycle trace 未对齐。

## 7. CModel-only / Not Integrated 功能

### CModel-only 或保留抽象

- VXM/SXM 的完整执行和现有 Macro 支持。
- CModel packed DDB depth 8 和 lazy expansion。
- Native local iMEM/IQ 的 NOP、Repeat、WAIT、SYNC、NOTIFY、MEM transport。
- QCTL shared-pool 高层控制语义。
- ProgramImage、manifest/统计类软件元数据。

### RTL 有实现但 CModel 尚未逐对象同步

- Program command RAM/PC/WAIT/JUMP/END。
- Instruction DMA config/outstanding-ID/RX 三类 FIFO 的精确协议。
- BOTH branch、VXM direct、endpoint RR、commit-valid、PRIME FIFO。
- Legacy 138-Queue Schedule 网络的完整 timing adapter。

### RTL Not Integrated

- VXM standalone Macro frontend。
- SXM Transpose/Permute standalone Macro frontends。

## 8. 后续边界

当前模型已经能够对 MEM/MXM common path 的下列量做有限、周期化统计：

- Program word/ingress/relay occupancy 和 backpressure；
- iMEM fetch response/word FIFO occupancy；
- decoded context FIFO、active context、timed issue FIFO occupancy；
- accepted/rejected writes、reads、full cycles、stall 和 issue cycles。

在开始 FIFO depth sweep 前仍需先完成两项：

1. 使用支持当前 SystemVerilog 的较新仿真器跑通完整
   `filelist_frontend_cluster_top.f`，建立 ProgramLoad→MEM/MXM issue 的同激励
   逐周期 Trace。
2. 决定 FIFO 深度研究是否包含 RTL command/config/RX/BOTH/endpoint/PRIME
   边界；若包含，必须先补齐 CModel 这些结构。

本轮没有启动最优深度扫描，没有启动 Checkpoint 7C，也没有修改 VMODEL。
