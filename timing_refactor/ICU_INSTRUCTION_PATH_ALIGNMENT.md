# FTLPU-CMODEL ICU 指令链路与本地 VMODEL 对齐

## 1. 结论与范围

本轮以 `D:\LPU\FTLPU-VMODEL-main` 当前磁盘工作树为 RTL
依据，不以 GitHub main 或 HEAD 的干净内容替代未提交/未跟踪文件。冻结版本：

- branch：`main`
- HEAD：`2cbaa104090c6df099894559794c0f981d4aafda`
- status（2026-11-18 本轮核对时）：135 modified、13 deleted、55 untracked

VMODEL 历史上有两种互斥的 ICU 输入模式：

1. 旧 `ENABLE_DISTRIBUTED_PROGRAM_FRONTEND=0`：138 条 legacy
   Schedule Queue，由 `lpu_icu/lpu_icu_queue` 执行基础指令、NOP 和
   Repeat。
2. 当前正式链：
   ProgramLoad→分发→FU-local iMEM→Macro Frontend；MEM 为 direct issue，
   MXM 暂保留 timed issue。

本轮 CModel 的主要同步范围是第二条 **MEM/MXM distributed
ProgramLoad/Macro 路径**。VXM/SXM 的 Program boundary 虽已在当前 RTL
顶层实例化，但 `ready=0`，standalone Macro frontend 没有接到该路径。
CModel 原有 VXM/SXM、Native Queue、NOP/Repeat、QCTL 和 FU datapath
功能均保留，不因 RTL 未贯通而删除。

## 2. VMODEL 当前真实指令链路

### 2.1 Legacy Schedule 模式（OBSOLETE / NOT_APPLICABLE）

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

当前磁盘 RTL 已删除 `rtl/icu/lpu_icu.sv`、`lpu_icu_queue.sv`、
`rtl/frontend/lpu_frontend_cluster.sv` 和 MEM `lpu_mem_bank_queue.sv`。
这部分只保留为历史说明，不再作为 CModel 当前正式链的接口依据。

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
  -> direct mem_issue valid/ready boundary
  -> MEM execution
```

每个 physical MEM bank 一套，共 208 套。证据：

- `rtl/mem/lpu_mem_subsystem.sv:176-200,219-220,255-256`
- `rtl/mem/lpu_mem_local_imem.sv:17-30,60-74,98-140`
- `rtl/mem/lpu_mem_macro_frontend.sv:127-173`
- `rtl/mem/lpu_mem_macro_scheduler.sv:24-36,326-388`

`lpu_mem_bank_queue.sv` 已删除。MEM Scheduler 直接由当前 Active Context
组合产生最终 `mem_issue_*` 边界，没有独立的 Scheduler 输出寄存器或额外一级
输出流水；当前 MEM subsystem 两种执行后端都把 `execution_issue_ready`
正式拉高，不存在隐藏的 timed Instruction Queue。

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
InstructionFetchDma + shared DDR model
ID FIFO 16 + RX FIFO 16 + 256-bit response beats
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
word FIFO 4 -> MEM RTL-equivalent 288-bit reservoir / Moore FSM
      |        (12 states, append/consume mutually exclusive)
      v
decoded MEM Context holding register (valid held until ready)
      |
      v
decoded context FIFO 4
      |
      v
active contexts 4
      |
      +--> MEM：direct `mem_issue_*`，无 timed issue FIFO/lookahead
      `--> MXM：保留现有 timed issue FIFO 16 / C+2 lookahead
              -> existing MXM FU accept path
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

### 3.2.1 A.1 MEM Relay Mapping 修正

本地 RTL 的实际公式为：

```text
4 slices/group × 2 banks/slice × 2 groups/relay = 16 banks/relay
52 slices/hemisphere × 2 banks/slice = 104 MEM queues/hemisphere
ceil(104/16) = 7 relays
```

CModel 原代码只使用 `2 * banks_per_slice = 4`，导致 Queue 4～95 过早
进入远端 Relay。现将 `mem_groups_per_relay=2` 纳入 JSON/CMake 生成配置，
并在 `hardware_params.hpp` 集中生成 `kProgramLoadMemBanksPerRelay=16`。
两个半区都按以下范围消费：R0 0–15、R1 16–31、R2 32–47、R3 48–63、
R4 64–79、R5 80–95、R6 96–103。MXM Load/Dequant/Compute 仍走 R6。

RTL 证据：

- `rtl/program_load/lpu_program_distribute_system.sv:59-67,365-370`
- `rtl/program_load/lpu_program_distribute_relay_chain.sv:5-14,120-156`
- `rtl/program_load/lpu_program_distribute_relay.sv:63-65,166-180`
- `rtl/core/lpu_pkg.sv:14-19`

### 3.2.2 A.2 Instruction DMA 实际接入

`ProgramLoader::begin_queue()` 现在提交 `InstructionFetchProgram`，实际数据来自
`InstructionFetchDma`，不再用 `C2cDmaEngine::issue()` 假装指令 DMA。新路径包含：

- RTL 默认 8-bit command ID、16-entry ordered outstanding FIFO；
- 16-entry 256-bit response FIFO，保存 valid-byte count、last、ID、error；
- 32/64-bit 控制记录和 96/128-bit iMEM word 的有界 assembler；
- RX full/word sink stall 到共享 DDR completion 的反压；
- sticky ID/response/length fault 和 reset 后 orphan response drain；
- 独立 command/RX/assembler occupancy、full/stall/read/write 统计。

Instruction DMA 与 C2C Data DMA 仍是两个对象，只共享同一个 `Ddr4Model`。
SystemPhase 顺序仍为 Instruction DMA LOW → C2C Data DMA LOW → DDR LOW，DDR
每物理周期只调用一次；共同带宽竞争由 `Ddr4Model` 的 request queue、latency、
channel 和 byte budget 统一处理。

RTL 对应：`rtl/program_load/lpu_instruction_dma.sv:1-45,75-158` 和
`rtl/program_load/lpu_program_load_path.sv:104-218`。CModel 对应：
`include/ftlpu/icu/instruction_fetch_dma.hpp`、
`include/ftlpu/system/program_loader.hpp`、
`include/ftlpu/system/tsp_slice_system.hpp:282-301,773-810`。

限制：

- ProgramImage 仍按 image 串行，不实现 RTL command RAM/PC。
- 没有 BOTH record 的两分支复制与独立完成。
- 没有 endpoint round-robin、VXM direct、PRIME FIFO 的逐对象模型。
- ProgramImage 仍串行提交 image，因此实际 ProgramLoader 工作负载通常只有
  一个 command ID 在途；独立 DMA 测试已覆盖多个 command outstanding。
- Instruction DMA 通过共享 `Ddr4Model` 的 32-byte read request 适配 RTL
  256-bit response beat；本地 RTL 没有独立 DMA testbench，因此完整接口逐周期
  Trace 仍是 PARTIAL。

### 3.3 local iMEM / Macro / MEM direct issue

- fetch latency 从 1 修正为 2；加入 response FIFO 4 和 word FIFO 4，
  空 FIFO 不做写到读旁路。
- MEM 已停止使用旧 generic Parser/DDB FSM；`IcuMacroV1Decoder<MemInstruction,96>`
  现在走独立 RTL 等价路径：12 个 RTL 同序状态、288-bit LSB-first reservoir、
  96-bit word append 与字段 consume 互斥、Template Header/Payload/Validate 分拍、
  `GET_DELTA→DICT_LOOKUP→APPLY_DELTA→OUTPUT_CONTEXT` 保留寄存边界。
- `OUTPUT_CONTEXT` 在 Context FIFO 不 ready 时保持 payload、状态、remaining counter
  和 reservoir 不变；MEM 的 `ddb_layout().entry_bits==0`，旧 DDB 仅保留给尚未迁移的
  MXM/SXM generic decoder 路径。
- MEM/MXM 新增 decoded context FIFO 4；新 decoder context 先进入 FIFO，
  下一周期才可 admission 到 active context。
- MEM/MXM active context 都为 4。
- MEM 不再写入 `timed_issues_`；Scheduler 在当前 `schedule_cycle` 直接产生
  `mem_issue_valid_o/instruction_o/cycle_o`。MXM 暂时保留 timed issue FIFO 16。
- MEM PRIME 不执行或预生成基础指令；必须等 fetch 完成、Decoder done、
  Context FIFO 排空到 Active Context 后才 ready。当前 raw Macro 测试实际
  PRIME 为 21 cycles。
- Active Context 在本拍 retire 时不能同拍为 Context FIFO 提供空槽旁路；
  Context FIFO 自身允许 full+dequeue replacement，但 empty 时无 bypass。
- VXM/SXM 没有强制套用 MEM/MXM timed queue；保持原 CModel 功能并标为
  CModel-only / RTL Not Integrated。

证据：

- `config/ftlpu-lpu32.json:4-39`
- `include/ftlpu/icu/macro_bitstream_decoder.hpp:922-1054`
- `include/ftlpu/icu/distributed_queue.hpp:1972-2050,2730-2792`

### 3.4 与 RTL Instruction Path Cycle Specification 的关系

VMODEL 已建立独立周期规范目录：
`D:\LPU\FTLPU-VMODEL-main\docs\rtl_instruction_path_spec\`。

- 本文件只维护 RTL canonical signal/state 到 CModel member/view、方向/位宽、fire、
  采样点和 Mapping Status。
- RTL Spec 维护 RTL 内部 FSM、enable、寄存器、SRAM、FIFO、credit、backpressure、
  fault 和逐周期证据。
- 两份资料互相引用但不复制完整映射表。RTL 周期契约变化时，先更新对应 Spec，再
  复核本文件和 Scoreboard 规则。
- 首份正式 Spec 是
  `docs/rtl_instruction_path_spec/fu_frontend/mem_frontend.md`，版本 1.0；其源码绑定、
  VCS Trace 和未覆盖场景以该文档为准。

## 4. 正式接口映射

本表是本文件唯一的 pin/member 映射表。`Direct` 表示名称和握手语义可逐 pin
对应；`SEMANTIC` 表示 CModel 保留软件对象或原子 staged operation，不能伪装成
完整 RTL pin；`NOT_APPLICABLE` 表示当前 RTL 已删除该级。

| 阶段 / RTL canonical signal | CModel member / view | RTL / CModel 源码位置 | 方向 / 位宽 | fire 或语义 | 采样点 | Mapping Status |
|---|---|---|---|---|---|---|
| Program descriptor destination | `ProgramImage.kind/queue_index` | RTL `lpu_program_load_pkg.sv:8,61-73`；CModel `program_image.hpp` | RTL packed 12-bit；CModel typed fields | CModel 不构造虚假 packed pin；由 `relay_destination()` 解析 | submit validation | `SEMANTIC`；宽度常量为 `hw::kProgramDestinationWidth` |
| DMA command `ctrl_dma_cmd_valid_i/ready_o/address_i/length_i/id_i` | `InstructionDmaProtocolInputs` + `InstructionDmaProtocolCore` | RTL `lpu_instruction_dma.sv:11-17,76-81`；CModel `instruction_dma_protocol_core.hpp:53-196` | in/out；1/64/32/8 bit | `ctrl_dma_cmd_fire=valid&&ready` | LOW 正沿前 | `Direct` |
| DMA DDR request `ddr_read_req_valid_o/ready_i/address_o/length_o/id_o` | 同名 accessors/inputs | RTL `lpu_instruction_dma.sv:18-23,76-80`；CModel `instruction_dma_protocol_core.hpp:188-196` | out/in；1/64/32/8 bit | `ddr_read_req_fire=valid&&ready` | LOW 正沿前 | `Direct` |
| DMA DDR response `ddr_read_rsp_valid_i/ready_o/data_i/keep_i/last_i/id_i/error_i` | 同名 `InstructionDmaProtocolInputs` | RTL `lpu_instruction_dma.sv:25-32,91`；CModel `instruction_dma_protocol_core.hpp:58-252` | in/out；1/256/32/1/8/1 bit | `ddr_read_rsp_fire=valid&&ready` | LOW 正沿前；FIFO Current 正沿后 | `Direct`，108-row Golden 逐周期一致 |
| DMA reassembly output `reasm_rsp_valid_o/ready_i/data_o/keep_o/last_o/id_o/error_o` | `reasm_rsp_*` accessors | RTL `lpu_instruction_dma.sv:34-40,133-140`；CModel `instruction_dma_protocol_core.hpp:220-252` | out/in；同 response payload | `reasm_rsp_fire=valid&&ready` | LOW 正沿前 | `Direct` |
| Distribution→MEM `mem_program_valid_o/ready_i/control_o/destination_o` | `MemProgramPortSignals` + typed `ProgramImage` | RTL `lpu_program_distribute_system.sv:18-31`；CModel `program_loader.hpp:108-118,148-152` | valid/ready 1 bit；control 1；destination 12 | 当前 CModel 每次只有一个 active image/endpoint | LOW 正沿前 view | `SEMANTIC`，多端口 RR/BOTH 未建模 |
| `mem_program_imem_address_o` | `MemProgramPortSignals::mem_program_imem_address_o`、`PendingWrite::program_imem_address` | RTL `lpu_program_distribute_system.sv:25-26,410-411`；CModel `program_loader.hpp:111,542,679-688` | out；17-bit local iMEM **word address** | 随 endpoint valid 保持；只在 fire 后推进 | LOW 正沿前 | `Direct/SEMANTIC`（单 active endpoint） |
| `mem_program_data_o/strb_o/last_o` | `MemProgramPortSignals` | RTL `lpu_program_distribute_system.sv:27-30,412-420`；CModel `program_loader.hpp:112-116,681-688` | out；128/4/1 bit | MEM96 必须 `strb=4'b0111`；`last` 对应 image 最后 word | LOW 正沿前 | `Direct/SEMANTIC`；ProgramLoader test 已检查 |
| MEM local load `load_valid_i/ready_o/load_imem_address_i/load_data_i/load_strb_i/load_last_i` | `MemLocalImemLoadPortSignals`、`load_mem_program_word()` | RTL `lpu_mem_local_imem_frontend.sv:13-20,93-104`；CModel `distributed_queue.hpp:234-248,723-801` | in/out；1/17/128/4/1 bit | `load_fire=valid&&ready`；检查 0111、范围、连续、last 后禁止写 | staged ICU operation 在共同 POSEDGE 提交 | `SEMANTIC` staged pin adapter；错误用 terminal exception + fault view |
| MEM PRIME `prime_valid_i/prime_ready_o` | `prime_mem_program()` + 同一 load view | 同上；CModel `distributed_queue.hpp:772-801` | in/out；1 bit | `prime_fire=valid&&ready`；要求 configured、last、非零 word count | Program image seal 的共同 POSEDGE | `SEMANTIC` staged pin adapter |
| local iMEM request `imem_read_req_valid_o/ready_i/address_o/last_o` | `IcuMacroV1Decoder::ImemReadPortSignals` | RTL `lpu_mem_imem_fetch.sv:15-18,62-65`；CModel `macro_bitstream_decoder.hpp:160-174,1088-1102` | out/in；1/address/1；production 默认 address=17 bit，Level1 `IMEM_DEPTH=16` 实例为 4 bit | `imem_read_req_fire=valid&&ready`；阻塞时 address 不推进 | 每次 frontend tick 正沿前 view | `Direct/Partial`；Trace 按实例参数化宽度比较，不改变 production 17-bit 映射 |
| local iMEM response `imem_read_rsp_valid_i/ready_o/data_i/last_i` | 同一 `ImemReadPortSignals` | RTL `lpu_mem_imem_fetch.sv:20-23,67-71`、`lpu_mem_local_imem.sv:60-71`；CModel `macro_bitstream_decoder.hpp:1028-1056` | in/out；1/96/1 | `imem_read_rsp_fire=valid&&ready`；two-stage SRAM + response FIFO4；无空旁路 | 正沿前 valid/fire，正沿后 FIFO Current | `Direct/Partial` |
| 通用 FIFO `enqueue_valid_i/ready_o/data_i`、`dequeue_valid_o/ready_i/data_o`、`level_o` | `RtlQueueFifoInputs/Outputs`、`rtl_outputs()` | RTL `lpu_queue_fifo.sv:8-16,37-42`；CModel `rtl_fifo.hpp:19-33,90-138` | 参数化 data width；level=`clog2(depth+1)` | `fire=valid&&ready`；empty 无 bypass；full+pop 可 replacement | LOW 正沿前；POSEDGE 后更新 level/head | `Direct` |
| Macro word FIFO | decoder 内 `word_fifo_` + statistics | RTL `lpu_mem_macro_frontend.sv:80-94`；CModel `macro_bitstream_decoder.hpp:998-1056` | 97-bit，depth4 | 同通用 FIFO；empty 无 bypass/full replacement | frontend tick | `Direct/Partial`，未改成独立 ClockedModule |
| MEM Decoder state / subphase / reservoir | `MemState`、`MemTemplatePhase`、`mem_reservoir_`、`mem_*_phase_` | RTL `lpu_mem_macro_decoder.sv:43-75,77-185,345-721,733-1013`；CModel `macro_bitstream_decoder.hpp:MemState,mem_parser_step,mem_tick` | state 4 bit；reservoir 288 bit + count 9 bit | Control 直接接收；后续 Word append 与 consume 互斥；Context valid 只在 OUTPUT_CONTEXT | 每次 tick 保存本拍 Current state/phase/reservoir trace view | `Direct` for MEM；MXM/SXM 不假设同构 |
| Context FIFO handshake/level | `MacroContextFifoPortSignals`、`context_fifo_port_signals()`、`context_fifo_occupancy()` | RTL `lpu_mem_macro_frontend.sv:127-140`；CModel `distributed_queue.hpp:250-258,975-980,2188-2230` | context 271-bit，depth4 | empty 无 bypass；full+dequeue replacement；Active full 本拍 retire 不提供 admission bypass | valid/ready/fire 在 LOW/PRE_POSEDGE；level 在 POSEDGE/POST_COMB | `Direct/Partial`，payload 保持 typed semantic object |
| Scheduler Context admission / `active_count_o` | `decoded_context_fifo_`→`active_macro_storage_`、`active_macro_count()` | RTL `lpu_mem_macro_scheduler.sv:10-22,294-302,373-375`；CModel `distributed_queue.hpp:2207-2230,2278-2306` | valid/ready + typed schedule；count 3 bit（ACTIVE_CONTEXTS=4） | 运行时 `start_cycle < schedule_cycle+1` 为 late；新 Context 次周期可见 | 正沿提交；count 等组合 reduction 稳定后的 POSEDGE/POST_COMB | `Direct/Partial` |
| Scheduler→MEM `mem_issue_valid_o/ready_i/instruction_o/cycle_o` | `MemIssuePortSignals`、`dispatch_mem_issue()` | RTL `lpu_pkg.sv:145-150`、`lpu_mem_macro_scheduler.sv:24-36,333-375`；CModel `instruction_codec.hpp:57,209-243`、`distributed_queue.hpp:220-228,1229-1247,3139-3201` | valid/ready 各 1 bit；正式 RTL instruction container 64 bit；CModel 使用 typed `MemInstruction`，物理编码为 64 bit；cycle 32 bit | `mem_issue_fire=valid&&ready`；到期 backpressure=terminal fault 11；冲突=9；missed=8 | RUN 周期正沿前 view；accept 后推进 Active | `Direct/SEMANTIC`；CModel exception 同时保留 fault view |
| MEM sink ready | `MemArrayModel::mem_issue_ready()` | RTL `lpu_mem_subsystem.sv:219-220,255-256`；CModel `mem_array.hpp:264-270` | input to Scheduler；1 bit/queue | 当前 native/legacy execution 均正式 tie-high | ICU dispatch LOW | `Direct`；不是伪造 ready |
| MEM timed Instruction Queue / C+2 lookahead | 无 | RTL `lpu_mem_bank_queue.sv` 已删除；CModel MEM `timed_issue_depth/lookahead=0` | N/A | MEM 不读写 `timed_issues_`；MXM 保留旧路径 | N/A | `NOT_APPLICABLE` for MEM |
| legacy `lpu_icu_queue` | CModel Native/compatibility queue controls | RTL 文件已删除；CModel 保留软件兼容功能 | N/A | 不属于正式 distributed MEM chain | N/A | `OBSOLETE RTL / CModel compatibility-only` |
| VXM/SXM Program Macro | 原有 CModel path | 本轮未接入 | N/A | 不扩展 B1 外部 adapter | N/A | `NOT_IN_SCOPE` |

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

### 6.0 2026-11-18 MEM Decoder RTL 等价重构增量

- `raw_two_issue`：Control C14；Context FIFO enqueue C24/C28；MEM issue C3/C7，
  与现有 RTL trace 相同。
- `raw_repeat4`：Control C12；Context FIFO enqueue C20；MEM issue C0/C1/C2/C3，
  与现有 RTL trace 相同。
- CModel 定向回归 9/9 PASS：instruction codec、macro schedule、RAW decoder、
  Level1 trace、queue control、VXM sequential、C2C packet codec、RTL FIFO、
  instruction fetch DMA。queue-control 的旧“两个 MEM Context 不得交错”断言已按
  当前 Active Context Scheduler 语义改为：无 due-cycle 冲突即可交错，queue-local
  WAIT 暂停逻辑 schedule cycle 后可继续。
- RAW decoder 单测另覆盖 Context backpressure hold、跨 96-bit word、Dictionary
  lookup、Narrow/Wide Escape、Extended Template，以及 fault 2/3/6/7/8。
- Release 全量工程已成功构建。Decoder/ICU 相关定向 CTest 为 9/9 PASS。
- Release 全量 CTest 为 66/80 PASS；其余 14 项中，`subsystem.program.loader`
  的 stress fixture 在一个合法 3-word MEM Macro 后继续填充随机 word。RTL 等价的
  decoder 在 terminal done 后会拉低 word-ready，因而不会吞掉这些 padding，Program
  Loader 也就不能把 160-word 描述符判成 fetch complete。该失败是旧 fixture 与当前
  正式 frontend 契约不兼容，不能通过让 done 后继续消费 word 或人为补拍来掩盖；
  后续应把 fixture 改为精确长度的合法 Macro image，或明确改走 Native load-only 路径。
  另外 13 项集中在既有 VXM/MXM/SXM 数据路径、SRAM 配置或权重到达时序，本轮没有
  把它们归因于 MEM decoder。
- Debug 全量工程在若干大型测试目标上触发 MSVC C1128（需要 `/bigobj`）；同一源码的
  Release 全量构建通过，因此它记录为 Debug 构建配置限制，不作为 decoder 行为失败。
- 修正 Monitor 层级与真实采样边界后，VCS/CModel 离线 Scoreboard 为
  `CYCLE_PASS`：410/410 pair matched，0 differences。覆盖 load/PRIME、iMEM
  request/response、Context FIFO enqueue/dequeue/level、完整 decoded Context、
  MEM issue（含 Scheduler conflict/fault/fault_code）、Active Context count 和
  Word FIFO level。
- 修正内容不是 pipeline 补拍：Level1 iMEM 地址按 RTL test 实例
  `IMEM_DEPTH=16` 比较 4-bit，production 映射仍为 17-bit；Scheduler status
  直接取 4-bit Scheduler pin，不再误取 6-bit frontend fault；FIFO level 和
  `active_count_o` 在 NBA 后等待组合逻辑稳定再以 `POSEDGE/POST_COMB` 采样；
  START/MEM_ISSUE/CASE_END 顺序按同一 test harness 对齐。没有使用 cycle offset。
- RTL VCS `X-2025.06-SP1_Full64` 编译和两个 directed case 均 PASS；运行输入为
  `/home/RD/ryan/projects/LPU/build/raw_macro_level1/source` 局部精确快照，运行目录为
  `/home/RD/ryan/projects/LPU/build/raw_macro_level1/run_rtl_alignment_20261118`。
  VCS Trace 保留在虚拟机；本地临时副本在比较后删除。
- 为首份 Instruction Path Spec 增加的 Fetch/iMEM 只读观测点已在 VCS
  `run_instruction_path_spec_20261118` 重新运行：两个 case 和总测试 PASS；Trace
  5391 行，SHA-256
  `23b7f389effa917d177e421efffcff95132a0533e779071d40bcabf22abc8c41`。新增观测包括
  Fetch `busy/done/address/request/response remaining`、local-iMEM 两级 valid、
  `outstanding_q` 和 response FIFO level；没有改变 DUT 激励或正式 RTL。
- 新 Trace 重新执行离线 Scoreboard，仍为 `CYCLE_PASS`：410/410 pair matched，
  0 differences。报告位于
  `out/icu_rtl_alignment/raw_macro_level1/scoreboard_instruction_path_spec_vcs/`。

### 6.1 CModel

以下是 MEM Decoder 重构前、同日完成的 ProgramLoad/ICU 历史定向基线；当前重构后的
验证结论以 6.0 为准。当时重新构建并运行的定向 Release 测试 10/10 PASS：

- `unit.icu.rtl_fifo`
- `unit.icu.macro_schedule`
- `unit.icu.raw_macro_decoder`
- `subsystem.program.loader`
- `subsystem.program.relay_mapping`
- `subsystem.program.configuration`
- `subsystem.icu.configuration`
- `unit.icu.clocked`
- `subsystem.system.timing`

关键结果：

- raw Macro：`prime_cycles=21`；PRIME 不产生 FU issue，Cycle 0 的 direct
  `mem_issue_*`、Repeat=4 连续四拍和 due-cycle backpressure fault 11 均通过。
- configured ProgramLoad：170 cycles（Instruction DMA 实际路径）。
- Program endpoint 的 17-bit word address、128-bit data、`strb=0111`、last
  和 valid/ready/fire 关系已由 `program_loader_test` 检查。
- Program word/ingress/relay write/read 守恒检查通过。
- Stage A Instruction DMA Golden：108 rows，`mismatches=0`，
  `semantic_failures=0`；临时 CModel trace 已删除。
- A.1 新增 East/West 全部 208 条 MEM Queue 穷举映射；A.2 新增单读、
  多 outstanding、ID/error、RX FIFO full/recovery、reset/final drain 测试。
- A.1 代表 Trace：Queue 4 从错误 R1 修正到 R0，9 words 的 relay writes
  从错误预期 18 变为 9；Queue 16 从错误 R4 修正到 R1，实际 18；Queue 96
  保持 R6，实际 63。
- 当时没有重跑全量 79 项；旧 72/79 结果仅为历史基线，不能替代 6.0 中当前
  Decoder 重构后的验证结论。

### 6.2 RTL 与跨模型 Golden

直接使用本地 VMODEL 源码、ModelSim 10.1c，输出只写入 CModel
`out/icu_rtl_alignment/rtl_golden`：

| 验证 | 结果 |
|---|---|
| `lpu_queue_fifo_tb` | PASS |
| 相同激励 FIFO RTL/CModel，12 cycles×17 fields | PASS，逐字段一致 |
| legacy ICU Queue NOP/Repeat/MEM stride | 历史 PASS；当前 RTL 文件已删除，NOT_APPLICABLE |
| Program config FIFO | PASS，5 checks |
| Program reassembler | PASS，7 checks |
| Program elastic stage | PASS |
| 7-stage relay chain | PASS，80 checks |
| `lpu_instruction_dma` RTL compile | PASS；本地没有独立功能 TB，功能 Golden 为 PARTIAL |
| MEM Macro frontend | 历史 PASS；本轮以当前源码静态 pin/周期契约和 CModel 定向测试复核 |
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
- MXM/SXM generic Macro 路径仍保留 CModel packed DDB depth 8 和 lazy expansion；
  MEM 已切换为 RTL 等价逐状态 Decoder，不再使用该 DDB。
- Native local iMEM/IQ 的 NOP、Repeat、WAIT、SYNC、NOTIFY、MEM transport。
- QCTL shared-pool 高层控制语义。
- ProgramImage、manifest/统计类软件元数据。

### RTL 有实现但 CModel 尚未逐对象同步

- Program command RAM/PC/WAIT/JUMP/END。
- Program config FIFO 与 Instruction DMA command metadata 的完整多 image 并发配对。
- BOTH branch、VXM direct、endpoint RR、commit-valid、PRIME FIFO。
- 已删除的 Legacy 138-Queue Schedule 网络不再要求 formal adapter；只保留
  CModel compatibility path。

### RTL Not Integrated

- VXM standalone Macro frontend。
- SXM Transpose/Permute standalone Macro frontends。

## 8. 后续边界

当前模型已经能够对 MEM/MXM common path 的下列量做有限、周期化统计：

- Program word/ingress/relay occupancy 和 backpressure；
- iMEM fetch response/word FIFO occupancy；
- decoded context FIFO、active context；MXM timed issue FIFO occupancy；
- MEM direct `mem_issue_*` valid/ready/fire/cycle/fault；
- accepted/rejected writes、reads、full cycles、stall 和 issue cycles。

在开始 FIFO depth sweep 前仍需先完成两项：

1. 使用支持当前 SystemVerilog 的较新仿真器跑通完整
   `filelist_frontend_cluster_top.f`，建立 ProgramLoad→MEM/MXM issue 的同激励
   逐周期 Trace。
2. 决定 FIFO 深度研究是否包含 RTL command/config/BOTH/endpoint/PRIME
   边界；Instruction DMA outstanding/RX 已可统计，但其他结构若纳入扫描仍须补齐。

本轮没有启动最优深度扫描，没有启动 Checkpoint 7C，也没有修改 VMODEL。
