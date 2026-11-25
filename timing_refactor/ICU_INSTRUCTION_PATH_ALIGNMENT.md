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
   ProgramLoad→分发→FU-local iMEM→Macro Frontend；MEM 与 MXM Scheduler
   都直接驱动 FU issue valid/ready，不存在 Scheduler 后的 Instruction Queue。

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
  -> direct issue valid/ready
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
      `--> MXM：typed direct `issue_*`，无 timed issue FIFO/lookahead
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
  和 reservoir 不变；MEM 的 `ddb_layout().entry_bits==0`。MXM 仍保留其 stream-specific
  Decoder/DDB 语义，但其 local-iMEM 传输已经改用正式 128-bit SRAM 路径。
- MEM/MXM 新增 decoded context FIFO 4；新 decoder context 先进入 FIFO，
  下一周期才可 admission 到 active context。
- MEM/MXM active context 都为 4。
- MEM/MXM 正式路径均不写入 `timed_issues_`；Scheduler 在当前
  `schedule_cycle` 直接产生 typed issue valid/instruction/cycle。Ready 低时按 RTL
  合同锁存 terminal backpressure fault，Active Context 不推进。
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
| WRITE/PRIME shared Distributor input `load_record_valid_i/ready_o/control_i/data_i/strb_i/scope_i/destination_i/imem_address_i/last_i` | `ProgramDistributionHardwareInterfaceView::load_record_input`；WRITE 来自 Word FIFO，PRIME 来自 registered 32-entry Prime FIFO | RTL `lpu_program_load_system.sv:376-410`、`lpu_program_distributor.sv`；CModel `program_distribution_transport.hpp`、`program_loader.hpp` | valid/ready/fire 1 bit；control 1；data 128；strb 4；scope 2；destination 12；address 17；last 1 | `fire=valid&&ready`；payload 在 stall 期间保持；3 个 WRITE 后由 registered load-done 选择 1 个 PRIME；只在 fire 后各自出队 | `PRE_POSEDGE`，cycle offset 0 | `Direct`；R3 local Trace 为 3 WRITE + 1 PRIME |
| Distributor elastic / East-West routes | `elastic_output`、`elastic_level`、`distributor_route_output[2]`、`east_done/west_done` | RTL `lpu_program_distributor.sv`、`lpu_program_distribute_elastic_stage.sv`；CModel `program_distribution_transport.hpp:47-73,288-341` | channel payload 同上；elastic depth=2；done 各 1 bit | elastic full+dequeue 可 replacement；BOTH 的两分支可在不同周期完成，全部完成后才释放队首 | `PRE_POSEDGE`，POSEDGE 统一提交 | `Direct`；R3 不用影子状态或固定 offset |
| East/West Relay0..6 input/forward/local | `relay_input[2][7]`、`relay_forward_output[2][7]`、`mem_endpoint_input[2][7]`、`relay_level[2][7]` | RTL `lpu_program_distribute_relay.sv`、`lpu_program_distribute_relay_chain.sv`；CModel `program_distribution_transport.hpp:226-282,344-377` | 每级 depth=2；channel payload 同 Distributor | local/forward 由正式 destination 决定；relay 满时即使同拍出队也不做 replacement admission | `PRE_POSEDGE`，POSEDGE 统一提交 | `Direct`；两半区和 7 级接口均进入 R3 Trace |
| Outer/VXM/Program Endpoint input | `outer_endpoint_input[2]`、`vxm_endpoint_input`、`mem_endpoint_input[2][7]`；兼容消费适配为 `MemProgramPortSignals` | RTL `lpu_program_distribute_system.sv`；CModel `program_distribution_transport.hpp`、`program_loader.hpp` | channel payload 同上；MEM address 17 bit | R3 边界停在 Endpoint input；Endpoint 的 local iMEM 写入/提交属于 R4；现有生产消费适配不回写 Transport 状态 | `PRE_POSEDGE` | `Direct` 到 Endpoint input；R3 joint PASS，0 functional diff / 0 hold violation |
| `mem_program_imem_address_o/data_o/strb_o/last_o` | `ProgramDistributionRecord::{imem_address,data,strb,last}`、`MemProgramPortSignals` | RTL `lpu_program_distribute_system.sv`；CModel `program_distribution_transport.hpp:18-29`、`program_loader.hpp:1400-1429` | out；17/128/4/1 bit | MEM96 必须 `strb=4'b0111`；valid stall 时全部 payload 字段保持 | LOW 正沿前 | `Direct`；ProgramLoader 与 R3 hold 检查通过 |
| MEM local load `load_valid_i/ready_o/load_imem_address_i/load_data_i/load_strb_i/load_last_i` | `MemLocalImemLoadPortSignals`、`load_mem_program_word()` | RTL `lpu_mem_local_imem_frontend.sv:13-20,93-104`；CModel `distributed_queue.hpp:234-248,723-801` | in/out；1/17/128/4/1 bit | `load_fire=valid&&ready`；检查 0111、范围、连续、last 后禁止写 | staged ICU operation 在共同 POSEDGE 提交 | `SEMANTIC` staged pin adapter；错误用 terminal exception + fault view |
| MEM PRIME `prime_valid_i/prime_ready_o` | `prime_mem_program()` + 同一 load view | 同上；CModel `distributed_queue.hpp:772-801` | in/out；1 bit | `prime_fire=valid&&ready`；要求 configured、last、非零 word count | Program image seal 的共同 POSEDGE | `SEMANTIC` staged pin adapter |
| local iMEM request `imem_read_req_valid_o/ready_i/address_o/last_o` | `IcuMacroV1Decoder::ImemReadPortSignals` | RTL `lpu_mem_imem_fetch.sv:15-18,62-65`；CModel `macro_bitstream_decoder.hpp:160-174,1088-1102` | out/in；1/address/1；production 默认 address=17 bit，Level1 `IMEM_DEPTH=16` 实例为 4 bit | `imem_read_req_fire=valid&&ready`；阻塞时 address 不推进 | 每次 frontend tick 正沿前 view | `Direct/Partial`；Trace 按实例参数化宽度比较，不改变 production 17-bit 映射 |
| local iMEM response `imem_read_rsp_valid_i/ready_o/data_i/last_i` | 同一 `ImemReadPortSignals` | RTL `lpu_mem_imem_fetch.sv`、`lpu_mxm_imem_fetch.sv`、MEM/MXM `lpu_*_local_imem.sv`；CModel `sram_1rw.hpp`、`mem_local_frontend_fetch.hpp`、`macro_bitstream_decoder.hpp` | in/out；MEM 96 bit、MXM 128 bit、Last 1 bit | 单端口同步 SRAM read；wrapper response stage + ordinary response FIFO4；无空旁路 | 正沿前 valid/fire，正沿后 Current | `Direct/Partial` |
| 通用 FIFO `enqueue_valid_i/ready_o/data_i`、`dequeue_valid_o/ready_i/data_o`、`level_o` | `RtlQueueFifoInputs/Outputs`、`rtl_outputs()` | RTL `lpu_queue_fifo.sv:8-16,37-42`；CModel `rtl_fifo.hpp:19-33,90-138` | 参数化 data width；level=`clog2(depth+1)` | `fire=valid&&ready`；empty 无 bypass；full+pop 可 replacement | LOW 正沿前；POSEDGE 后更新 level/head | `Direct` |
| Macro word FIFO | decoder 内 `word_fifo_` + statistics | RTL `lpu_mem_macro_frontend.sv:80-94`；CModel `macro_bitstream_decoder.hpp:998-1056` | 97-bit，depth4 | 同通用 FIFO；empty 无 bypass/full replacement | frontend tick | `Direct/Partial`，未改成独立 ClockedModule |
| MEM Decoder state / subphase / reservoir | `MemState`、`MemTemplatePhase`、`mem_reservoir_`、`mem_*_phase_` | RTL `lpu_mem_macro_decoder.sv:43-75,77-185,345-721,733-1013`；CModel `macro_bitstream_decoder.hpp:MemState,mem_parser_step,mem_tick` | state 4 bit；reservoir 288 bit + count 9 bit | Control 直接接收；后续 Word append 与 consume 互斥；Context valid 只在 OUTPUT_CONTEXT | 每次 tick 保存本拍 Current state/phase/reservoir trace view | `Direct` for MEM；MXM/SXM 不假设同构 |
| Context FIFO handshake/level | `MacroContextFifoPortSignals`、`context_fifo_port_signals()`、`context_fifo_occupancy()` | RTL `lpu_mem_macro_frontend.sv:127-140`；CModel `distributed_queue.hpp:250-258,975-980,2188-2230` | context 271-bit，depth4 | empty 无 bypass；full+dequeue replacement；Active full 本拍 retire 不提供 admission bypass | valid/ready/fire 在 LOW/PRE_POSEDGE；level 在 POSEDGE/POST_COMB | `Direct/Partial`，payload 保持 typed semantic object |
| Scheduler Context admission / `active_count_o` | `decoded_context_fifo_`→`active_macro_storage_`、`active_macro_count()` | RTL `lpu_mem_macro_scheduler.sv:10-22,294-302,373-375`；CModel `distributed_queue.hpp:2207-2230,2278-2306` | valid/ready + typed schedule；count 3 bit（ACTIVE_CONTEXTS=4） | 运行时 `start_cycle < schedule_cycle+1` 为 late；新 Context 次周期可见 | 正沿提交；count 等组合 reduction 稳定后的 POSEDGE/POST_COMB | `Direct/Partial` |
| Scheduler→MEM `mem_issue_valid_o/ready_i/instruction_o/cycle_o` | `MemIssuePortSignals`、`dispatch_mem_issue()` | RTL `lpu_pkg.sv:145-150`、`lpu_mem_macro_scheduler.sv:24-36,333-375`；CModel `instruction_codec.hpp:57,209-243`、`distributed_queue.hpp:220-228,1229-1247,3139-3201` | valid/ready 各 1 bit；正式 RTL instruction container 64 bit；CModel 使用 typed `MemInstruction`，物理编码为 64 bit；cycle 32 bit | `mem_issue_fire=valid&&ready`；到期 backpressure=terminal fault 11；冲突=9；missed=8 | RUN 周期正沿前 view；accept 后推进 Active | `Direct/SEMANTIC`；CModel exception 同时保留 fault view |
| MEM sink ready | `MemArrayModel::mem_issue_ready()` | RTL `lpu_mem_subsystem.sv:219-220,255-256`；CModel `mem_array.hpp:264-270` | input to Scheduler；1 bit/queue | 当前 native/legacy execution 均正式 tie-high | ICU dispatch LOW | `Direct`；不是伪造 ready |
| MEM/MXM timed Instruction Queue / C+2 lookahead | 无 | RTL MEM/MXM scheduler 均直接输出 issue valid/ready；CModel formal queues `timed_issue_depth/lookahead=0` | N/A | 正式 MEM/MXM 不读写 `timed_issues_`；历史容器只隔离给 compatibility 代码 | N/A | `NOT_APPLICABLE` for formal MEM/MXM |
| legacy `lpu_icu_queue` | CModel Native/compatibility queue controls | RTL 文件已删除；CModel 保留软件兼容功能 | N/A | 不属于正式 distributed MEM chain | N/A | `OBSOLETE RTL / CModel compatibility-only` |
| VXM/SXM Program Macro | 原有 CModel path | 本轮未接入 | N/A | 不扩展 B1 外部 adapter | N/A | `NOT_IN_SCOPE` |

Phase 3 分阶段证据：R2 DMA/Reassembler/Word FIFO 已通过联合 scoreboard；R2 入口、
schema 和 89-row Trace 保持独立可运行。R3 Distributor/Elastic/两半区 Relay0..6/
Endpoint-input 已完成 CModel local directed verification（89 rows，3 WRITE + 1 PRIME，
0 fault）；共享 Trace 使用生产 ready，独立 Transport 单测覆盖 Endpoint backpressure、
payload hold、BOTH 和 Relay-full 无 replacement。2026-11-20 联合 Scoreboard 结果为
`R3_PASS`：Input Equivalence PASS、RTL Trace Integrity PASS、R2 prerequisite PASS、
R3 functional diff=0、hold violation=0。96-bit WRITE 的 10 个 inactive-lane diagnostics
按 `strb=4'b0111` 协议保留，不计为有效 payload 差异。该结论仅覆盖 R3；跨阶段 Full
Joint 仍有非 R3 差异，不能据此声明全系统周期一致。

R4 将 canonical MEM/MXM 写入从 ProgramLoader 的延迟完成通知移到 Endpoint
接受沿。Transport 以 retained head 的目标查询对应 Queue 的 Current
`ProgramEndpointState`，不再维护独立的 primed 数组。WRITE 握手通过 ICU
staged operation 更新 local iMEM 和装载元数据；完成通知仍用于装载 drain。
COMMIT_QUEUE 将 PRIME 加入共享物理通道，只有目标 Endpoint 接受该 PRIME
才执行 seal，改变该 Queue 的 `prime_started`。这里的 load/PRIME effect 对应
`lpu_mem_local_imem_frontend.sv` 与 `lpu_mxm_local_imem_frontend.sv`。
独立 R4 Trace 在 PRE_POSEDGE 记录握手和 Current 状态，测试同时验证接受沿后的
iMEM 和元数据。

R4 联合验收为 `R4_PASS`。92 周期 MEM 生产场景重新通过 R2/R3，
MEM Endpoint 与独立 MXM Endpoint 直驱比较均为零功能差异。报告位于
`out/phase3_r4_joint_scoreboard/final_20261120`（VMODEL）。

R5 使用独立的 QCTL Program Load 端口。RTL `lpu_top.sv` 将
`qctl_load_mode`、`partition_{valid,ready,queue,base,count,generation}` 与
`load_{valid,ready,address,data}` 直接转入 Cluster 的 QCTL Global Subsystem。
物理 Pool 地址为 9 bit，记录为 96 bit。Partition 握手建立目录和 endpoint
配置，Record 握手写 Pool。退出 load mode 后 Cache 可向 Pool 发出读取请求，
没有 PRIME 或 commit record。CModel 对应入口为
`InstructionControlUnit::drive_qctl_program_load()`，在 SystemPhase 的 ICU
Next 上消费端口并由共同时钟提交。原 `ProgramStorage::Qctl` 的软件映像 API
继续按既有兼容流程运行。QCTL 共享 Instruction DMA/Distribution 目标状态为
`RTL_NOT_IMPLEMENTED`，Program Target kind 7 仍为 RESERVED。

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

### 6.0.1 RAW Macro Level 2：MEM 执行端 CModel 基线

Level 2 从正式 64-bit MEM machine word 开始，检查外部 Issue 到 MEM command
wave、SRAM 和 Stream Register 的执行端周期语义。它不重复 Level 1 的 local
iMEM、Fetch、Decoder、Context FIFO 和 Scheduler 比较，也不引入 bit 30 私有路径、
三周期兼容路径、Replay 或 busy-based flow control。

共享输入定义位于
`tests/support/raw_macro_level2/shared_input.hpp`。五个定向 case 都通过
`isa::encode_mem_instruction()` 产生 machine word，并由测试在注入前使用
`isa::decode_mem_instruction()` 往返校验。测试同时导出
`shared_input_manifest.json`，其中保留准确 issue cycle、MEM slice/bank、地址、
stream/direction、每个 tile 的 SRAM preload 或 SR boundary stimulus。当前 case 为：

- `mem_read_single`；
- `mem_write_single`；
- `mem_read_consecutive`，同一 bank 每周期一条；
- `mem_write_consecutive`，同一 bank 每周期一条；
- `mem_read_write_mixed`，同一 bank 读写交替，单周期不发两条同 bank 指令。

测试使用真实 `TspSliceSystem::tick()`，因此实际经过 `SystemPhase::low()`、全局
`Clock::validate_all()/commit_all()` 和 `SystemPhase::high()`。MEM read timing 显式
锁定并断言为 `MemArrayModel::ReadTiming::RegisteredR1`。映射如下：

| Level 2 事件/信号 | RTL canonical 依据 | CModel 依据 | 周期边界 | 映射等级 |
|---|---|---|---|---|
| `execution_issue_ready` | `lpu_mem_subsystem.execution_issue_ready_o` | `MemArrayModel::mem_issue_ready()` | Issue 所在 LOW 正沿前；恒为 1 | `DIRECT` |
| `MEM_ISSUE_ACCEPT` | 外部 Issue + `lpu_mem_command_adapter` | 正式 64-bit word 解码后 `enqueue_instruction()` | Cycle C LOW 正沿前 | `SEMANTIC`，外部 Issue adapter |
| `MEM_TILE_COMMAND` | `mem_bank_control_column.leaf_cmd` | `executed_instructions()` | Tile 0 在 C；Tile k 在 C+k 的 LOW 正沿前执行 | `SEMANTIC` |
| `MEM_SRAM_READ_SAMPLE` | `mem_bank_superlane_leaf.mem_array` | `executed_transfers()` Read | 执行边沿前取样，C 的 POSEDGE/NBA 注册 R1 | `SEMANTIC` |
| `MEM_READ_RESPONSE` | `mem_bank_superlane_leaf.read_data_o` | `read_responses()` | C 的 `POST_NBA`；Trace 旧字段名 `POST_POSEDGE` 规范化为同一采样区 | `SEMANTIC` |
| `MEM_SRF_INJECT_COMMIT` | `lpu_mem_srf_integration` producer → SRF state | `staged_read_responses()` | R1 后下一执行周期 C+1 的 `POST_NBA` | `SEMANTIC` |
| `MEM_STREAM_CONSUME` | 正沿前 selected SRF state；`registered_stream_consume` 仅诊断 | `consumed_transfers()` | Write 使用正沿前 command Current 与 SR Current | `SEMANTIC` |
| `MEM_SRAM_WRITE_COMMIT` | `mem_bank_superlane_leaf` SRAM write | POSEDGE 后 SRAM lane 回读 | 与对应 Write execution edge 同一 `POST_NBA` | `SEMANTIC` |
| `mem_busy_derived` / `MEM_DRAINED` | `mem_busy_o || |mem_producer_valid_o` | `!MemArrayModel::idle()` / `idle()` | `POST_COMB` | `DERIVED`；不是正式 complete pin |

Trace schema 为 `ftlpu.raw_macro.level2.trace.v1`，在 Level 1 公共字段之外增加
hemisphere、slice、bank、tile、opcode、address、stream、direction、preserve、
data、category 和 scope。事件域固定为 `ICU/MEM/SRAM/STREAM`；记录分类固定为
`CANONICAL/DIAGNOSTIC/EVENT`。标准事件为
`MEM_ISSUE_ACCEPT`、`MEM_TILE_COMMAND`、`MEM_SRAM_READ_SAMPLE`、
`MEM_SRAM_WRITE_COMMIT`、`MEM_STREAM_CONSUME`、`MEM_READ_RESPONSE`、
`MEM_SRF_INJECT_COMMIT` 和 `MEM_DRAINED`。

`tools/cycle_scoreboard/raw_macro_level1_scoreboard.py` 保持 Level 1 默认 CLI 和
规则兼容，同时支持 Level 2 的附加字段、scope 过滤、完整 differences、模块/
信号/事件/映射审计、masked/missing-mapping/uncompared 统计和按 category 的匹配率。
Level 2 规则位于 `raw_macro_level2_rules.json`，默认只比较
`ICU/MEM/SRAM/STREAM`；不会用固定 cycle offset，也不会放宽 X/Z。

2026-11-19 最终联合结果为指定 Level 2 范围内 `CYCLE_PASS`：CModel Trace 226 条，
VCS RTL Trace 392 条；后者包含 166 条 `UNMAPPED` 只读诊断记录。离线 Scoreboard
比较 `ICU/MEM/SRAM/STREAM`，结果为 226/226 matched、0 differences：
`CANONICAL=12/12`、`DIAGNOSTIC=29/29`、`EVENT=185/185`。五个 case 的 Drain
分别为单 READ C4、单 WRITE C3、连续三 READ C6、连续三 WRITE C5、混合 C6。

首次比较曾报告 99 个 architecture order differences。逐字段无序多重集复核证明两边
185 个正式事件在 case、cycle、phase/sample、Tile、地址、stream、data 和 scope 上
完全相同，差异仅来自同拍并行 Tile 的 JSONL 枚举顺序：CModel 按事件类别批量记录，
RTL Monitor 按 Tile 交错记录。Level 2 因而改用显式 `KEYED_IDENTITY`，身份键包含上述
全部硬件字段；value/width 仍严格比较，Cycle 或 Tile 改动自检仍 FAIL。没有使用
+cycle/-cycle offset、补拍、X/Z 放宽或删除必需字段。`POST_POSEDGE` 只按同义采样区
规范化为 `POST_NBA`，不改变周期。

受影响的 instruction-codec、Clock、MEM、Stream 和 TSP timing 定向 CTest 为
14/14 PASS。Level 1 和 Level 2 原有 Scoreboard 自检 12/12 继续 PASS，新增的同拍并行
顺序与 Cycle/Tile 负例使总自检为 14/14 PASS。新目录中重跑的 Level 1 CModel Trace
为 2522 行，与既有基线逐行 0 differences；既有 410/410 VCS Level 1 报告未覆盖。
本轮没有修改 CModel MEM 功能，只修改了 Scoreboard 的严格事件身份规范化。

正式报告位于
`out/icu_rtl_alignment/raw_macro_level2/scoreboard_vcs_20261119/`。本次
`CYCLE_PASS` 只覆盖共享 Manifest 驱动的 East/Slice0/Bank0 缩小阵列中正式
64-bit Issue→adapter→4-Tile command wave→leaf SRAM→SRF 路径；不覆盖 Program
Load、local iMEM、Macro frontend、QCTL、完整 52-slice/2-bank 并行阵列或其他 FU。

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
- decoded context FIFO、active context；
- MEM/MXM direct issue valid/ready/fire/cycle/fault；
- accepted/rejected writes、reads、full cycles、stall 和 issue cycles。

在开始 FIFO depth sweep 前仍需先完成两项：

1. 使用支持当前 SystemVerilog 的较新仿真器跑通完整
   `filelist_frontend_cluster_top.f`，建立 ProgramLoad→MEM/MXM issue 的同激励
   逐周期 Trace。
2. 决定 FIFO 深度研究是否包含 RTL command/config/BOTH/endpoint/PRIME
   边界；Instruction DMA outstanding/RX 已可统计，但其他结构若纳入扫描仍须补齐。

本轮没有启动最优深度扫描，没有启动 Checkpoint 7C，也没有修改 VMODEL。

## 9. Phase 1 MEM 空间覆盖（CYCLE_PASS）

Phase 1 新增独立的 `raw_macro_phase1_space` 套件，不覆盖 Level 1/Level 2
基线。六个 profile 的独立 VCS/validator 与 CModel↔RTL 严格周期 Scoreboard
现已全部通过，状态更新为 **CYCLE_PASS**。

固定空间契约：

- 2 Hemisphere × 52 Slice × 2 Bank = 208 条 Bank-local issue lane；
- Shared Input Manifest v2 分开记录覆盖范围 `active_mem_slices` 与 RTL
  实例规模 `physical_mem_slices`。前者限制测试操作坐标，后者必须是每组
  4 Slice 的整数倍；`tile_baseline` 因而是 active=1、physical=4；
- 每条命令经过 4 个 Tile，Tile 不是输入选择维度；
- 正式 MEM instruction transport 为 64-bit，空间目标由 Queue/issue lane
  提供，不修改指令编码；
- 同一物理 Bank 每周期最多接受一条命令；
- RTL 对照必须使用 `USE_NATIVE_MEM=1`。

CModel 产物按 profile 分开：

- `tile_baseline`
- `bank_f4_d128`
- `slice_f8_d128`
- `full_reduced_f52_d128`
- `queue_census_f52_d128`
- `full_formal_f52_d8192`

对应 physical Slice 数依次为 4、4、8、52、52、52；active Slice 数仍为
1、4、8、52、52、52，Queue ID 始终使用完整芯片的每半区 104 条映射，
不会因缩小覆盖范围而重编号。

每个 profile 单独输出 Shared Input Manifest、CModel Trace 和结果摘要；顶层
另有 Master Manifest 与 per-coordinate Coverage CSV。Trace 的 Architecture
Identity 包含 Hemisphere/Slice/Bank/Tile/Opcode/Address/Stream/Direction，Signal
Identity 保留真实 endpoint instance，不再把所有 Bank 折叠成
`mem_e_s0_b0`。`mem_busy_derived` 是 Hemisphere 级诊断，不冒充 Bank 信号。

RTL 进入对照前的门禁：

1. Testbench 明确使用 native MEM path，而不是只支持 Bank0 的 legacy path；
2. RTL Trace 对每个事件直接给出 canonical Hemisphere/Slice/Bank/Tile；
3. RTL Shared Input parser 按 Manifest 的 Queue/issue lane 投递同一个 64-bit
   machine word；
4. 同一 Boundary/Stream 的碰撞规则未定义时，不得通过重排输入掩盖；
5. `full_formal_f52_d8192` 只做地址 0/8191 sentinel，不默认进行全叶大规模写；
6. RTL Trace 和 Scoreboard 全部通过后，才能把单个 profile 更新为
   `CYCLE_PASS`。

当前未覆盖：Gather/Scatter、有限 issue 入口反压、Local iMEM/Macro Frontend
取指路径以及 ProgramLoad→MEM 的全链逐周期对照。这些不属于 Phase 1
MEM Read/Write 空间身份验证。

### 9.1 独立 VXM 回归周期校准（STALE_CYCLE_ASSUMPTION_RESOLVED）

`subsystem.topology.vxm_repeated_distributed_input` 原测试仍按
`group + 1` 安排 MEM read，但当前 `TspSliceSystem` 默认采用
`RegisteredR1`，且 SystemPhase 先执行 VXM capture、后执行 MEM route
staging，因此 MEM 到 VXM 的静态调度延迟为 `group + 2`。本轮只将该测试
的延迟公式校准为 `group + 2`，保留 `kIssueCycle=20`、VXM 功能逻辑以及
Bundle ready 静态调度断言，不属于 CModel 功能修复。

校准后定向 CTest 7/7 通过（VXM repeated、stream topology、full-chip
topology、RMSNorm、Level 1、Level 2、Phase 1）；Level 1/2/Phase 1
Scoreboard selftest 分别为 8/8、6/6、6/6 通过。该校准本身不构成 CModel
功能修复；Phase 1 状态只在完成下述跨模型 Scoreboard 后更新为
**CYCLE_PASS**。

### 9.2 六 Profile 跨模型周期结果（CYCLE_PASS）

最终对照使用 CModel `cmodel_host_20261119_contract2` 与 VMODEL 六套稳定
VCS Trace，报告位于：

`out/icu_rtl_alignment/raw_macro_phase1_space/phase1_cycle_compare_20261119_contract2/`

六个 profile 均为 `PASS`、`difference_count=0`。合计 CModel 4913 行、RTL
5178 行；严格比较 4913 对并全部匹配，其中 Architecture Event 4476 对、
`execution_issue_ready` 265 对、`mem_busy_derived` 172 对。RTL 额外 265 行
均为明确标记 `UNMAPPED` 的 `native_command` 诊断记录；没有 X/Z、周期偏移、
字段屏蔽或额外 ignore。

这里的 case prime 仅表示正式 Case 之前准备物理 SRAM/SRF 初始条件：Read
数据先写入 SRAM，Write 所需的 boundary state 在 Case 前一正沿进入 SRF。
它不是 ICU Macro Frontend 的 PRIME，也没有修改或平移调度时间。正式 Issue、
Architecture Trace 和 `global_cycle/logical_cycle/transport_tick` 仍从 cycle 0
开始，Scoreboard 按原始周期逐项比较。

## 10. Phase 2 / P2.1 Local-iMEM / Fetch Mapping

P2.1 的 RTL Golden 是
`FTLPU-VMODEL-main/docs/rtl_instruction_path_spec/fu_frontend/mem_frontend.md`
v1.5。本节只冻结 Local-iMEM、Fetch、Response FIFO、97-bit Macro Word FIFO
和 Decoder word 输入边界。QCTL、完整 Decoder、Context FIFO、Active Context
和 MEM FU 执行属于后续门禁。

### 10.1 Canonical Signal 与 Diagnostic Signal

| RTL boundary | CModel P2.1 object | Mapping | Sample |
|---|---|---|---|
| Fetch descriptor `start_*` | `MemLocalFrontendFetchModel::Ports::start_*` | `DIRECT` | `LOW/PRE_POSEDGE` |
| Fetch→Local-iMEM `imem_read_req_*` | `Ports::imem_read_req_*` | `DIRECT` | `LOW/PRE_POSEDGE` |
| Local-iMEM→Fetch `imem_read_rsp_*` | `Ports::imem_read_rsp_*` | `DIRECT` | `LOW/PRE_POSEDGE` |
| Fetch→Word FIFO `word_*` | `Ports::word_*` | `DIRECT` | `LOW/PRE_POSEDGE` |
| Word FIFO→Decoder `word_valid_i/word_ready_o/word_i/word_last_i` | `Ports::decoder_word_*` | `DIRECT` | `LOW/PRE_POSEDGE` |
| Local-iMEM request/response stage Valid | `request_stage_valid()` / `response_stage_valid()` | Canonical Diagnostic `DIRECT` | `POSEDGE/POST_COMB` |
| Local-iMEM Outstanding | `outstanding()` | Canonical Diagnostic `DIRECT` | `POSEDGE/POST_COMB` |
| Response FIFO / Word FIFO level | `response_fifo_level()` / `word_fifo_level()` | Canonical Diagnostic `DIRECT` | `POSEDGE/POST_COMB` |
| Fetch Busy/Done/Fault/address/counts | corresponding model accessors | Canonical Diagnostic `DIRECT` | `POSEDGE/POST_COMB` |

Payload only participates in comparison while its matching Valid is high. This includes
Fetch descriptor `start_base_address_i/start_word_count_i`, which are compared only while
`start_valid_i=1`. Invalid payload is not converted to zero for cross-model comparison; it
is excluded by the Valid gate.
No fixed cycle offset, X/Z relaxation, field mask or model-specific word re-encoding is
allowed. Machine bytes are shared through
`tests/support/raw_macro_p2_1/machine_words.txt`.

The production MEM and MXM Macro decoders own typed
`LocalImemFrontendFetchModel<96/128>` value state backed by `Sram1Rw`. They copy only
finite request/response/FIFO state plus copy-on-write SRAM storage with the ICU
Current/Next snapshot. MEM and MXM use the same SRAM timing contract; SXM and other
non-integrated frontends retain the compatibility fetch model. The component is advanced
exactly once by its Decoder in LOW.

### 10.2 No formal MEM/MXM Instruction Queue

The production Raw MEM/MXM paths call `issue_active_macro()` and drive a typed direct
issue boundary. `timed_issue_depth` and `timed_issue_lookahead` are zero for all four
formal queue kinds. State advances only on issue fire; Ready low at the exact deadline
raises the Scheduler backpressure fault rather than queuing or replaying the instruction.

The following objects are functional storage boundaries, not a decoded Scheduler output
Instruction Queue:

- Local-iMEM response FIFO, depth 4;
- Macro Word FIFO, 97 bit, depth 4;
- decoded Context FIFO, 271 bit, depth 4;
- Active Context storage, 4 slots;
- QCTL per-Queue entry limit 16 or QCTL cache depth 4;
- `MemArrayModel::instruction_queues_`, which is a CModel FU execution adapter behind a
  hard-ready Issue endpoint and does not insert an ICU-visible Issue cycle.

The legacy `DistributedIcuQueue::iq_` and `mem_iq_depth` configuration remain compatibility
objects for non-Raw/semantic paths. They are not read by `tick_raw_macro()` and must not be
reported, traced or sized as the current Raw MEM hardware queue. Removing those objects
would affect unrelated legacy paths and is outside P2.1.

### 10.3 F01-F12 status contract

| ID | CModel implementation status |
|---|---|
| F01 | component directed test: single word, Last and Done |
| F02 | component directed test: four words, request II=1 and order |
| F03 | component directed test: non-zero base and final Last |
| F04 | component directed test: request stall and stable request fields |
| F05 | component directed test: response gaps without fabricated words |
| F06 | component directed test: Fetch word backpressure and payload hold |
| F07 | component directed test: four-credit boundary and blocked fifth request |
| F08 | component directed test: full Macro Word FIFO pop/push replacement |
| F09 | component directed tests: zero count and base+count overflow faults |
| F10 | component directed test: accepted wrong-Last response both terminates Fetch (`busy=0`, one-cycle `done=1`) and latches the response fault |
| F11 | component directed tests: reset and clear with outstanding requests, stale valid discarded |
| F12 | retained production-path `raw_two_issue` / `raw_repeat4` Level1 compatibility test |

The standalone F01-F11 component contract and the production MEM path use the same
canonical component. Production promotion additionally requires the frozen Level1 and
Level2/space regressions because their Parser, Context FIFO, Active Context and Scheduler
semantics are outside the standalone P2.1 component boundary.

### 10.4 Shared 1RW SRAM and SRAM FIFO model

- `timing::Sram1Rw<T,Depth>` mirrors `rtl/common/lpu_sram.sv`: active-high CEN/WEN,
  one operation per edge, write owns the port, synchronous registered read response,
  idle/write clears read-valid while read-data holds, reset clears only output registers.
- `timing::SramFifo<T,Depth>` mirrors `rtl/common/lpu_sram_fifo.sv`: logical level,
  read-pending and output-valid are distinct; at most one read is in flight; output data
  holds under backpressure; read arbitration owns the 1RW port; pointer wrap is explicit
  for non-power-of-two depths.
- MEM 208×96-bit and MXM 12×128-bit local iMEM use the primitive through the typed local
  frontend. Their response FIFO, Macro Word FIFO and Context FIFO remain ordinary FIFO;
  the standalone SRAM FIFO is not substituted for them.
- VXM/SXM have no formal Program-load local-iMEM instance in the current RTL and remain
  `NOT_IMPLEMENTED`, not silently synthesized in CModel.

### 10.5 Hardware Interface View and scoreboard scope

The retained Level1 comparison reuses the saved RTL instruction-path Trace and compares
410 ruled pairs with zero mismatch. The formal CModel Hardware Interface View covers
Program Load, local-iMEM read request/response, data/valid/last, backpressure, FIFO level,
decoded Context, Active Context state and direct FU issue events. These are the
architecture-visible cycle boundaries used for CModel/RTL acceptance.

Physical SRAM macro pins are an RTL-only Diagnostic. CModel is not required to expose
same-named CEN/WEN/address/write-data/read-data/read-valid pins, and no separate pin-level
scoreboard is an acceptance gate. The `Sram1Rw` and `SramFifo` directed tests remain the
CModel-side contract checks for synchronous read, single-port arbitration, reset,
backpressure and occupancy behavior.

## 11. Phase 2-B QCTL canonical ownership and timing

Phase 2-B keeps the shared Pool physical organization while separating three state
owners. `IcuQctlPool` owns partition base/count/generation, physical address mapping,
the single read port and load/configuration faults. Each `IcuQctlCache` owns only its
fetch offset, retire offset, in-flight read bookkeeping and cached head records. Each
`IcuQctlController` owns only the five-state QCTL semantic machine (`IDLE`,
`SYNC_WAIT`, `EVENT_WAIT`, `NOTIFY_WAIT`, `MEMPACKET_WAIT`). The Controller has no
local Scheduler cycle counter: `scheduler_cycle` is an input and
`scheduler_counter_ce` is its only counter-control output.

| RTL-visible boundary | CModel owner | Cycle contract |
|---|---|---|
| Pool load/configuration and partition map | `IcuQctlPool` | sampled before the edge; accepted state commits once at the edge |
| Cache read request/response and head | `IcuQctlCache` | one shared Pool grant per cycle; response belongs to the granted Queue and matching generation |
| Control state and head retirement | `IcuQctlController` | blocking record suppresses Scheduler CE on its hit cycle; wait states keep CE low; release adds no cycle offset |
| SYNC arrival / NOTIFY acceptance / Release | `IcuQctlArrivalReleaseBarrier` | arrival bits are idempotent; NOTIFY is held by its Controller; all participants must arrive before one NOTIFY handshake; Release is registered and clears the round |
| 32-bit Event TAG broadcast | `IcuQctlEventManager` | early events are retained by the Event side, never by a Controller; matching waiters consume the broadcast |
| MEM_PACKET ordinal request | Controller plus external packet-retire input | request and ordinal remain stable until the matching retirement; SRAM write Commit is a distinct Event source |
| Endpoint observation | `IcuQctlHardwareInterfaceView` | observer-only snapshot; it owns no architectural state |

The formal SYNC/NOTIFY path contains no historical notification Token or release-credit
counter. A SYNC contributes one bit to the current round's arrival mask. An early
NOTIFY does not create a credit and does not retire; it remains `valid` in
`NOTIFY_WAIT`. Once every participant has arrived, the accepted NOTIFY creates one
Release vector, all waiting SYNC records retire from that vector, and the arrival state
is cleared before the next round. Reset also clears arrival, Release and Controller
state. The old `legacy_release_credits_` member in `DistributedIcuQueue` is isolated to
the non-formal compatibility path; `qctl_endpoint_active()` never increments or consumes
it, and the production `Icu` notification routes skip formal QCTL endpoints.

The production call chain is:

1. `ProgramLoader` seals the image and configures QCTL Pool partitions.
2. frontend PRIME fills each QCTL Cache and publishes primed/done status.
3. ICU LOW copies Current to Next and calls the full `dispatch()` path.
4. `begin_qctl_cycle()` evaluates Pool/Cache/Controller/Fabric from old state and drives
   each Queue's formal endpoint snapshot.
5. MEM/MXM direct-issue paths consume their endpoint CE/handshakes without an added
   Scheduler-to-FU instruction queue.
6. `complete_qctl_cycle()` commits the QCTL state exactly once; the ICU positive edge
   then publishes the complete Next state.

The shared Phase 2-B stimulus contains 46 actions across no-control, WAIT_EVENT,
SYNC/NOTIFY, MEM_PACKET, Pool/Cache arbitration and fault/reset cases. The CModel trace
records 10 canonical boundaries per physical cycle. Passing this directed CModel suite
and the production ProgramLoader test establishes `READY_FOR_QCTL_JOINT_SCOREBOARD`;
it is not a claim of cross-model `CYCLE_PASS` until the same stimulus is compared with
the RTL trace.

## 12. Phase 3 production-path closure (2026-11-21)

This section supersedes the earlier `PARTIAL` statements for the verified Phase 3 scope.
The comparison is bound to the current local RTL working tree, the shared directed input,
`PRE_POSEDGE` sampling and `cycle_offset=0`. It does not promote untested FU internals to
system-level coverage.

### 12.1 RTL to CModel production mapping

| Phase 3 boundary | RTL owner | CModel owner / view | Status |
|---|---|---|---|
| Command RAM, command FSM, Load Start | `lpu_program_load_controller` | `ProgramLoadCommandController`, `ProgramLoadHostAdapter`, `ProgramLoader::tick_canonical_before_dma()` | `CYCLE_PASS` |
| Config FIFO | `lpu_program_load_config_fifo` | `ProgramLoadConfigFifo` | `CYCLE_PASS` |
| DDR response / DMA beat | `lpu_instruction_dma` | `InstructionFetchDma` and `ProgramLoadR2HardwareInterfaceView::dma` | `CYCLE_PASS` |
| Reassembler / Word FIFO | `lpu_program_load_reassembler`, `lpu_program_load_word_fifo` | `ProgramLoader` reassembly state, word FIFO and `ProgramLoadR2HardwareInterfaceView` | `CYCLE_PASS` |
| Distributor / Elastic / Relay | `lpu_program_distributor`, `lpu_program_distribute_elastic_stage`, `lpu_program_distribute_relay_chain` | `ProgramDistributionTransport` and `ProgramLoadR3HardwareInterfaceView` | `CYCLE_PASS` |
| MEM/MXM Program Endpoint | MEM/MXM local Frontend load/PRIME ports | `DistributedIcuQueue::program_endpoint_state()`, staged `InstructionControlUnit` program operations | `CYCLE_PASS` |
| QCTL distributed program store | QCTL Load Manager, relay chain, Endpoint and global subsystem | `IcuQctlDistributedProgramStore`, QCTL Pool/Cache/Controller and `IcuQctlHardwareInterfaceView` | `CYCLE_PASS` |
| FU Stream boundaries | `lpu_fu_subsystem_cluster` and registered SRF boundaries | `StreamRegisterFabric`, `StreamTopology`, FU port maps | `CYCLE_PASS` |
| MEM Frontend to direct Issue | MEM local iMEM, Macro Frontend, Context and Scheduler | `DistributedIcuQueue` raw Macro path and direct typed Issue view | `CYCLE_PASS` for the directed R8 scope |

VXM and SXM do not have a formal Program-load local iMEM in the current RTL. Their R6
status is `NOT_APPLICABLE`, not an omitted CModel implementation. No FU Scheduler output
Instruction Queue, C+2 lookahead or extra issue register was reintroduced.

### 12.2 Lifecycle and sampling contract

- Command DONE is an R1 control result. After `Started`, `Draining` or `Completed`, it
  cannot drive the CModel back through `AllCommitted -> Prime -> Ready`.
- `program_ready`, `program_running` and `program_done` follow the RTL Program-load
  lifecycle. Host `kernel_done` remains the stricter C++ drain predicate and is not used
  as the canonical RTL `program_done` signal.
- An unconfigured formal Program Endpoint is ready. A configured compatibility Native
  queue without Endpoint WRITE state still waits for actual local-frontend prefetch.
- Cycle 0 asserts Program reset in both models. The equivalent early Start request remains
  asserted through the first post-reset sampling cycle because RTL reset clears its sticky
  pending state while the Host Adapter preserves the software request.
- A 92-row observation window captures equal idle tails for the R3/R4/R8 reports. It does
  not delay state or events and is not a cycle offset.

### 12.3 Final directed evidence

| Scope | Result |
|---|---|
| R2 DMA/Reassembler/Word FIFO | `R2_PASS`, 0 differences |
| R3 Distributor/Relay/Elastic | `R3_PASS`, 0 differences |
| R4 MEM and MXM Endpoints | `R4_PASS`, 0 differences |
| R5 QCTL distributed path | 444/444 matched, 0 differences |
| R7 six FU Stream boundaries | 260/260 matched, 0 differences |
| R8 Program Load to MEM Issue | Joint `PASS`, 0 differences, MEM Issue cycles 53 and 57 in both models |

The R8 joint event counts are identical: four command writes, one Load Start, one Load
transaction, one Config enqueue/dequeue, two DDR and DMA response fires, three
Reassembler/Word-FIFO/MEM Program transfers, four Distributor/Elastic/East-route Endpoint
transfers (three WRITE plus one PRIME), and two MEM Issues. No fixed offset, hard padding,
X/Z widening or unjustified field masking is used.

The current QCTL conclusion is now `QCTL_PHASE3_DISTRIBUTED_CYCLE_PASS`; it supersedes the
earlier `READY_FOR_QCTL_JOINT_SCOREBOARD` wording at the end of Section 11.

## 13. New MXM Core hardware interface view (2026-11-22)

This section records the direct standalone-Core boundary introduced after Phase 3.  It
does not change the verified Program-load/MXM-local-iMEM mapping above.  The old
`MxmControlSlice` interface is no longer used by `TspSliceSystem`; Legacy ICU MXM words
are intentionally not translated inside the cycle model.  Macro decode and final stream
formatting remain outside this boundary.

Default geometry is ROWS=4, COLS=4, TERMS=8, OUTPUTS=8, command Queue depth 16,
Weight Stream count 32 and Accumulator depth 1024.  All inputs are sampled in LOW from
old Current, all registered state commits at POSEDGE, HIGH is observation-only and the
Core has no NEGEDGE state.  `valid && ready` defines enqueue/launch Fire; CModel `*_fire`
members are derived observations and are not additional RTL pins.

| Canonical RTL boundary | RTL owner / width | CModel owner / representation | Direction and cycle contract | Status |
|---|---|---|---|---|
| `rst_ni`, `run_i` | `lpu_mxm_core`; 1 bit each | `MxmCore::Inputs` | RTL→Core; active-low reset, run hold/reset semantics sampled before POSEDGE | `DIRECT` |
| `compute_enqueue_valid_i`, `compute_enqueue_ready_o`, `compute_launch_i`, `compute_launch_ready_o`, `compute_instruction_i` | Core/Frontend; 1/1/1/1/7 bits | `MxmCore::Inputs/Outputs`, `ComputeInstruction` | Queue push/pop at Fire; launch starts registered row Wave | `DIRECT` |
| `load_enqueue_valid_i`, `load_enqueue_ready_o`, `load_launch_i`, `load_launch_ready_o`, `load_instruction_i` | Core/Frontend; 1/1/1/1/2 bits | `MxmCore::Inputs/Outputs`, `LoadInstruction` | Queue push/pop at Fire; Matrix/Column launch drives fixed-length load state | `DIRECT` |
| `param_load_*`, `param_load_instruction_i` | Core/Frontend; handshake plus 1-bit payload | `MxmCore::Inputs/Outputs`, `ParamLoadInstruction` | independent Queue/Wave; parameter-buffer ID preserved | `DIRECT` |
| `dequant_*`, `dequant_instruction_i` | Core/Frontend; handshake plus 1-bit payload | `MxmCore::Inputs/Outputs`, `DequantInstruction` | independent Queue/Wave; D1/D2 and D3 write timing is registered | `DIRECT` |
| `backend_*`, `backend_instruction_i` | Core/Backend; handshake plus 13-bit payload | `MxmCore::Inputs/Outputs`, `BackendInstruction` | opcode[1:0], address[11:2] logical 10 bits, out_bf16; relay then Context/Bank/ALU pipelines | `DIRECT` |
| `activation_valid_i`, `activation_i` | Core/Frontend; 4 and 4×8×16 bits | `Inputs::activation_valid_i/activation_i` | Core input sampled in LOW; one registered horizontal link per column | `DIRECT` |
| `weight_stream_valid_i`, `weight_stream_i`, `weight_stream_consume_o` | Core/Frontend; 4×32×8 valid/data bytes | corresponding `Inputs/Outputs` arrays | per-row/per-stream/per-lane valid and consume; data is unsigned packed byte, zero-point is signed when decoded | `DIRECT` |
| `*_row_valid_o`, `*_row_instruction_o`, `compute_row_tag_o`, `load_row_index_o` | Core/Frontend registered Waves | corresponding `MxmCore::Outputs` arrays | post-commit observation of the current Wave stage | `DIRECT` |
| `cell_weight_*`, `cell_activation_*` | Core/Frontend; 4×4 bits per signal | corresponding `Outputs` matrices | cell-boundary control observations, no SRAM macro pins | `DIRECT` |
| `partial_valid_o`, `partial_o`, `partial_tag_o` | Core/Frontend; 4, 4×8×32, 4×16 bits | corresponding column arrays | top partial after Dot/Partial pipelines | `DIRECT` |
| `returned_valid_o`, `returned_o`, `returned_tag_o` | Core/Result Return; 4, 4×8×32, 4×16 bits | corresponding row arrays | registered down/left return path | `DIRECT` |
| `backend_issue_*`, `backend_slot_*` | Core/Backend; per-column valid/bf16/op/address | corresponding `MxmCore::Outputs` arrays | issue Wave and Context slot observations | `DIRECT` |
| `alu_valid_o`, `operand_a_o`, `operand_b_o`, `alu_address_o`, `alu_tag_o` | Core/Backend | corresponding `MxmCore::Outputs` arrays | valid only when partial, Context and synchronous SRAM response align | `DIRECT` |
| `bank_request_valid_o`, `bank_write_o`, `bank_address_o` | Core/Backend; 4×2 requests, 9-bit bank row | corresponding `Outputs` arrays | architectural logical-bank request view; physical macro CEN/WEN pins remain RTL Diagnostic only | `DIRECT` |
| `result_valid_o`, `result_bf16_o`, `result_o`, `result_address_o`, `result_tag_o` | Core/Backend output | corresponding `MxmCore::Outputs` arrays | registered Backend output, FP32 or converted BF16 per command | `DIRECT` |
| `frontend_fault_o`, `backend_fault_o`, `fault_o` | Core; sticky fault outputs | corresponding `MxmCore::Outputs` booleans | sticky until reset/run reset, aggregate is OR | `DIRECT` |
| CModel `*_enqueue_fire`, `*_launch_fire` | no physical output port | `MxmCore::Outputs` derived fields | `valid && ready` at LOW sample; trace convenience only | `DERIVED` |
| Legacy `MxmControlInstruction` → new five Command payloads | no one-to-one RTL adapter | ICU Legacy Queue vs `MxmCore::Inputs` | no implicit conversion; compiler/decoder migration is outside this model | `CONFLICT` |
| SRAM macro pins | RTL implementation detail | deliberately absent | Pin-level comparison is outside the CModel canonical layer | `NOT_APPLICABLE` |

Source evidence: RTL `rtl/mxm/lpu_mxm_core.sv`, `lpu_mxm_frontend.sv`,
`lpu_mxm_result_return.sv`, `lpu_mxm_backend.sv`, `lpu_mxm_accumulator_bank.sv`,
`rtl/mxm/icu/lpu_mxm_isa_pkg.sv` and `lpu_mxm_backend_isa_pkg.sv`; CModel
`include/ftlpu/mxm/core.hpp`, `include/ftlpu/system/tsp_slice_system.hpp` and
`tests/unit/mxm/core_clocked_test.cpp`.  This mapping establishes
`MXM_INTERFACE_UNIFIED`; it is not a module-level CModel↔RTL `CYCLE_PASS` claim.

## 14. MXM Datapath C1-C6 module-level cycle mapping (2026-11-23)

This section closes the direct-module Datapath scope that follows the interface view in
Section 13.  Equivalent legal inputs are applied at the real module boundary.  The RTL
Testbench does not bypass internal state, and the CModel harness advances production
`ClockedModule` objects through `LOW -> validate -> commit -> HIGH -> NEGEDGE`.  Both traces
use `PRE_POSEDGE` and `POST_POSEDGE` samples with `cycle_offset=0`.

| Scope | RTL owner | CModel owner / canonical view | Compared architectural behavior | Result |
|---|---|---|---|---|
| C1 arithmetic | `lpu_mxm_bf16_dot_csa`, `lpu_mxm_fp32_add_pipe`, `lpu_mxm_output_pipe` | `Bf16DotCsaPipe`, `Fp32AddPipe`, `OutputPipe` in `include/ftlpu/mxm/datapath_clocked.hpp` | input/valid/tag, Dot stage valid/tag, Add A1/A2/output, FP32/BF16 result, bubbles and reset flush | `MXM_C1_CYCLE_PASS`, 1792/1792 |
| C2 Dequant / Weight Buffer | `lpu_mxm_weight_dequant_pipe`, `lpu_mxm_weight_buffer` | `WeightDequantPipe`, `WeightBufferPipe`, shared `shift_weight_buffer()` | INT8/scale/zero-point, D1/D2/D3, write accept/forward, buffer select/ready/busy, stored weight valid/data | `MXM_C2_CYCLE_PASS`, 1658/1658 |
| C3 Supercell | `lpu_mxm_supercell` and its Dot/Add instances | `SupercellPipe` | activation and weight propagation, buffer select, all group Dot stages, partial/tag/valid and fault | `MXM_C3_CYCLE_PASS`, 1360/1360 |
| C4 Chain / 4x4 Array | formal 4x4 `lpu_mxm_supercell` array wiring | sixteen `SupercellPipe` objects on one CModel `Clock` | every cell boundary: horizontal activation/weight, vertical partial, valid/tag and parallel operation | `MXM_C4_CYCLE_PASS`, 59840/59840 |
| C5 Result Return | `lpu_mxm_result_return` | `ResultReturnPipe` and shared `advance_result_return()` used by `MxmCore` | all top inputs, all registered down/left links, diagonal turn and four left endpoints | `MXM_C5_CYCLE_PASS`, 504/504 |
| C6 Backend Column | `lpu_mxm_backend_column`, `lpu_mxm_accumulator_bank`, `lpu_mxm_fp32_add_pipe`, `lpu_mxm_output_pipe` | `BackendColumnPipe` | command/context, logical bank request/read/write, synchronous response, ALU, pending hazard state, DIRECT_OUT/STORE/ACC_STORE/ACC_OUT and output | `MXM_C6_CYCLE_PASS`, 1960/1960 |

The six Scoreboards compare 67114/67114 pairs with 0 differences.  They compare canonical
module ports and the minimum registered state required to establish cycle behavior; SRAM
macro pins are not part of the CModel contract.  In C2, 52 RTL data samples contain X/Z only
while their associated weight Valid is zero.  Valid is always compared, and data is gated
strictly by that invalid state rather than normalized to zero.

`shift_weight_buffer()` and `advance_result_return()` are shared with the integrated
`MxmCore` production path.  `SupercellPipe` and `BackendColumnPipe` are the formal modular
CModel timing components used at their real hardware boundary; this C1-C6 result does not
claim C7 Command/Control alignment or C8 full-Core joint alignment.  The integrated system
path remains `TspSliceSystem -> MxmCore`, Legacy MXM is not restored, and no formal MXM
Datapath RTL function was changed for this comparison.

Evidence is retained under `out/mxm_datapath_cycle/c1` through `c6`.  The corresponding
CModel tests are `tests/unit/mxm/datapath_c*_cycle_trace_test.cpp`; RTL observation-only
Testbenches are `sim/tb/mxm/lpu_mxm_c*_cycle_trace_tb.sv`; Scoreboard contracts are
`tools/cycle_scoreboard/mxm_datapath_c*_rules.json`.

## 15. MXM five-stream instruction path C7-C8 closure (2026-11-23)

This section supersedes the C7/C8 exclusion at the end of Section 14 for the directed
scope defined below.  It does not change the C1-C6 evidence and does not restore Legacy
MXM or an ICU post-Scheduler Instruction Queue.

### 15.1 Production topology and canonical mapping

The production path is
`ProgramLoader -> five independent local iMEM/Fetch/Decoder/Context/Scheduler lanes ->
QCTL -> MxmCore native command ports`.  Each Scheduler owns its absolute-cycle context
and advances only when the destination reports Ready and the corresponding Fire occurs.
There is no decoded ICU queue after the Scheduler.  The native queues inside `MxmCore`
remain part of the formal MXM implementation; when one is empty, its enqueue payload is
available to the same-cycle launch path exactly as in RTL.

| Stream | Program target kind | Native width | QCTL queues | CModel command/view | Core boundary |
|---|---:|---:|---:|---|---|
| Load | 1 | 2 | 208..211 | `MxmLoadCommand`, `mxm_load_queue()` | `load_enqueue_*`, `load_launch_*` |
| Compute | 3 | 7 | 216..219 | `MxmComputeCommand`, `mxm_compute_queue()` | `compute_enqueue_*`, `compute_launch_*` |
| Dequant | 2 | 1 | 212..215 | `MxmDequantCommand`, `mxm_dequant_queue()` | `dequant_enqueue_*`, `dequant_launch_*` |
| Param Load | 7 | 1 | 220..223 | `MxmParamLoadCommand`, `mxm_param_load_queue()` | `param_load_enqueue_*`, `param_load_launch_*` |
| Backend | 8 | 13 | 224..227 | `MxmBackendCommand`, `mxm_backend_queue()` | `backend_enqueue_*`, `backend_launch_*` |

Program destination is 13 bits: hemisphere bit 12, target kind bits 11:8 and local
index bits 7:0.  QCTL has 228 queues and R6 directory size 18.  CModel ownership is split
between `include/ftlpu/icu/distributed_queue.hpp`, `include/ftlpu/system/icu.hpp`,
`include/ftlpu/system/program_loader.hpp`, `include/ftlpu/icu/qctl_distributed_program_store.hpp`
and `include/ftlpu/mxm/core.hpp`.  Typed command encoding is centralized in
`include/ftlpu/mxm/instruction.hpp`; the raw Macro decoder emits the five canonical native
commands rather than a Legacy MXM control word.

During PRIME the Scheduler cycle is intentionally held at zero while the whole Macro image
is decoded.  Therefore the decoder-side PRIME admission check must not apply the runtime
`AdmissionLookahead` window to future absolute start cycles.  Runtime Scheduler admission
and late-cycle checks are unchanged.  This rule permits legal contexts such as start cycle
35 and 58 to be prepared without adding a cycle offset or an output register.

### 15.2 Sampling and joint evidence

Scheduler-to-Core Ready/Valid/Fire is sampled at `LOW/PRE_POSEDGE`.  Wave, Datapath,
Result Return and Backend registered outputs are sampled at `HIGH/POST_POSEDGE` after the
single validate/commit barrier.  An RTL Fire monitor must latch `valid && ready` before the
accepting edge; recomputing it after the edge is invalid because the Scheduler may already
have retired the context and deasserted Valid.

| Scope | Compared behavior | Result |
|---|---|---|
| C7 five-stream Frontend/Control | five Issue Valid/Ready/Fire signals, native instruction, absolute issue cycle, conflict and frontend fault | 111/111 matched, 0 differences |
| C7 five-stream Local Frontend | iMEM request/response, Fetch word, Decoder word, Context enqueue/dequeue, Word/Context FIFO levels, Active Context, Fetch Complete, Decoder Done, Ready and fault | 2430/2430 matched, 0 differences |
| C8 full directed path | real Program Load, five local Frontends, direct Core Fire, five Core command queue levels, row waves, cell valids, partial/return/backend/result payloads and faults | `MXM_FULL_PATH_CYCLE_PASS`, 3589/3589 matched, 0 differences |

The non-overlapping C7 Frontend and Scheduler-to-Core comparisons total 2541/2541.
Core enqueue/launch bypass, all five command queue levels and Instruction Wave are then
closed inside the integrated C8 comparison rather than counted a second time.  The combined
evidence establishes `MXM_C7_CYCLE_PASS`.

The C8 legal scenario performs Param Load at cycle 0, Load and Dequant at cycle 1,
32 consecutive Dequant issues through cycle 32, Compute at cycle 35 and Backend
`DIRECT_OUT` at cycle 58.  Both traces contain 91 rows, 36 command Fires, four partials,
four returned tokens and four results.  C7 records 210 RTL X values only on invalid
response/word Last payloads; all are explicitly gated by their matching Valid signal and
reported as gated X/Z coverage.  No X/Z is converted to zero.  No fixed offset, hard
padding, X/Z widening or required-field masking is used.

CModel directed regression is 11/11 PASS, including the five-stream Frontend, production
Program Loader path, QCTL, full topology, `MxmCore` and both C7/C8 trace tests.  Formal
VMODEL source was read-only; VCS used an observation-only Testbench copy under `/tmp` and
did not modify RTL behavior.  Evidence is retained under `out/mxm_full_path_cycle/c7` and
`out/mxm_full_path_cycle/c8`.  Together with the frozen C1-C6 baseline, the explicitly
compared MXM C1-C8 scopes total 73244/73244 matched with 0 differences and establish
`MXM_FULL_PATH_CYCLE_PASS` for this directed production path, not for untested programs or
other FUs.

The corresponding stage verdicts are `MXM_INSTRUCTION_PATH_READY`,
`MXM_FULL_CLOCK_MODEL_READY`, `MXM_FULL_INTERFACE_UNIFIED` and
`MXM_FULL_PATH_CYCLE_PASS`.  They refer to the five-stream directed production scope
documented here; they do not extend to VXM, SXM or arbitrary full-system programs.

## 16. QCTL Runtime Control with MEM/MXM issue closure (2026-11-23)

This joint directed test closes the runtime-control gap between the formal QCTL controller
and real MEM/MXM instruction issue.  It uses the production `C2cDmaSystem` CModel and the
RTL `lpu_fu_subsystem_cluster`; it does not restore a post-Scheduler Instruction Queue and
does not use Legacy MXM.  MXM is represented only by the five native `MxmCore` streams.

### 16.1 Canonical mapping and timing contract

| Canonical behavior | RTL source boundary | CModel source boundary |
|---|---|---|
| Event ingress | `qctl_event_valid/tag/targets` into the global QCTL subsystem | `IcuQctlRuntime::drive_inputs()` through `TspSliceSystem::drive_qctl_runtime_inputs()` |
| Runtime head and logical cycle | QCTL endpoint/controller head and scheduler cycle | `IcuQctlRuntime::queue_cycle_status()` |
| Barrier and notification | SYNC arrival/release and NOTIFY valid/ready | `IcuQctlRuntime` Pool/Cache/Controller hardware view |
| Event wait/consume | event wait, matching tag and consume result | `IcuQctlRuntime` Event Buffer and Controller outputs |
| MEM_PACKET retirement | top-level `qctl_mem_packet_retired` input | canonical external retirement input to `TspSliceSystem` and `DistributedInstructionQueue` |
| Scheduler clock enable | endpoint/controller CE into the selected queue | `scheduler_ce` from the post-edge QCTL cycle status |
| MEM issue | MEM Scheduler Valid/Ready/Fire, instruction and absolute cycle | production MEM distributed queues 0 and 1 |
| MXM issue | Load/Dequant/Compute/Param/Backend Scheduler Valid/Ready/Fire | production queues 208/212/216/220/224 and the native `MxmCore` command views |

Both sides are sampled at `HIGH/POST_POSEDGE`.  Global cycle 0 is the first observation
immediately after START has committed.  The Event Buffer makes an accepted, non-backpressured
event available to the controller evaluation associated with that edge; it is not delayed by
an extra software-only staging cycle.  A formal top-level MEM_PACKET retirement input is
consumed before the same cycle's Scheduler CE is observed, matching the RTL Current/Next
ownership rather than adding a compensating offset.

Payload comparison is validity-gated by its architectural Valid: head cycle/opcode/operand
by `head_valid`, MEM_PACKET ordinal by `mem_packet_valid`, issue payload/cycle/Ready by
`issue_valid`, and event tag by `event_in_valid`.  The matching Valid, Fire, CE, fault and all
other control fields remain mandatory comparisons.  This is invalid-payload qualification,
not value masking; no fixed cycle offset, hard padding or widened X/Z tolerance is used.

### 16.2 Joint evidence and boundary

The shared input contains QCTL SYNC/Release, NOTIFY, Event Wait/Consume, MEM_PACKET retire,
pause/resume through Scheduler CE, two MEM issue streams and the five native MXM issue
streams.  Each trace has 29 cycles x 7 queues = 203 rows and contains 9 actual issue events.
The normalized Scoreboard compares 3450/3450 pairs with 0 differences and reports
`QCTL_RUNTIME_ISSUE_CYCLE_PASS`.

Evidence is retained under `out/qctl_runtime_issue_joint_current`: CModel and RTL CSV traces
under `cmodel` and `rtl`, and the normalized report under `scoreboard`.  The CModel directed
test is `tests/subsystem/system/qctl_runtime_issue_joint_trace_test.cpp`; the RTL
observation-only Testbench is `sim/tb/lpu_qctl_runtime_issue_joint_tb.sv`; the shared adapter
and contract are `tools/cycle_scoreboard/qctl_runtime_issue_joint_scoreboard.py` and
`qctl_runtime_issue_joint_rules.json`.

The CModel QCTL unit regression and Phase 2B trace regression also pass after the runtime
timing correction.  VCS compilation and dynamic simulation pass for the joint Testbench.
No formal RTL function was changed: VMODEL changes are confined to the Testbench, filelist
and run script.  This result closes the documented directed QCTL Runtime + MEM/MXM issue
scope; it does not claim VXM/SXM coverage, arbitrary-program coverage or Full LPU
Integration.

## 17. MEM-SR-SXM CModel clock ownership and interface view (2026-11-24)

This checkpoint is CModel-only. It establishes the production `TspSliceSystem` clock
ownership and a sparse hardware-interface view whose names follow the current local RTL.
VMODEL was used read-only as the naming and semantic reference. VCS, RTL Trace and a
CModel/RTL Scoreboard were intentionally not run in this phase.

### 17.1 Clock ownership and LOW dependency order

`MemArrayModel`, `StreamRegisterFabric` and `SxmSlice` remain separate
`ClockedModule` owners registered with the one global `Clock`. Their Current/Next state is
validated and committed by the same POSEDGE barrier; no second clock or local commit loop was
introduced. HIGH remains observation-only.

The production LOW order is now explicit rather than hidden inside one combined callback:

1. SR begins a cycle and exposes the old Current state.
2. ICU dispatch prepares native commands.
3. MEM reads SR Current and stages SR Next.
4. SXM reads SR Current and stages SR Next.
5. C2C and passive SR routes read Current and stage Next.
6. The global Clock validates and commits all registered state once.

Consequently a producer's SR Next is never visible to another module in the same LOW merely
because of C++ call order. `RegisteredR1` MEM reads sample SRAM at edge A, stage the saved
response into SR Next at the following LOW/edge B, and become observable as SR Current only
after edge B. `MemArrayModel::read_response_latency_cycles()` makes that extra registered
cycle an explicit schedule contract. The wide MEM-to-SXM regression now derives its launch
cycle from this contract instead of assuming Main-read timing.

SR collision and consume behavior also follows the leaf RTL contract. Two or more candidates
for one physical cell produce no winner for that cell and assert its leaf `collision_o`; other
cells still commit. Consuming an invalid cell asserts `invalid_consume_o` without aborting
unrelated state updates.

### 17.2 Canonical RTL interface map

All CModel interface samples below are PRE_POSEDGE/LOW samples. Sparse vectors contain only
asserted packed-port elements; their location metadata is the C++ equivalent of the RTL flat
index. This is representation compression, not an event abstraction or a second state owner.

| Boundary | RTL canonical fields | CModel hardware view | Mapping | Width and semantic notes |
|---|---|---|---|---|
| SR leaf/fabric | `inject_valid_i`, `inject_data_i`, `consume_i`, `collision_o`, `invalid_consume_o`, `state_valid_o`, `state_data_o` | `StreamRegisterFabric::HardwareInterfaceView` | DIRECT / DERIVED packed indexing | 8-bit lane cell; 8 lanes; 32 streams/direction. Status is flattened by `{direction,column,superlane}`. State outputs sample old Current during LOW. |
| MEM issue | `bank_issue_valid_i`, `bank_issue_i` | `MemArrayModel::HardwareInterfaceView` | DIRECT | One asserted sparse entry identifies `{slice,bank,tile}`; native MEM payload is 32 bits. |
| MEM/SR data | `mem_boundary_consume_o`, `mem_producer_valid_o`, `mem_producer_data_o`, `mem_producer_direction_o`, `mem_producer_stream_o`, `mem_producer_boundary_o` | matching `MemArrayModel::HardwareInterfaceView` fields | DIRECT / DERIVED packed indexing | Producer segment is 64 bits (8 x 8-bit lanes); direction is 1 bit, stream index 5 bits, boundary 4 bits in the current RTL integration. |
| MEM status | `mem_fault_valid_o`, `mem_busy_o` | matching scalar view fields | SEMANTIC | Active-high LOW observation; no separate CModel shadow state is added. |
| SXM commands | `run_i`, `transpose_issue_valid_i`, `transpose_issue_instruction_i`, `permute_issue_valid_i`, `permute_issue_instruction_i` | `SxmSlice::HardwareInterfaceView` | DIRECT for encodable canonical commands | Each instruction payload is 416 bits. The view is populated from the real production pending command, not a test setter. |
| SXM boundaries | `east_inner_valid_i/data_i`, `east_outer_valid_o/data_o`, `west_outer_valid_i/data_i`, `west_inner_valid_o/data_o` | matching sparse boundary samples in `SxmSlice::HardwareInterfaceView` | DIRECT / DERIVED packed indexing | Four tiles x 32 streams per direction; each data segment is 64 bits. Inputs record actual Current consumption and outputs record actual SR Next staging. |
| SXM status | `fault_o`, `conflict_o` | matching scalar view fields | SEMANTIC | Current CModel faults/conflicts are summarized at the subsystem boundary. Exact sub-fault decomposition remains for the later cycle-alignment phase. |
| CModel-only SXM modes | legacy `Distribute`, `ShiftSelect`, wide/broadcast encodings without a canonical 416-bit command | no fabricated RTL command assertion | NOT_APPLICABLE | Functional CModel behavior is preserved, but canonical issue Valid is omitted when no current RTL encoding exists. |
| SXM Program Load | RTL `program_valid_i/.../program_ready_o` | existing CModel packed-Macro local frontend | CONFLICT / UNMAPPED | Current RTL `lpu_sxm_subsystem` holds Program Load Ready low because local-iMEM transport is not implemented. No false DIRECT mapping is claimed. |

Canonical RTL sources are `rtl/srf/ftlpu_sr_hemisphere_fabric.sv`,
`rtl/srf/ftlpu_sr_superlane_col_dir.sv`,
`rtl/integration/lpu_mem_srf_sxm_turnaround_integration.sv`, and
`rtl/sxm/lpu_sxm_subsystem.sv` in the current local VMODEL tree.

### 17.3 Directed CModel evidence and boundary

A clean targeted build passes 16/16 affected tests: two SR tests, six MEM tests, five SXM
unit tests, the new `subsystem.system.mem_sr_sxm_clock` test, the two-matrix wide-transpose
smoke test, and the full 96-matrix East/West MEM -> packed-Macro SXM -> MEM test. The new
integration test checks RegisteredR1 sample/response/SR commit, one-column passive routing,
the four-phase count, real Packed-Macro SXM issue, canonical 416-bit Transpose visibility,
lane-accurate consume and single state advancement.

The independently existing `subsystem.system.timing` test still fails at its MXM-only
`compute_queue_level() == 1` assertion. No MXM code was changed to hide that failure, and it
is not counted in the 16/16 MEM-SR-SXM result.

Within this directed CModel scope the verdicts are `CMODEL_MEM_SR_SXM_CLOCK_READY` and
`MEM_SR_SXM_INTERFACE_UNIFIED`. They do not claim RTL cycle equivalence, Full LPU coverage,
or coverage of SXM Program Load transport that the current RTL does not implement.

## 18. MEM-SR cycle alignment and authoritative SXM boundary correction (2026-11-24)

This checkpoint adds real cross-model evidence for the portions that do not depend on an SXM
instruction path. It also corrects the acceptance boundary after the project owner identified
the newer triangle implementation as authoritative.

### 18.1 S1 StreamRegisterFabric

The CModel `StreamRegisterFabric` was compared against
`rtl/srf/ftlpu_sr_hemisphere_fabric.sv` and its leaf implementation using identical directed
drives at `LOW/PRE_POSEDGE` and registered observations at `HIGH/POST_POSEDGE`. The compared
signals are `inject_valid_i`, `inject_data_i`, `consume_i`, `collision_o`,
`invalid_consume_o`, `state_valid_o` and `state_data_o`. East/West cells, consecutive writes
and consumes, collision with no winner, invalid consume, unrelated-cell commit, same-cycle
consume/write and reset state are included.

Result: `SR_CYCLE_PASS`, 119/119 matched, 0 differences, 0 RTL X/Z samples.

### 18.2 S2 MEM to SR

The current formal RTL integration exposes the native Registered-R1 MEM read path. The
CModel comparison therefore uses `MemArrayModel::ReadTiming::RegisteredR1`; CModel `Main`
remains `NOT_APPLICABLE_TO_CURRENT_RTL` and was not disguised as an equivalent path. The
comparison covers native bank issue, boundary input, MEM consume/producer identity and data,
SR Current/Next visibility, fault and busy.

The first run found one real semantic mismatch at cycle 11: CModel `mem_busy_o=1`, RTL
`mem_busy_o=0`. RTL defines busy from the command pipeline only; a registered read response
does not extend it. `MemArrayModel::HardwareInterfaceView` was corrected to scan only current,
next and queued command state. SRAM contents, request/response timing and transfer behavior
were not changed.

Result: `MEM_SR_CYCLE_PASS`, 441/441 matched, 0 differences, 0 RTL X/Z samples. The affected
CModel MEM read-mode, clocked and synchronous-read regressions pass. Formal RTL function was
not modified; VMODEL changes are observation-only Testbenches/filelists.

### 18.3 Authoritative SXM and stop boundary

The authoritative new SXM is `rtl/sxm_native/sxm_triangle_core.sv`, with the standalone
integration path `lpu_sxm_triangle_mem_integration -> sxm_triangle_writetap_core ->
sxm_triangle_full_core`. Its CModel counterparts are `SxmMatrixCore` and
`SxmMatrixMemReference`.

This implementation is not yet connected to the formal instruction path. It is also not
instantiated by the formal RTL FU cluster, while `SxmMatrixCore/SxmMatrixMemReference` is not
instantiated by `TspSliceSystem`. Consequently the older `SxmSlice`, 96-bit native
`sxm_slice`, and compatibility cores must not be used as substitutes to claim current SXM
production alignment. Section 17 remains historical evidence for the prior CModel-only
`SxmSlice` checkpoint, not an acceptance result for the new triangle core.

S3 `SXM_SR_CYCLE_PASS`, S4 `MEM_SR_SXM_CYCLE_PASS`, and S5
`TSP_MEM_SR_SXM_INTEGRATION_PASS` are therefore `BLOCKED/NOT_RUN` until the new triangle SXM
is connected to the real instruction and system production paths. No standalone triangle
component test is promoted to a joint-cycle PASS.

Final evidence is retained under `out/mem_sr_sxm_cycle_current/s1` and `s2`. Build products
and VM VCS intermediates were removed after preserving the CModel/RTL traces and scoreboard
summaries.
