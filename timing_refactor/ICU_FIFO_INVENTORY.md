# FTLPU ICU ProgramLoad / Macro Frontend FIFO 与 Buffer 清单

## 1. 口径

本清单对应以下本地快照：

- VMODEL：`D:\LPU\FTLPU-VMODEL-main` 当前脏工作树，分支 `rtl-handshake-naming`，HEAD `2cbaa104090c6df099894559794c0f981d4aafda`。ProgramLoad 新文件多为未跟踪文件，因此以下参数来自当前磁盘源码，不代表干净提交。
- CModel：`D:\LPU\FTLPU-CMODEL-clock` 当前 ICU A.1/A.2 工作区；有效生成配置来自 `out/icu_rtl_alignment/generated/include/ftlpu/core/hardware_config.hpp`。
- 本轮已执行 CModel 定向与全量测试，并以本地 VMODEL 做 Relay 功能自检和 Instruction DMA 编译核对；本地 VMODEL 没有独立 Instruction DMA 功能 TB，因此对应跨模型功能 Golden 仍标记为 PARTIAL。

分类：

- **硬件 FIFO**：有固定 entry 数、握手和读写状态。
- **硬件 Buffer/RAM**：有固定物理容量，但不一定是 FIFO。
- **Pending Event Queue**：C++ 中为模拟延迟保存的在途事件，不自动等价于硬件 FIFO。
- **软件容器**：保存程序元数据或统计，不能纳入硬件深度/PPA 扫描。

优先级：P0 直接决定全局 ProgramLoad/取指吞吐和反压；P1 决定局部 decode/issue 吞吐；P2 是控制或错误隔离缓冲；P3 是软件容器或非硬件状态。

### 1.1 2026-11-15 对齐实施更新

本节是当前实现状态，优先于后文保留的 Checkpoint 7B 调查快照。后文
第 2～4 节的 RTL 清单仍有效；第 5～10 节中将 CModel 描述为
`active context=32/40`、`fetch latency=1`、缺少显式 word/relay/context/timed
FIFO 的内容已经过时。

当前 CModel 对 MEM/MXM 分布式 ProgramLoad/Macro 共同路径增加或修正了以下
有限硬件边界：

MEM Relay 的物理分组已按 RTL 修正为 `4 slices/group × 2 banks/slice ×
2 groups/relay = 16 banks/relay`；每半区 104 个 Queue 因而映射到 7 级
Relay，最后一级只覆盖 96–103。`mem_groups_per_relay` 已成为独立生成配置，
不再由 ProgramLoader 内部魔数推断。

| CModel 当前结构 | 容量/时序 | RTL 对应 | 当前状态 |
|---|---:|---|---|
| Instruction DMA outstanding ID | 16 × 8-bit ID | `lpu_instruction_dma/u_outstanding_id_fifo` | Direct/Partial：ordered ID、last-beat release、sticky mismatch fault 已建模 |
| Instruction DMA RX FIFO | 16 × 256-bit payload + valid-bytes/ID/last/error metadata | `lpu_instruction_dma/u_response_fifo` 16×298b | Direct/Partial：深度、full replacement、下游反压、error 已建模；C++ entry 非 packed bit-vector |
| Instruction word/control assembler | 32/64/96/128-bit record；有界 35/39/43/47B reservoir | RTL DMA RX + 64B reassembler 的分层对应 | Partial：ProgramLoader 高层仍串行 image |
| Program reassembly | 64 B | `lpu_program_load_reassembler` 64 B | Partial：容量对应，CModel DMA/descriptor 协议仍不同 |
| Program word FIFO | 8 entries | `lpu_program_load_word_fifo` 8×164b | Partial：显式有限 FIFO 和反压已建模，C++ entry 为语义对象 |
| Distribution ingress FIFO | 2 entries | `lpu_program_distribute_elastic_stage` 2×165b | Partial/Direct：深度和逐拍推进对应 |
| East/West relay buffers | 2 hemispheres × 7 stages × 2 entries | 两条七级 relay chain，每级 2×165b | Partial：registered-ready、每拍最多一 hop、MEM 近端/MXM 尾端延迟已建模；尚无 BOTH、跨源仲裁 |
| MEM/MXM local iMEM | MEM 131072×96b；MXM 65536×128b | 同容量 RTL local iMEM | Partial：存储容量对应，C++ 仍使用稀疏/语义 backing |
| iMEM fetch + response FIFO | fetch latency 2；4 entries | 两级 fetch + response FIFO depth 4 | Direct/Partial：credit、无空旁路和有界 response 已建模 |
| Macro word FIFO | 4 entries | MEM/MXM word FIFO depth 4 | Direct/Partial |
| Macro parser reservoir | 3 words | MEM 3×96b；MXM 3×128b | Direct/Partial |
| decoded context FIFO | 4 entries | MEM/MXM context FIFO depth 4 | Direct/Partial：只在共同 MEM/MXM Macro 路径启用 |
| active contexts | MEM 4；MXM 每 stream 4 | RTL active contexts 4 | Direct/Partial |
| timed issue FIFO | 16 entries；目标 issue 提前 2 cycles 准备 | MEM/MXM timed queue depth 16 | Partial：深度、无空旁路、due-cycle dequeue 已建模；payload 为 C++ 语义对象 |
| Native IQ | 16 entries | legacy ICU queue depth 16，不是 distributed Macro 同名边界 | CModel-only/legacy compatibility |
| Macro DDB | 8 packed descriptors | RTL Macro 字典 7 entries，但无同构 DDB FIFO | CModel-only 内部展开边界，不能与 RTL dictionary 混同 |

配置集中在 `config/ftlpu-lpu32.json:4-39,256-270`，经
`cmake/FTLPUHardwareConfig.cmake:150-173` 和
`cmake/hardware_config.hpp.in` 生成。实现和统计证据：

- Program word/ingress/relay FIFO 与 occupancy/full/read/write 统计：
  `include/ftlpu/system/program_loader.hpp:47-53,405-418,568-651`。
- 两级 fetch、response FIFO、word FIFO 与各自统计：
  `include/ftlpu/icu/macro_bitstream_decoder.hpp:72-81,922-1054`。
- context FIFO 与 timed issue FIFO 的容量、admission、due-cycle issue 和统计：
  `include/ftlpu/icu/distributed_queue.hpp:530-532,1972-2050,2730-2792`。
- 通用 RTL FIFO Current/Next 模型：
  `include/ftlpu/icu/rtl_fifo.hpp`。

尚未一一复刻的 RTL ProgramLoad 边界：

- 32-entry command RAM 与 LOAD/PRIME/WAIT/JUMP/END 控制机。
- 独立 config FIFO 的多 image 并发配对；Instruction DMA outstanding/RX 已接入。
- BOTH 分叉的独立完成位、VXM direct 分支、endpoint round-robin 与 commit-valid。
- 独立 32-entry PRIME FIFO。
- VXM/SXM distributed Program endpoint：当前 RTL endpoint `ready=0`，standalone
  Macro frontend 未接入本次共同 Golden。

因此当前模型可以统计已对齐 MEM/MXM 路径的 occupancy、full cycles、accepted
reads/writes 和 backpressure，但不能用这些数据推断尚未建模的 BOTH/仲裁/PRIME
结构的最优深度。

### 1.2 已执行验证

- CModel 定向测试：FIFO、Macro schedule、raw Macro decoder、ProgramLoader、
  ICU clocked、TSP timing integration 全部 PASS。
- CModel 全量 Release：79 项中 72 PASS；7 个失败与 Checkpoint 4～7B 的冻结
  baseline 完全同类（4 个 MXM accumulator 容量、1 个 MEM SRAM 容量、2 个
  VXM active-bundle），没有新增 ICU 失败。
- RTL/CModel FIFO 共同激励：12 cycles、17 fields 逐字段一致，覆盖 empty
  no-bypass、full replacement、reject、simultaneous R/W、pointer wrap 和 drain。
- 本地 RTL 自检：legacy ICU NOP/Repeat、Program config FIFO、reassembler、
  elastic stage、7-stage relay chain、MEM Macro frontend、MXM 三类 Macro
  frontend 全部 PASS。
- 完整 distributed cluster 的 ModelSim 编译被 10.1c 对较新 SystemVerilog
  语法的支持限制阻塞，因此完整 ProgramLoad→FU cluster 逐周期 Golden 仍为
  PARTIAL，不能标记为 PASS。

## 2. VMODEL 通用 FIFO 语义

ProgramLoad 多个 FIFO 复用 `lpu_queue_fifo`。其关键契约是：

- `dequeue_valid = level != 0`，读数据组合地取 `data_mem[read_ptr]`，因此是 fall-through read，但**空 FIFO 不做 enqueue-to-dequeue bypass**。
- `enqueue_ready = level < DEPTH || dequeue_fire`，因此满 FIFO 若同拍成功 dequeue，可以 replacement enqueue，level 不变。
- 数据只在正沿、`enqueue_fire` 时写；指针和 level 在异步低有效 reset 时清零，data memory 不清。
- 同拍成功读写时两指针都前进，level 保持。

证据：VMODEL `rtl/queue/lpu_queue_fifo.sv:21-24,37-50,52-78`。

Relay buffer 是例外：`ready` 只看注册 level `< BUFFER_DEPTH`，满时即使同拍 dequeue 也不接受 replacement，用于切断跨 relay 的组合 ready 链。证据：`rtl/program_load/lpu_program_distribute_relay.sv:149-192,215-250`。

## 3. VMODEL ProgramLoad FIFO / Buffer 完整清单

| 名称 / 典型实例路径 | 类型 | Depth × Width | Producer → Consumer | 写/读时序、满空与反压 | Reset / 同拍读写 | 优先级 | 源码证据 |
|---|---|---:|---|---|---|---|---|
| Controller command RAM `u_program_load_system/u_controller/command_mem_q` | RAM，不是 FIFO | 32 × 155b（默认 command addr=5） | Host command load → controller PC/fetch | Host 仅在不 busy 时 ready；active record 在正沿从 RAM 读出，属于 registered fetch | controller reset 清 PC/active 状态；RAM 数据不作为 FIFO 清空 | P2 | `rtl/program_load/lpu_program_load_controller.sv:1-3,55-59,126-129,185-239` |
| Load config FIFO `u_program_load_path/u_config_fifo/u_fifo` | 硬件 FIFO | 16 × 155b | controller transaction → reassembler descriptor | 与 DMA command 原子接受；头记录保持到对应 image 最后一个 response 被接受 | 继承 `lpu_queue_fifo`：异步清 ptr/level；full+dequeue replacement | P0 | `rtl/program_load/lpu_program_load_config_fifo.sv:1-3,37-85`；`lpu_program_load_path.sv:104-141` |
| DMA outstanding-ID FIFO `u_program_load_path/u_instruction_dma/u_outstanding_id_fifo` | 硬件 FIFO | 16 × 8b | issued DDR read ID → ordered response checker | 发请求前预留 credit；ID 不匹配产生 sticky fault | 继承通用 FIFO | P0 | `rtl/program_load/lpu_instruction_dma.sv:1-6,75-81,102-125` |
| DMA RX FIFO `u_program_load_path/u_instruction_dma/u_response_fifo` | 硬件 FIFO | 16 × 298b | DDR response `{data256,keep32,id8,error2}` → reassembler | RX FIFO ready 直接控制 DDR response ready；空时无输出，满时向 DDR 反压 | 继承通用 FIFO；full+dequeue 可 replacement | P0 | `rtl/program_load/lpu_instruction_dma.sv:50,90-92,127-158` |
| Reassembly reservoir `u_program_load_path/u_reassembler/buffer_q` | 固定 Buffer | 64B / 512b | DMA RX + config → 96/128b Program Word | 只有至少 12B/16B 才 output valid；下游 stall 保持；用 post-pop 空间决定 response ready | reset 清 count/上下文；支持同拍 output pop + response append | P0 | `rtl/program_load/lpu_program_load_reassembler.sv:46-68,132-170,173-239,263-342` |
| Program Word FIFO `u_program_load_path/u_word_fifo/u_fifo` | 硬件 FIFO | 8 × 164b | reassembler → distribution | 记录为 data128+strb4+format2+destination12+address17+last1；full 回压 reassembler | 继承通用 FIFO | P0 | `rtl/program_load/lpu_program_load_word_fifo.sv:1-3,29-67` |
| PRIME FIFO `u_program_load_system/u_prime_fifo` | 硬件 FIFO | 32 × 14b | controller PRIME `{scope2,destination12}` → distribution | 仅在 load path 和 distribution idle 后允许 enqueue；所有 WRITE 退役后才优先送 PRIME | 继承通用 FIFO | P2 | `rtl/program_load/lpu_program_load_system.sv:367-403` |
| Distribution ingress elastic `u_program_distribute_system/u_distributor/u_input_stage/u_fifo` | 硬件 FIFO | 2 × 165b | load system WRITE/PRIME → scope distributor | 保存 control1+Program record164；吸收一拍下游 backpressure | 继承通用 FIFO | P0 | `rtl/program_load/lpu_program_distribute_elastic_stage.sv:1-3,30-69`；`lpu_program_distributor.sv:67-92` |
| West relay buffers `u_west_relay_chain/g_relay[0..6]/u_relay/record_mem` | 硬件 FIFO/relay | 7 ×（2 × 165b） | distributor West/BOTH branch → local MEM 或 next relay/outer | 每级只看 registered occupancy；head 按本地/next 路由，阻塞保留；满时不允许同拍 replacement | async reset ptr/level；非满时可同拍 enqueue/dequeue，level不变；memory不清 | P0 | `rtl/program_load/lpu_program_distribute_relay_chain.sv:1-15,65-196`；`lpu_program_distribute_relay.sv:149-250` |
| East relay buffers `u_east_relay_chain/g_relay[0..6]/u_relay/record_mem` | 硬件 FIFO/relay | 7 ×（2 × 165b） | distributor East/BOTH branch → local MEM 或 next relay/outer | 与 West 相同 | 与 West 相同 | P0 | 同上 |
| BOTH branch done bits | 寄存状态，不是 FIFO | 2 × 1b / active record | distributor head → East/West branches | 分别记录两个 branch 已接受；只有两边都完成才能弹 head；VXM direct 不复制 | reset 清状态；按各 branch fire 更新 | P0（控制） | `rtl/program_load/lpu_program_distributor.sv:94-118,140-160` |
| Endpoint commit-valid | Pipeline 状态，不是 FIFO | 每 endpoint 1b | FU endpoint accept → distribution idle/ack | 接受后保持一拍，避免同一接受边沿被误判 idle | reset 清；正沿置位/清除 | P2 | `rtl/program_load/lpu_program_distribute_system.sv:550-566` |
| Per-endpoint RR pointers | 仲裁状态，不是 FIFO | 每 FU 类/半区若干 bits | relay/outer candidates → FU endpoint | 仅在实际 endpoint accept 时推进，保证竞争 source 的顺序/公平性 | reset 初始化 | P1 | `rtl/program_load/lpu_program_distribute_system.sv:349-388,568-596` |

ProgramLoad 总体可见存储不应简单相加成“iMEM 容量”：command RAM、descriptor/RX/word/relay buffers 是传输解耦；真正程序容量在 FU-local iMEM。

## 4. VMODEL FU-local iMEM / Macro Frontend 清单

### 4.1 MEM（每个 bank 一套；全芯片 208 套）

实例规模来自 `rtl/core/lpu_pkg.sv:12-18` 和 `rtl/mem/lpu_mem_subsystem.sv:56,130-184`。

| 名称 / 典型层次 | 类型 | 每实例 Depth × Width | 实例数 | Producer → Consumer | 读写/反压规则 | Reset / 同拍行为 | 优先级 | 证据 |
|---|---|---:|---:|---|---|---|---|---|
| MEM local iMEM `g_queue/u_local_frontend/u_local_imem/instruction_mem` | 单端口程序 RAM | 131072 × 96b | 208 | Program endpoint → iMEM fetch | load mode 才写；fetch mode 两级请求/返回流水；Program reset 不清数据，由 descriptor 限制可见范围 | SRAM 数据不清；模式隔离写/读 | P0 | `rtl/mem/lpu_mem_local_imem_frontend.sv:93-138`；`lpu_mem_local_imem.sv:35,55-73,91-117` |
| MEM iMEM response FIFO | 硬件 FIFO | 4 × 97b | 208 | two-stage SRAM return `{last,word96}` → Macro frontend | 深度覆盖在途两级响应；满时停止 fetch request | 通用 FIFO语义 | P0 | `rtl/mem/lpu_mem_local_imem.sv:1-5,55-57,76-117` |
| MEM decoder word FIFO | 硬件 FIFO | 4 × 97b | 208 | local iMEM response → Macro decoder | downstream decoder不收时保持；满时反压 response FIFO | 通用 FIFO语义 | P1 | `rtl/mem/lpu_mem_macro_frontend.sv:1-5,83-119` |
| MEM decoder reservoir | 固定 parser buffer | 3 × 96b = 288b | 208 | word FIFO → parser | 只有有容量才收 word；解析跨 word bitstream | reset 清 parser state | P1 | `rtl/mem/lpu_mem_macro_decoder.sv:30,173-174` |
| MEM dictionary | 固定状态表 | 7 entries，entry为 FU native/delta context | 208 | decoder control word → payload decode | parser 顺序访问；非 FIFO | reset/prime 按 decoder FSM | P1 | `rtl/mem/lpu_mem_macro_decoder.sv:94-137` |
| MEM context FIFO | 硬件 FIFO | 4 × 271b | 208 | Macro decoder → scheduler | context_valid 在 downstream ready 前保持；满时停 decoder | 通用 FIFO语义 | P1 | `rtl/mem/lpu_mem_macro_frontend.sv:33,130-143` |
| MEM scheduler ingress | 单项寄存状态 | 1 × context（非独立 FIFO） | 208 | context FIFO → active allocator | ingress/active credit不足时保持 | reset 清 valid | P1 | `rtl/mem/lpu_mem_macro_scheduler.sv:65-88` |
| MEM active contexts | 固定状态阵列 | 4 entries，宽度为分布式循环/时序状态 | 208 | scheduler ingress → due selector | 多 context 按计划周期竞争 issue；不是 FIFO | reset 清 valid/state | P1 | `rtl/mem/lpu_mem_macro_scheduler.sv:78-88,299-340` |
| MEM timed bank queue | 硬件 FIFO | 16 × 96b（cycle32+instruction64） | 208 | Macro scheduler → MEM bank | 无 empty bypass；仅 exact scheduled cycle 且 FU ready 才 dequeue；late/deadline sticky fault | 通用 FIFO存储；fault reset清 | P1 | `rtl/mem/lpu_mem_bank_queue.sv:19,36-65` |

注意：`lpu_mem_macro_scheduler` 的 `ISSUE_FIFO_DEPTH` 只是接口参数，`issue_fifo_level_o='0`，不构成另一个实体 FIFO；真实末级 FIFO 是 subsystem 中的 `lpu_mem_bank_queue`。证据：`rtl/mem/lpu_mem_macro_scheduler.sv:188`、`lpu_mem_subsystem.sv:166-184`。

### 4.2 MXM（4 个 MXM，每个三条 stream，共 12 套）

| 名称 | 类型 | 每实例 Depth × Width | 实例数 | Producer → Consumer | 规则 | 优先级 | 证据 |
|---|---|---:|---:|---|---|---|---|
| MXM local iMEM | 单端口程序 RAM | 65536 × 128b | 12 | Program endpoint → per-stream fetch | 与 MEM 相同的 load/fetch 模式；两级同步读 | P0 | `rtl/mxm/lpu_mxm_local_imem_frontend.sv:1-9,178-236`；`lpu_mxm_local_imem.sv:35-44,55-88,91-117` |
| MXM iMEM response FIFO | 硬件 FIFO | 4 × 129b | 12 | SRAM return → stream frontend | 满时停止 fetch | P0 | `rtl/mxm/lpu_mxm_local_imem.sv:35-44,55-88,98-117` |
| MXM word FIFO | 硬件 FIFO | 4 × 129b | 12 | local iMEM → Macro decoder core | 满时反压 iMEM response | P1 | `rtl/mxm/lpu_mxm_load_macro_frontend.sv:3-5,85-87`；`lpu_mxm_dequant_macro_frontend.sv:3-5,83-85`；`lpu_mxm_compute_macro_frontend.sv:4-6,86-88` |
| Load context FIFO | 硬件 FIFO | 4 × 257b | 4 | load decoder → load scheduler | valid/ready hold | P1 | `rtl/mxm/lpu_mxm_load_macro_frontend.sv:33,138-155` |
| Dequant context FIFO | 硬件 FIFO | 4 × 242b | 4 | dequant decoder → dequant scheduler | 同上 | P1 | `rtl/mxm/lpu_mxm_dequant_macro_frontend.sv:33,135-152` |
| Compute context FIFO | 硬件 FIFO | 4 × 288b | 4 | compute decoder → compute scheduler | 同上 | P1 | `rtl/mxm/lpu_mxm_compute_macro_frontend.sv:34,141-159` |
| MXM decoder reservoir | 固定 parser buffer | 3 × 128b = 384b | 12 | word FIFO → common core parser | 跨 word 解析，dictionary 7 entries | P1 | `rtl/mxm/lpu_mxm_macro_decoder_core.sv:42,191-192` |
| MXM active contexts | 固定状态阵列 | 4 entries/stream | 12 | scheduler ingress → issue selector | per-stream独立 | P1 | `rtl/mxm/lpu_mxm_subsystem.sv:1-8,187-236` 及各 scheduler |
| MXM timed queue | 硬件 FIFO | 16 × 81b（cycle32+normalized issue49） | 12 | per-stream scheduler → MXM local coordinator | 三流独立缓冲；coordinator再同步/配对 load/dequant | P1 | `rtl/mxm/lpu_mxm_subsystem.sv:238-290` |

### 4.3 VXM standalone（当前未接入 ProgramLoad）

| 名称 | 类型 | Depth × Width | Producer → Consumer | 集成状态 / 反压 | 优先级 | 证据 |
|---|---|---:|---|---|---|---|
| VXM word FIFO | 硬件 FIFO | 4 × 97b | external word input → decoder | standalone frontend内有效；subsystem Program ready固定0 | P1（接入后） | `rtl/vxm/lpu_vxm_macro_frontend.sv:1-88,107-119`；`lpu_vxm_subsystem.sv:5-7,30-31` |
| VXM context FIFO | 硬件 FIFO | 2 × 288b | decoder → scheduler | 满时反压 decoder | P1（接入后） | `lpu_vxm_macro_frontend.sv:38,144-165` |
| VXM timed issue FIFO | 硬件 FIFO | 4 × 128b（cycle32+packet96） | scheduler → frontend issue | downstream ready前保持 | P1（接入后） | `lpu_vxm_macro_frontend.sv:196-212`；`lpu_vxm_timed_issue_queue.sv:24,65-67` |

实际 VXM subsystem 的执行接口是 8 个 local compact queues（local instruction 最大 7 bit）和 15-bit global config；standalone 96-bit issue packet 是适配层格式，不能当成最终 local queue 宽度。证据：`rtl/core/lpu_pkg.sv:39-43`。

### 4.4 SXM standalone（当前未接入 ProgramLoad）

| 名称 | 类型 | Depth × Width | Producer → Consumer | 集成状态 / 反压 | 优先级 | 证据 |
|---|---|---:|---|---|---|---|
| Transpose word FIFO | 硬件 FIFO | 4 × 97b | external 96b word → decoder | standalone；subsystem Program ready=0 | P1（接入后） | `rtl/sxm/lpu_sxm_transpose_macro_frontend.sv:2-4,95-97`；`lpu_sxm_subsystem.sv:5-7,32-33` |
| Transpose context FIFO | 硬件 FIFO | 2 × 608b | decoder → scheduler | 单 active scheduler；满时停 decoder | P1（接入后） | `lpu_sxm_transpose_macro_frontend.sv:29,138-151` |
| Transpose timed issue FIFO | 硬件 FIFO | 4 × 448b（cycle32+issue416） | scheduler → SXM issue | downstream ready前保持 | P1（接入后） | `lpu_sxm_transpose_macro_frontend.sv:180-198` |
| Permute word FIFO | 硬件 FIFO | 4 × 97b | external 96b word → decoder | standalone | P1（接入后） | `rtl/sxm/lpu_sxm_permute_macro_frontend.sv:2-4,97-99` |
| Permute context FIFO | 硬件 FIFO | 2 × 613b | decoder → scheduler | 单 active scheduler | P1（接入后） | `lpu_sxm_permute_macro_frontend.sv:29,141-154` |
| Permute timed issue FIFO | 硬件 FIFO | 4 × 448b | scheduler → SXM issue | downstream ready前保持 | P1（接入后） | `lpu_sxm_permute_macro_frontend.sv:184-202` |

## 5. CModel Buffer / Queue 清单

C++ 对象通常没有可综合的“packed width”。下表中的 width 若写为“语义对象”，表示只能给出有效 payload/ABI 宽度，不能把 `sizeof(C++ type)` 当作硬件位宽。

### 5.1 ProgramLoader 与 DMA

| 结构 / 成员 | 类别 | 容量 / 逻辑宽度 | Producer → Consumer | 满空、延迟、反压 | 是否可能无限增长 | RTL 对应 | 优先级 | 证据 |
|---|---|---|---|---|---|---|---|---|
| `images_ : vector<ProgramImage>` | 软件容器 | 本次 program image 数；元数据对象 | `submit()` → `begin_queue/tick` | 无硬件 full/ready；image严格串行 | 可随输入增长 | controller RAM/config descriptors 的高层合并 | P3 | `include/ftlpu/system/program_loader.hpp:153-228,395` |
| `InstructionFetchDma` command/outstanding | 有限硬件行为模型 | 16 commands × 8-bit ID | ProgramLoader command → shared DDR requests | ordered ID、last response release、ID exhaustion、sticky fault；每拍至多接受一个 command | 有界 | RTL outstanding-ID FIFO | P0 | `include/ftlpu/icu/instruction_fetch_dma.hpp`；`program_loader.hpp` |
| `InstructionFetchDma` response FIFO | 有限硬件行为模型 | 16 entries × 256-bit payload + metadata | shared DDR completion → word/control assembler | full+dequeue replacement；full/stall 回压 DDR completion；记录 occupancy/full/read | 有界 | RTL 16×298b RX FIFO | P0 | 同上；`rtl/program_load/lpu_instruction_dma.sv:83-140` |
| `route_ : vector<optional<Packet>>` | DMA 到 reassembler 的有限弹性 pipeline | `route_latency_cycles` slots；每 slot ≤32B/256b | Instruction DMA assembled word → reassembly | 每 tick 从尾到头移动；尾端无64B空间则整条上游slot保持 | 固定为配置深度 | RTL DMA/reassembler 接口前的 CModel bus-latency adapter，不再代表 relay 网络 | P0 | `program_loader.hpp:441-472,709`；config `ftlpu-lpu32.json:259` |
| `reassembly_ : deque<uint8_t>` | 有容量约束的软件 Buffer | 最大 64 B | route tail → word assembler | 只有空间容纳packet才pop route；达到12/16B即可取word | 代码显式限制64B，不会无限增长 | RTL 64B reassembler，Partial | P0 | `program_loader.hpp:93,285-307,397` |
| `word_fifo_ : deque<PendingWrite>` | 有限硬件行为 FIFO | 8 entries；语义 Program Word | reassembler → distribution ingress | 满时回压 reassembler；读写计数、peak/full cycles 可观测 | 有界 | RTL Program Word FIFO，Direct/Partial | P0 | `program_loader.hpp:412-438,624-632,667-672`；config `ftlpu-lpu32.json:265` |
| `ingress_fifo_ : deque<PendingWrite>` | 有限硬件行为 FIFO | 2 entries；语义 Program record | Program Word FIFO → relay R0 | 满时回压 word FIFO；每拍至多向 R0 推进一项 | 有界 | RTL distribution ingress elastic，Direct/Partial | P0 | `program_loader.hpp:624-632,661-672`；config `ftlpu-lpu32.json:266` |
| `relay_fifos_` | 有限 registered-ready relay buffers | 2 hemispheres × 7 relays × 2 entries | ingress → local MEM/MXM endpoint | 每拍至多一 hop；ready 取 LOW 开始时 occupancy；满时禁止同拍 replacement | 有界 | RTL East/West relay chains，Direct/Partial | P0 | `program_loader.hpp:587-674,714-716`；config `ftlpu-lpu32.json:267-269` |
| `commits_ : deque<PendingWrite>` | 有界 Pending Event Queue | admission 受 commit latency guard；每项含64B storage+address+due | endpoint accept → `write_program_word()` | due cycle前保留；不是 Program Word FIFO；只建模 endpoint commit latency | 实际 admission 有界 | RTL endpoint commit/ack 的时间抽象，Different | P0 | `program_loader.hpp:386-405,633-645,711` |
| Program loader state/current index | 控制状态 | 单 active image | images_ → Instruction DMA/commit | 下一个 image 必须等当前全部 word commit | 固定 | controller PC/active descriptor 的软件抽象 | P2 | `program_loader.hpp:354-405,676-690` |
| Instruction assembler | 有限行为模型，已集成 | 32/64/96/128-bit records；256-bit response crossing | RX FIFO → ProgramLoader route/reassembly | 下游 stall 时保持完整 word；容量有界并统计 peak | 有界 | RTL RX FIFO + reassembler 的语义分层 | P0 | `include/ftlpu/icu/instruction_fetch_dma.hpp` |

### 5.2 `DistributedIcuQueue` 每 endpoint 状态

| 结构 | 类别 | 当前容量/宽度 | Producer → Consumer | 行为 | RTL 对应 | 优先级 | 证据 |
|---|---|---|---|---|---|---|---|
| Native local iMEM | 固定容量软件 RAM模型 | MEM 131072×96；MXM 65536×128；VXM/SXM 2048×96 | ProgramLoader direct write → fetch | `vector<optional<word>>`；reset清可见program状态 | FU local iMEM，容量Direct，读时序Partial | P0 | `include/ftlpu/system/icu.hpp:137-216,425-467`；CP7B config `hardware_config.hpp:24-39` |
| Raw Macro iMEM | 固定容量软件 RAM模型 | 同各FU word/depth | ProgramLoader → Macro decoder fetch | 与 Native 分开保存/配置 | RTL同一local iMEM模式的抽象，Different | P0 | `distributed_queue.hpp:526-570,2758-2819` |
| Native IQ `iq_ : deque<Entry>` | 有限行为 FIFO | 16 entries；Entry为语义指令对象 | matured fetch → queue-control/issue | free credit才fetch；commit response后dispatch/issue；每tick最多发起1 fetch | RTL当前Macro frontend没有同名Direct结构 | P1 | `distributed_queue.hpp:1016-1080,1992-2050,2758-2819`；config `hardware_config.hpp:24-39` |
| Native pending fetches | Pending Event Queue | 由IQ free credit与fetch latency间接限制；word为FU ABI宽 | local iMEM request → IQ commit | CP7B latency 1；matured response先commit | RTL two-stage fetch+response FIFO，Merged/Different | P0 | `distributed_queue.hpp:1973-2050`；config `hardware_config.hpp:46` |
| Macro pending fetches | Pending Event Queue | 由reservoir reservation限制；FU word96/128 | raw iMEM → reservoir | fetch II=1、latency=1；先commit再发新request | RTL local iMEM pipeline+response FIFO，Merged/Different | P0 | `macro_bitstream_decoder.hpp:569-572,877-1003` |
| Macro reservoir | 固定 parser buffer | 3 words：MEM/SXM 3×96，MXM 3×128 | pending fetch → parser | 预留credit防在途响应溢出；跨word解析 | RTL decoder reservoir，Direct/Partial | P1 | `macro_bitstream_decoder.hpp:125-162,430-505,921-1003`；config `hardware_config.hpp:49` |
| DDB | 固定 decoded-descriptor buffer | 8 entries；语义descriptor/run对象 | parser → context admission | DDB满可作为frontend primed条件；run capacity 8 | RTL word/context/dictionary流水的合并，Different | P1 | `macro_bitstream_decoder.hpp:220-227,1839-1875`；config `hardware_config.hpp:51-53` |
| Active Macro contexts—MEM | 固定硬件状态模型 | 32 × 256b | DDB/admission → due issue | 新context在已有context issue后接纳，下一周期可见 | RTL active contexts 4，深度Different | P1 | `distributed_queue.hpp:1903-1946,2773-2779`；config `hardware_config.hpp:27,44` |
| Active Macro contexts—MXM | 固定硬件状态模型 | 40 × 256b | 同上 | 同上 | RTL per-stream active contexts 4 | P1 | config `hardware_config.hpp:31,45` |
| Active Macro contexts—SXM Transpose | 固定硬件状态模型 | 4 × 256b | Macro decoder → SXM issue | CModel集成，RTL当前standalone未接ProgramLoad | RTL standalone single-active，Different | P1 | config `hardware_config.hpp:39`；`include/ftlpu/icu/sxm_macro_instruction.hpp:117-132`；`hardware_params.hpp:192-208` |
| Active Macro contexts—SXM Permute | 固定硬件状态模型 | 4 × 384b（当前32-byte physical vector配置） | 同上 | 同上 | RTL standalone single-active，Different | P1 | 同上；由 `kPhysicalVectorBytes=32`、lane bits=5 代入公式得384b |
| VXM Macro context配置 | 固定参数但路径禁用 | 配置4，Macro image在validate时拒绝 | 无实际Macro producer | 不应当作已启用FIFO | RTL VXM Macro亦未集成 | P3（当前无流量） | `include/ftlpu/system/icu.hpp:386-423`；config `hardware_config.hpp:32-35` |
| Queue-control/repeat/sync state | 寄存/控制状态 | 非FIFO；每queue一套 | decoded/native control → issue gating | NOP/Repeat/Sync改变消费与发射，不是payload queue | RTL scheduler/control 的合并抽象 | P1 | `distributed_queue.hpp:2246-2325,2758-2819` |

### 5.3 CModel 全局控制存储（相关但非 Program payload FIFO）

CP7B 配置还定义 Queue Control capacity 16、cache depth 4、read latency 2、pool capacity 512、pool read II=1，以及 common barrier capacity 1024。它们是控制协议/存储，不应与 FU-local iMEM、Native IQ 或 Macro DDB 淹没合并。证据：`out/checkpoint_7b_dma_ddr/generated/include/ftlpu/core/hardware_config.hpp:11-22`。当前 VMODEL ProgramLoad 调查路径以 WRITE/PRIME 为主，没有与这些 CModel QCTL/BARR partition 一一对应的完整结构，因此映射为 **Different/Unknown**，后续需单独做控制协议调查。

## 6. RTL ↔ CModel 对应表

| RTL FIFO / Buffer | RTL Depth × Width | CModel 结构 | CModel容量 | 映射 | 主要差异 | 优先级 |
|---|---:|---|---:|---|---|---|
| Controller command RAM | 32×155 | `images_` + state/current | image数，无硬件上限 | Merged/Different | 无RTL command opcode/PC/host backpressure周期 | P2 |
| Config FIFO | 16×155 | `images_[current_]` | 1 active | Missing/Merged | CModel串行image避免配对问题，无法研究FIFO occupancy | P0 |
| Outstanding ID FIFO | 16×8 | `InstructionFetchDma::outstanding_ids` | 16 commands | Direct/Partial | 真实路径；ProgramLoader 当前串行 image，独立测试覆盖多 outstanding | P0 |
| RX FIFO | 16×298 | `InstructionFetchDma::responses` | 16 semantic entries | Direct/Partial | data/ID/last/error 与反压对应；非 packed C++ bit-vector | P0 |
| Reassembler | 64B | `reassembly_` | 64B | Partial | 容量相同；同拍和错误状态边界不同 | P0 |
| Program Word FIFO | 8×164 | `word_fifo_` | 8 | Direct/Partial | C++ entry为语义对象；已统计 occupancy/full/read/write | P0 |
| Prime FIFO | 32×14 | state=`Prime` + per-queue configure | 无队列 | Merged | 不表现多Prime记录occupancy | P2 |
| Ingress elastic | 2×165 | `ingress_fifo_` | 2 | Direct/Partial | C++ entry为语义对象；无 scope/BOTH metadata | P0 |
| Relay buffers | 14×(2×165) | `relay_fifos_` | 2×7×2 | Direct/Partial | 具备拓扑、registered-ready和近端停靠；缺 BOTH/RR/commit-valid | P0 |
| BOTH done state | 2 bits/record | 无 | 0 | Missing | CModel destination已唯一化 | P0 |
| MEM local iMEM | 208×(131072×96) | 208×(131072×96) | 相同 | Direct容量 / Partial实现 | 两级 fetch 已对齐；存储 backing 仍为 C++ 语义模型 | P0 |
| MEM response FIFO | 208×(4×97) | 独立 response FIFO | 4 | Direct/Partial | C++ entry为语义对象 | P0 |
| MEM word FIFO | 208×(4×97) | 独立 Macro word FIFO | 4 | Direct/Partial | payload非 packed RTL vector | P1 |
| MEM context FIFO | 208×(4×271) | 独立 decoded context FIFO | 4 | Direct/Partial | FU-specific字段仍为语义对象 | P1 |
| MEM active contexts | 208×4 | 208×4 | 4 | Direct/Partial | allocator/selector未做完整 cluster 同迹对比 | P1 |
| MEM timed queue | 208×(16×96) | 独立 timed issue FIFO | 16 | Direct/Partial | due-cycle语义已建模，payload非 packed RTL vector | P1 |
| MXM local iMEM | 12×(65536×128) | 12×(65536×128) | 相同 | Direct容量 / Partial实现 | 两级 fetch 已对齐；存储 backing 仍为 C++ 语义模型 | P0 |
| MXM response FIFO | 12×(4×129) | 独立 response FIFO | 4 | Direct/Partial | C++ entry为语义对象 | P0 |
| MXM word/context FIFOs | per stream 4 entries | 独立 word/context FIFO | 4/4 | Direct/Partial | 保留额外 DDB8 作为 CModel parser 内部边界 | P1 |
| MXM active contexts | 12×4 | 12×4 | 4 | Direct/Partial | 尚无完整 cluster 同迹对比 | P1 |
| MXM timed queues | 12×(16×81) | 独立 timed issue FIFO | 16 | Direct/Partial | local coordinator前的精确 packed encoding仍不同 | P1 |
| VXM standalone 4/2/4 FIFOs | 未集成 | Native IQ16；Macro禁用 | 16/native | Different | 不能做Macro周期对比 | P1，待接入 |
| SXM standalone 4/2/4 FIFOs | 未集成 | Macro reservoir3+DDB8+contexts4 | 有限 | Different | CModel已集成，RTL只standalone | P1，待范围统一 |

## 7. 读写时序和反压重点

### 7.1 真正的全局 P0 链

```text
DDR response
 -> RX FIFO(16)
 -> Reassembler(64B)
 -> Program Word FIFO(8)
 -> Ingress Elastic(2)
 -> West/East Relay(7级×2entries)
 -> FU local iMEM write/PRIME
```

任一级满会逐级向上游反压；BOTH 记录需要两个分支各自完成，因而一个半区的阻塞可以阻塞整个 head record。Relay 的 registered-ready 让每级形成真正的时序边界，也意味着 full relay 不允许同拍 replacement，深度和每级停顿对吞吐有直接影响。

### 7.2 local iMEM P0 边界

RTL local iMEM 两级读后进入 4-entry response FIFO。当前 CModel 已使用两拍 fetch
和独立 4-entry response FIFO；它既吸收固定流水返回，也决定是否继续 fetch，
occupancy/full/read/write 均可观测。存储实现仍是语义 backing，不宣称 SRAM Macro 同构。

### 7.3 Macro P1 边界

word FIFO、context FIFO、active contexts 和 timed issue queue共同决定压缩流展开速度。
当前 CModel 的共同 MEM/MXM 路径已经对齐 word FIFO 4、context FIFO 4、
active contexts 4 和 timed issue FIFO 16；FU-specific entry 编码及完整 RTL
cluster Trace 仍是 Partial。

## 8. FIFO 深度研究顺序

### P0：先做周期等价，再扫描

1. DMA outstanding/RX FIFO（16）。
2. 64B reassembly、8-entry Program Word FIFO。
3. 2-entry ingress elastic、两半区每级 2-entry relay。
4. FU local iMEM response FIFO（4）及真实两级读延迟。
5. BOTH branch completion 与 endpoint commit acknowledgement（虽不是FIFO，却决定全局反压）。

理由：这些结构任何一个满都会回压外部指令装载或阻塞全部/半区 Program 分发。当前 CModel 已有 DMA、word、ingress、relay occupancy；BOTH/endpoint commit-valid 等未建模边界若纳入扫描，仍须先补齐。

### P1：在 P0 对齐后扫描

1. Decoder word FIFO（4）。
2. Context FIFO（MEM/MXM 4，VXM/SXM standalone 2）。
3. Active context 深度（RTL/CModel MEM/MXM 当前均为4）。
4. Timed issue queue（RTL MEM/MXM 16；VXM/SXM standalone 4）。
5. CModel DDB 8、admission lookahead 8、run capacity 8。

理由：这些主要影响单个 queue/stream 的 Macro decode 与 issue burst，不一定立即阻塞全局 ProgramLoad，但会通过 frontend ready 和 local fetch最终传播回上游。

### P2：控制与隔离容量

- Command RAM 32、Prime FIFO 32。
- Endpoint commit-valid、RR pointer、sticky fault state。
- CModel QCTL cache/pool 与 barrier storage（需另做控制协议映射后再决定是否扫描）。

### P3：不纳入硬件 FIFO 深度优化

- CModel `vector<ProgramImage>`、统计结构、日志/trace容器。
- 纯 descriptor/codec 临时对象和测试 fixture。

## 9. 结论

当前已可对 Instruction DMA outstanding/RX、Program Word、Ingress、两半区 Relay、
local iMEM response/word、context 和 timed issue FIFO 做有限容量 occupancy 与
backpressure 统计。尚未补齐的 P0/P2 边界主要是 command/config 多 image 配对、
BOTH、endpoint RR/commit-valid 和 PRIME FIFO；在把它们纳入深度结论前仍需明确
实验范围并建立完整 cluster 同激励 Trace。
