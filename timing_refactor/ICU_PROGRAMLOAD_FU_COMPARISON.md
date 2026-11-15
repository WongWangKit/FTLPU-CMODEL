# FTLPU ICU ProgramLoad、FU Macro Decoder 与 Frontend 对比调查

## 1. 调查范围、基线和证据等级

本报告只调查当前本地源码，不修改 CModel、RTL、CMake、ISA、Checkpoint 或 FIFO 参数；本轮没有编译、仿真、综合，也没有执行 VCS。结论中的“实现”表示源码静态证据，“测试覆盖”仅表示存在对应测试代码或历史日志；除非特别注明，**不表示 CModel 与 RTL 已用同一激励逐周期比较通过**。

### 1.1 本地基线

| 对象 | 当前基线 | 状态 |
|---|---|---|
| VMODEL | `D:\LPU\FTLPU-VMODEL-main`；分支 `rtl-handshake-naming`；HEAD `2cbaa104090c6df099894559794c0f981d4aafda`；origin 为个人 `FTLPU-VMODEL-DEV` | 相关 RTL 是本地脏工作树快照。`rtl/program_load` 中旧文件被删除，新的 `lpu_program_load_*`、`lpu_program_distribute_*` 文件处于未跟踪状态；MEM/MXM/VXM/SXM 集成文件也有本地修改。因此本报告对应“当前磁盘内容”，不是 HEAD 的干净内容。 |
| CModel | `D:\LPU\FTLPU-CMODEL-clock`；没有可用 Git 元数据 | 时序重构工作区，停止点为 Checkpoint 7B；Checkpoint 7C 尚未开始。证据见 `timing_refactor/TIMING_REFACTOR_CHANGELOG.md:290-315`、`timing_refactor/MIGRATION_CHECKLIST.md:149-172`。 |

为避免大量哈希干扰阅读，只冻结 10 个关键文件的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| VMODEL `rtl/program_load/lpu_program_load_system.sv` | `C21BB3B8D3573135B4E1B23F938E91E04A634B4324E7AD4B4F2C170D0FFE4AAF` |
| VMODEL `rtl/program_load/lpu_program_load_path.sv` | `E3CF60C6C7FE8DA464D707573D890B8358AFE059B91ACC481F97A9D002290EC9` |
| VMODEL `rtl/program_load/lpu_program_distribute_system.sv` | `62B7458A298591B0DD8C2916D285F9FB32608DE6A093BC56298B87DB89C177C0` |
| VMODEL `rtl/mem/lpu_mem_subsystem.sv` | `DB9D35A6FF92EC6BA534586F7010EBCB04ADD80EBD2719079C89FC68C0E1BA5C` |
| VMODEL `rtl/mxm/lpu_mxm_subsystem.sv` | `C8481393F5679E93FFB1448E41C7DF7139325C352B2BED3F8D87A549F9D18132` |
| CModel `include/ftlpu/system/program_loader.hpp` | `E5E5D70D970502AFD1F809D91273034688CFA7C4D86BDA5040E6F1987F0C3076` |
| CModel `include/ftlpu/system/icu.hpp` | `72FCB47024C907B8F52459E1E5733E9617E64355D3BD54E27B848C7F6DF0DFCD` |
| CModel `include/ftlpu/icu/distributed_queue.hpp` | `C94B7D1C4FCC53CA417AC1D87CE5451FFE0146B25E13AA596CE019BA7519EEC1` |
| CModel `include/ftlpu/icu/macro_bitstream_decoder.hpp` | `906843AB80589AC37F8D30B79C5AE1DFCCFD8F6244FC58C5B8F4C443A24BD536` |
| CModel CP7B 生成配置 `out/checkpoint_7b_dma_ddr/generated/include/ftlpu/core/hardware_config.hpp` | `CC8E38B08F5284AC34E4E4852A65BC978C75CC1DD72381D7109100360DE0DA36` |

### 1.2 结论摘要

1. **VMODEL ProgramLoad 是真实的多级硬件通路**：命令控制器、配置 FIFO、256-bit Instruction DMA、Outstanding-ID FIFO、返回 FIFO、64-byte 重组器、Program Word FIFO、全局分发器、两条半区 Relay Chain、FU-local iMEM 和 Frontend 均有明确容量与 valid/ready 反压。
2. **CModel ProgramLoader 是定时行为抽象，而不是 RTL 模块逐级复刻**：它复用 `C2cDmaEngine`，以软件路由槽、64-byte `deque` 和延迟提交队列把数据直接写入 `InstructionControlUnit` 的本地程序存储；没有 RTL 的配置 FIFO、Program Word FIFO、BOTH 扇出、Relay Chain 和 FU 端口仲裁的等价结构。
3. **RTL 当前真正贯通 ProgramLoad→local iMEM→Macro Frontend 的是 MEM 和 MXM**。分发网络可以把 SXM/VXM 记录送到边界，但 `lpu_program_load_system` 的描述符校验仅接受 MEM 和三类 MXM；VXM/SXM subsystem 又将 Program endpoint `ready` 固定为 0。独立 VXM/SXM Macro Frontend 源码存在，但未接入当前 ProgramLoad 集成链。
4. **CModel 支持范围不同**：MEM/MXM/SXM 可以装载 Macro；VXM Macro 被明确拒绝。VXM 仍有本地 Native 队列。因此 CModel 的 SXM 路径在功能覆盖上反而超过当前 RTL 集成，VXM 装载语义也与 RTL 不同。
5. **最关键的周期差异**是：RTL local iMEM 有两级请求/返回流水及 4-entry response FIFO，而 CModel CP7B 配置的 fetch latency 是 1；RTL MEM/MXM Macro scheduler 只有 4 个 active contexts，而 CModel 是 MEM 32、MXM 40；RTL 分发/反压是物理多级网络，CModel 是单个串行 image 加载器。
6. 目前只能建立功能事件映射，**不能宣称周期级等价**。后续应先同步 P0 缓冲和可观察事件，再做同激励逐周期 Golden 比较，之后才适合扫描 FIFO 深度。

## 2. VMODEL ProgramLoad 实际架构

### 2.1 编码、目标和传输粒度

ProgramLoad 的统一传输记录使用 128-bit Program Word；96-bit 格式通过 4-bit strobe=`4'b0111` 表示，128-bit 格式使用 `4'b1111`。目标字段为 12 bit：`hemisphere[11] + kind[10:8] + index[7:0]`。目标种类包括 MEM、MXM Load/Dequant/Compute、VXM、SXM Transpose/Permute；scope 为 East、West 或 Both。证据：VMODEL `rtl/program_load/lpu_program_load_pkg.sv:2-16,18-45,47-74`。

需要区分三个概念：

- 128 bit 是 ProgramLoad 统一运输宽度，不代表所有 FU 的物理指令都为 128 bit。
- MEM/SXM/VXM Macro Word 为 96 bit，MXM Macro Word 为 128 bit；FU issue 宽度又分别是 MEM 64、MXM 48、VXM 128、SXM 416 bit。证据：`rtl/core/lpu_pkg.sv:145-153`。
- destination 是物理装载目标，不是运行时 opcode；MXM 的三种 stream 通过 target kind 区分。

### 2.2 端到端数据流

```text
Host command
  valid/ready
      |
      v
lpu_program_load_controller
  command RAM 32 x 155b；逐命令 PC/状态机
      | LOAD_IMEM transaction（DMA cmd 与 config 原子接受）
      +-------------------------------+
      |                               |
      v                               v
lpu_instruction_dma             lpu_program_load_config_fifo
  outstanding ID FIFO 16x8        16x155
  DDR read response FIFO 16x298       |
      | 256b response                  |
      +---------------+---------------+
                      v
          lpu_program_load_reassembler
             64-byte reservoir
             96/128-bit Program Word
                      |
                      v
          lpu_program_load_word_fifo
                  8x164
                      |
                      v
          lpu_program_distributor
           ingress elastic FIFO 2x165
              scope decode / BOTH split
                 /             \
                v               v
       West relay chain     East relay chain
         7 relays             7 relays
       each 2x165           each 2x165
        | MEM local           | MEM local
        | outer MXM/SXM       | outer MXM/SXM
        +----------+----------+
                   |
       per-FU round-robin / commit acknowledgement
          | MEM(2) | MXM(4) | SXM(2) | VXM(1)
          v        v        v        v
      FU subsystem Program endpoints
          |
          v
  FU-local iMEM -> response FIFO -> Macro Frontend -> timed issue queue -> FU
```

模块证据：

- `lpu_program_load_path` 将 config FIFO、Instruction DMA、reassembler 和 word FIFO 串联；默认参数为 DMA 256 bit、config/RX/outstanding 深度 16、重组 64 B、word FIFO 深度 8：`rtl/program_load/lpu_program_load_path.sv:1-11,104-141,143-245`。
- LOAD 命令只有 DMA command 与 config 两侧同时 ready 才接受，避免描述符与返回数据错配：同文件 `:104-107`。
- `lpu_program_load_system` 在写入全部退役后才把 PRIME 记录送入同一分发网络，Prime FIFO 为 `32 x 14`；完成条件是 `command_done && load_path_idle && distribution_idle && !error`：`rtl/program_load/lpu_program_load_system.sv:367-420`。
- 顶层 ProgramLoad 到 FU cluster 的集成位于 `rtl/lpu_top.sv:276-298`；分发输出接入 FU cluster 见 `rtl/core/lpu_fu_subsystem_cluster.sv:286-315`。

### 2.3 命令、DMA、重组和写入时序

1. `lpu_program_load_controller` 用参数化 command RAM 保存命令；默认深度 32，记录宽度是 `150 + command_address_width`，默认地址宽 5，因此为 155 bit。活动记录在正沿从 RAM 取出；host 仅在 controller 不 busy 时 ready。证据：`rtl/program_load/lpu_program_load_controller.sv:1-3,55-59,126-129,185-239`。
2. Instruction DMA 在发出 DDR 请求前预留 outstanding credit；ID FIFO 深度 16。DDR 返回只有在 RX FIFO 可接收时才 ready；返回记录宽度为 `256+32+8+2=298` bit，深度 16。ID 顺序错误或 DDR error 形成 sticky fault。证据：`rtl/program_load/lpu_instruction_dma.sv:1-6,50,75-81,90-110,112-158`。
3. 重组器保存 512-bit/64-byte reservoir 和跨 burst 的 image 上下文。只有完整 12 B 或 16 B word 可用时才置输出 valid；下游 stall 时输出保持。它支持同周期 pop 一个 Program Word 并 append 一个 DMA response，config 只在接受该 image 的最后一个 response 时弹出。证据：`rtl/program_load/lpu_program_load_reassembler.sv:46-68,132-170,173-239,263-342`。
4. Program Word FIFO 保存 `{data128,strb4,format2,destination12,address17,last1}`，共 164 bit、深度 8。证据：`rtl/program_load/lpu_program_load_word_fifo.sv:1-3,29-67`。

### 2.4 分发能力、顺序和反压

- 分发入口有 2-entry elastic stage，吸收一拍反压：`rtl/program_load/lpu_program_distribute_elastic_stage.sv:1-3,30-69`。
- `BOTH` 在中央 distributor 中对 East/West 分别维护 done 位，只有两个物理分支都接受才完成；VXM 是中央 East direct route，不做 BOTH 复制。证据：`rtl/program_load/lpu_program_distributor.sv:67-118,120-169`。
- 每半区 52 个 MEM slice、每组 4 个 slice、每 relay 覆盖 2 组，因此需要 7 个 relay；两半区共 14 个 relay buffer。Relay 只在本地 MEM endpoint 接受、下一级接受或 route invalid 时 dequeue。证据：`rtl/program_load/lpu_program_distribute_relay_chain.sv:1-15,30-32,65-86,107-218`。
- Relay 的 `ready` 只依赖本级注册 occupancy 和路由合法性，故切断下游组合 ready 链；满时即使头部本周期被取走，也不接受 replacement。这与通用 `lpu_queue_fifo` 的“full+dequeue 可同拍 replacement”不同。证据：`rtl/program_load/lpu_program_distribute_relay.sv:149-192,215-250`；通用规则见 `rtl/queue/lpu_queue_fifo.sv:21-24,37-50,52-78`。
- 每半区从 7 个 relay source 和 outer source 对 MEM/MXM/SXM 端点做 round-robin；只有 FU 实际接受才推进仲裁指针。一次被接受的写入还通过一拍 `commit_valid` 阻止 distribution 过早报告 idle。证据：`rtl/program_load/lpu_program_distribute_system.sv:312-408,416-545,550-609`。
- 物理输出端口是 2 条 MEM 半区通道、4 条 MXM 通道、2 条 SXM 半区通道和 1 条 VXM 通道；具体 subsystem 实例见 `rtl/core/lpu_fu_subsystem_cluster.sv:135-173,386-396,429-439,473-483,552-560`。

结论：网络可以在不同分支上并行推进，但**不是任意多个 FU 每周期无条件同时写入**。同一 endpoint 受仲裁限制，某一 relay/full endpoint 的阻塞会沿该分支回传；入口和 BOTH 记录又可能把阻塞扩散到全局上游。

### 2.5 当前目标覆盖限制

`lpu_program_load_pkg` 定义了 VXM/SXM 目标，distribution 也能把记录送到对应边界；但当前 integrated loader 的描述符合法性只接受 MEM 和三类 MXM，其他 kind 进入 invalid 分支。证据：`rtl/program_load/lpu_program_load_system.sv:183-238`。此外：

- VXM subsystem 明确注明 Program endpoint 尚未实现，`program_ready_o=0`：`rtl/vxm/lpu_vxm_subsystem.sv:5-7,30-31`。
- SXM subsystem 同样将 Program endpoint `ready` 固定为 0：`rtl/sxm/lpu_sxm_subsystem.sv:5-7,32-33`。
- RTL 测试源码专门检查 SXM/VXM 记录能到达边界并在永久 ready-low 时保持 valid，这证明的是“边界可达与 hold”，不是“成功装入 iMEM”：`sim/tb/program_load/lpu_program_load_fu_boundary_tb.sv:141-226`。
- 顶层 Program 测试源码覆盖 MEM 与 MXM 的本地 iMEM 写入，并明确期望拒绝 VXM/SXM：`sim/tb/lpu_top_fu_subsystem_program_tb.sv:435-508,546-576,608-657`。

因此当前 VMODEL 的集成状态应写成：**MEM/MXM 已贯通；VXM/SXM 只有 distribution 边界，未贯通 local iMEM/Frontend。**

## 3. VMODEL FU-local iMEM、Macro Decoder 和 Frontend

### 3.1 FU 结构总表

| FU | 实际 Queue / iMEM 组织 | Macro Decoder / Frontend | 当前 ProgramLoad 集成 | 关键特征 |
|---|---|---|---|---|
| MEM | 2 hemispheres × 104 banks = 208 个 local frontend/iMEM；每个 iMEM `131072 x 96` | 集成 | 是 | 96-bit word；3-word/288-bit reservoir；7-entry dictionary；4-entry word FIFO、4-entry context FIFO、4 active contexts；输出到 16-entry timed bank queue。 |
| MXM | 4 个 MXM × Load/Dequant/Compute 三条独立 stream = 12 个 local frontend/iMEM；每个 `65536 x 128` | 三套专用 frontend，共用 decoder core 结构 | 是 | 三流独立装载、独立 Macro context 和独立 timed queue；issue 宽分别归一到 49 bit 进入 coordinator。 |
| VXM | 执行侧为 8 个 local compact queue + 1 个 global config；Program local iMEM 未接入 | 独立 standalone Macro Frontend 存在 | 否 | standalone frontend 以 96-bit word 产生 96-bit adapter issue packet；不能等同实际 7-bit local instruction/15-bit global config。 |
| SXM | 执行侧每半区 Transpose/Permute 两类 416-bit issue；Program local iMEM 未接入 | Transpose、Permute 各有 standalone Macro Frontend | 否 | 两类 context 编码不同，均以 96-bit descriptor 生成 416-bit issue；standalone issue queue 为 448 bit（cycle32+issue416）。 |

物理规模证据：`rtl/core/lpu_pkg.sv:12-18,28-43,145-180`。

### 3.2 MEM

MEM subsystem 默认 `IMEM_DEPTH=131072`、active contexts 4、FU timed queue depth 16；每半区实例化 104 个 local frontend，并在每个 frontend 后实例化一个 bank timed queue。Program endpoint 检查 kind/index、只接受 96-bit strobe。证据：`rtl/mem/lpu_mem_subsystem.sv:1-9,56,82-104,130-184`。

每个 MEM local frontend：

```text
Program WRITE/PRIME
  -> local_iMEM 131072x96
       -> two-stage synchronous fetch
       -> response FIFO 4x97
       -> Macro Frontend word FIFO 4x97
       -> MEM Macro Decoder (288b reservoir, dictionary 7)
       -> context FIFO 4x271
       -> scheduler ingress register + 4 active contexts
       -> subsystem timed bank queue 16x96
       -> MEM bank execution
```

Program image 在 PRIME 前必须地址连续、格式 96 bit、只有一个 last；frontend ready 还要求 fetch complete、decoder done 和 scheduler prime ready。证据：`rtl/mem/lpu_mem_local_imem_frontend.sv:1-7,93-107,115-197,199-233`。

local iMEM 是单端口风格：load mode 写入；fetch mode 经过请求级和返回级两拍，返回 FIFO 深度 4。program reset 不清 SRAM 数据，只通过 descriptor 限制可见范围。证据：`rtl/mem/lpu_mem_local_imem.sv:1-5,35,55-57,65-73,76-117`。

Macro Frontend 的实际缓冲为 word FIFO `97x4`、context FIFO `271x4`；scheduler 的 `ISSUE_FIFO_DEPTH` 参数没有生成实体 FIFO，`issue_fifo_level_o` 固定为 0，真正 issue 缓冲是 subsystem 的 `lpu_mem_bank_queue`。证据：`rtl/mem/lpu_mem_macro_frontend.sv:1-5,33,76-178`、`rtl/mem/lpu_mem_macro_scheduler.sv:65-88,188,299-340`。

MEM Macro Decoder 用固定 288-bit reservoir（3×96 bit）解析压缩 stream，字典深度 7；context 输出在 downstream ready 前保持。证据：`rtl/mem/lpu_mem_macro_decoder.sv:30,88-137,173-183,314-608,785-820`。Timed queue 的记录为 `cycle32 + instruction64 = 96` bit、深度 16，无 empty bypass；只在计划周期且 FU ready 时 issue，错过 deadline 形成 sticky fault。证据：`rtl/mem/lpu_mem_bank_queue.sv:19,36-65`。

### 3.3 MXM

每个 MXM 有 Load、Dequant、Compute 三条独立 Program stream。subsystem 根据 destination kind 选择 stream，每条 stream 有自己的 local iMEM frontend 和 16-entry timed queue，之后由 local coordinator 同步/配对 Load 与 Dequant。证据：`rtl/mxm/lpu_mxm_subsystem.sv:1-8,131-165,187-290`。

共同结构：

```text
per-stream Program WRITE/PRIME
 -> 65536x128 local iMEM
 -> two-stage fetch + response FIFO 4x129
 -> word FIFO 4x129
 -> stream-specific Macro decoder/frontend
 -> context FIFO 4 entries
 -> scheduler ingress + 4 active contexts
 -> timed queue 16x81 (cycle32 + normalized issue49)
 -> MXM local coordinator
```

三类 context 宽度不同：Load 257 bit、Dequant 242 bit、Compute 288 bit；对应原生 issue 分别为 29、16、49 bit。证据：`rtl/mxm/lpu_mxm_load_macro_frontend.sv:3-5,33,85-87,138-155`、`lpu_mxm_dequant_macro_frontend.sv:3-5,33,83-85,135-152`、`lpu_mxm_compute_macro_frontend.sv:4-6,34,86-88,141-159` 中各自的 `CONTEXT_WIDTH`、word/context FIFO 与 scheduler 实例。共同 decoder core 使用 384-bit reservoir（3×128 bit）和 7-entry dictionary：`rtl/mxm/lpu_mxm_macro_decoder_core.sv:42,191-192`。

### 3.4 VXM 与 SXM standalone frontends

这些模块是“源码存在但当前未集成 ProgramLoad”的证据，不能按已贯通功能统计：

- VXM frontend：word FIFO `97x4`、context FIFO `288x2`、timed issue FIFO `128x4`；需要外部 descriptor/base cycle，decoder 一次消费一个 96-bit descriptor，不是 MEM/MXM 的三词 reservoir。证据：`rtl/vxm/lpu_vxm_macro_frontend.sv:1-88,107-212`、`rtl/vxm/lpu_vxm_macro_decoder.sv:110-122`、`rtl/vxm/lpu_vxm_timed_issue_queue.sv:24,65-67`。
- SXM Transpose：word FIFO `97x4`、context FIFO `608x2`、timed issue FIFO `448x4`。Permute：word FIFO `97x4`、context FIFO `613x2`、timed issue FIFO `448x4`。两者各自 decoder 从 96-bit Run2D descriptor 生成一个 context，scheduler 为单 ingress、单 active context。证据：`rtl/sxm/lpu_sxm_transpose_macro_frontend.sv:2-4,29,95-97,138-183` 与 `lpu_sxm_permute_macro_frontend.sv:2-4,29,97-99,141-187`，以及对应 decoder/scheduler 文件。

## 4. CModel 当前实际数据流

### 4.1 ProgramLoader

```text
vector<ProgramImage> images_（逐 image 串行）
       |
       v
C2cDmaEngine::issue(Load)
  32-byte physical vector，FIFO/credit depth 16
       |
       v
route_ : vector<optional<Packet>>
  CP7B 当前 1 个软件弹性槽，每槽最大 32 B
       |
       v
reassembly_ : deque<byte>，显式限制 64 B
       |
       v
commits_ : deque<PendingWrite>
  address + due cycle + 64 B storage
       |
       v
InstructionControlUnit::write_program_word()
       |
       v
DistributedIcuQueue local vector<optional<word>>
       |
       +-- Native: pending fetch -> IQ deque(16) -> queue-control/issue
       |
       `-- Macro: pending fetch -> 3-word reservoir -> DDB(8)
                    -> active contexts -> issue
```

`ProgramLoader` 明确说明 DDR 由 enclosing system 每物理周期 tick 一次，且程序加载复用实际 `C2cDmaEngine` 的 FIFO/credits。构造函数直接持有 `Ddr4Model`、`InstructionControlUnit` 和 `C2cDmaEngine`：CModel `include/ftlpu/system/program_loader.hpp:87-99,387-401`。

加载行为：

- `submit()` 先验证 storage 顺序、destination 唯一性和 image 合法性，再 reset 旧程序；每个 `ProgramImage` 带 source address、word/byte count、kind/index/mode/storage。证据：`program_loader.hpp:153-228`、`include/ftlpu/system/program_image.hpp:6-30`。
- image 串行处理；当前 image 的全部 word commit 后才 `begin_queue()` 下一个 image，因此没有 RTL 那种不同 destination 经分发网络并行写入的行为。证据：`program_loader.hpp:258-282,373-385`。
- DMA 返回被拆成每周期最多 32 B 的 packet；1-stage route 在 endpoint 有 64 B 空间时前移。完成 word 后按 `imem_write_ii=1`、commit latency=1 形成 PendingWrite，并直接调用 ICU 写本地存储。证据：`program_loader.hpp:285-358,382-398`；参数见 CP7B `hardware_config.hpp:78-83`。
- 所有 image commit 后进入 Prime，逐周期调用 `icu_.tick_program_frontends()`，直到所有 frontend ready，再等待 start/dependency gate。证据：`program_loader.hpp:235-255`。

这里复用的是 **C2C 数据 DMA 行为模型**，不是 CModel 中另一个 `InstructionFetchDma`。后者有 AXI 256-bit、默认 burst 16、单 outstanding、bounded assembler 和 word sink backpressure，但 grep 调用关系只到其单元测试，没有接入 `ProgramLoader` 或 system top。证据：`include/ftlpu/icu/instruction_fetch_dma.hpp:23-55,64-94,262-340,483-498`。所以不能用它的周期行为替代当前 ProgramLoader 的真实路径。

### 4.2 FU-local queue 和 Macro frontend

CModel ICU 数量为：VXM 8、MEM 208、MXM 4 个单元×3 stream、SXM 两类×2 hemisphere。别名和深度在 `include/ftlpu/system/icu.hpp:103-216`，实际数组在 `:2458-2477`。

CP7B 有效配置：

| FU | Word / iMEM / Native IQ | Macro active context |
|---|---|---:|
| MEM | 96 bit / 131072 / 16 | 32 |
| MXM | 128 bit / 65536 / 16 | 40 |
| VXM | 96 bit / 2048 / 16 | 4；但 Macro image 被拒绝 |
| SXM | 96 bit / 2048 / 16 | 4 |

证据：`out/checkpoint_7b_dma_ddr/generated/include/ftlpu/core/hardware_config.hpp:24-53`。

`DistributedIcuQueue` 是一个物理 endpoint 的合并行为模型，内部同时拥有 local iMEM、raw Macro iMEM、native IQ、pending fetch、queue-control state、Macro decoder 和 active-context storage。证据：`include/ftlpu/icu/distributed_queue.hpp:425-445,496-570,2758-2819`。

- Native path 每周期最多发起一个 fetch；响应成熟后先 commit，再 dispatch/issue、再发起新 fetch，IQ 是显式上限 16 的 `std::deque`。证据：`distributed_queue.hpp:1016-1080,1973-2050`。
- Macro path 先让已有 active context issue，再接纳本周期解出的 context，因此新 context 最早下一周期可见。证据：`distributed_queue.hpp:1903-1946`。
- `MacroBitstreamDecoder` 实现 `local iMEM -> bounded reservoir -> parser -> finite DDB -> one-context lazy expander`；当前 reservoir 3 words、DDB 8、fetch II=1、fetch latency=1。证据：`include/ftlpu/icu/macro_bitstream_decoder.hpp:125-162,220-227,430-505,877-1003,1839-1875` 与 CP7B 配置 `hardware_config.hpp:46-53`。
- CModel 的 active contexts 是固定容量硬件状态模型，不应因底层 C++ 容器实现而视为无限软件队列。证据：`distributed_queue.hpp:2773-2779`。
- image 验证明确拒绝 VXM Macro，但允许 SXM Macro；Program word 通过函数直接写到相应 queue，再 seal descriptor。证据：`include/ftlpu/system/icu.hpp:386-467`。

## 5. 多对多模块映射

| 功能 | VMODEL RTL | CModel C++ | 关系 | 判断 |
|---|---|---|---|---|
| Program 请求/命令 | `lpu_program_load_controller` + 32×155 command RAM | `vector<ProgramImage>` + `ProgramLoader::submit/begin_queue` | Merged/Different | CModel 没有 RTL opcode/PC/WAIT/JUMP 命令机；用已解析 image 顺序驱动。 |
| Instruction DMA | `lpu_instruction_dma`，256b、多 outstanding、ID/RX FIFO | 当前实际为 `C2cDmaEngine`；另有未集成 `InstructionFetchDma` | Different | CModel 使用数据 DMA 抽象，协议和 outstanding 结构不同。 |
| Descriptor 配对 | config FIFO 16×155 | 当前 `images_[current_]` | Missing/Merged | CModel 单 active image，所以不需要硬件 FIFO 配对，但不能复现其 occupancy/backpressure。 |
| Burst 重组 | 64 B RTL reservoir + valid/ready | `deque<byte>`，限制 64 B | Partial | 容量相同；状态、同拍 pop/append、错误检测和握手不同。 |
| Program Word FIFO | 8×164 | `deque<PendingWrite>` | Partial/Different | PendingWrite 是提交事件，不是相同数据边界；无明确 8-entry 深度。 |
| scope/BOTH 分发 | elastic + distributor + 双分支 done | 无 | Missing | CModel image 已指定一个 queue，不能表现 BOTH 半区分叉反压。 |
| Relay 网络 | 2×165 ingress + 14 个 2×165 relay buffer | `route_` 1 个 optional packet | Merged/Different | CModel 仅抽象固定路由延迟，不表现拓扑、仲裁和局部拥塞。 |
| FU endpoint 写入 | valid/ready、RR、commit ack | `write_program_word()` 直接函数调用 | Different | 功能可映射到“word committed”，周期和并行性不可直接对应。 |
| local iMEM | MEM 96×131072，MXM 128×65536；同步读 | `vector<optional<word>>`，同深度 | Partial | 容量对应；RTL 两级 fetch+FIFO，CModel pending fetch=1 周期。 |
| MEM/MXM Macro parse | FU-specific decoder/frontend | generic `MacroBitstreamDecoder` + FU codec | Partial | 语义目标相同，但内部资源、上下文深度和周期不同。 |
| timed issue buffering | RTL per queue/stream 16-entry timed queue | `DistributedIcuQueue` active contexts/issue logic | Merged/Different | CModel 没有一一对应的末级 16-entry FIFO occupancy。 |
| VXM Program frontend | boundary present，subsystem ready=0；standalone Macro frontend 未接入 | 8 local queues可装 Native；Macro rejected | Different | 两边都没有贯通 VXM Macro，但 Native Program 行为也不相同。 |
| SXM Program frontend | boundary present，subsystem ready=0；standalone frontends 未接入 | Transpose/Permute local queues支持 Macro | Missing in RTL integration / Different | 不能以 CModel SXM 行为当作当前 RTL Golden。 |

## 6. 关键周期差异

| 抽象事件 | VMODEL | CModel 当前实现 | 风险 |
|---|---|---|---|
| 接受程序请求 | controller 空闲时接受命令，活动记录再由 RAM 正沿取出 | `submit()` 一次验证并立即建立整个 image 列表 | CModel 少了命令 RAM/PC 的周期与容量。 |
| DDR 请求/返回 | 256b，多 outstanding 16；ID/RX FIFO 分离 | `C2cDmaEngine` 以 32B physical vectors，深度 16 | 只可比较外部 request/return 事件，内部信用不可直接同名比较。 |
| Program word 形成 | 64B reservoir，支持 pop+append，同步 valid/ready | tick 内从 `deque` 取 12/16B 并生成 PendingWrite | 容量相近，边界周期不同。 |
| 目标路由 | 入口2级、BOTH分叉、14 relay、RR、commit ack | 当前 1 个 route slot，image 目标已确定 | CModel无法产生真实局部拥塞和跨分支反压。 |
| local iMEM 写可见 | FU endpoint handshake 后正沿写；commit_valid 再延迟一拍影响 idle | due cycle 调用 `write_program_word()` | 完成周期定义不同。 |
| local iMEM 读 | 两级同步流水，response FIFO4 | CP7B fetch latency=1 的 pending fetch | 至少一拍差异，并影响 FIFO occupancy。 |
| Macro parser | FU-specific 3-word reservoir、word/context FIFO | 通用 3-word reservoir + DDB8 | 解码语义可对齐，内部 stage/credit 不同。 |
| active contexts | MEM/MXM 4 | MEM32、MXM40 | 对 decoder backpressure 和长程序 Prime/issue 有重大影响。 |
| 末级 issue | 独立 timed queue16；deadline fault | active context 直接驱动 endpoint/队列控制 | Burst 吸收、occupancy、late 行为不能直接对等。 |
| 多目标装载 | 网络可让不同 endpoint 并行接受 | image 严格串行，下一 image 等上一 image commit | 总加载周期和并行反压显著不同。 |

## 7. 反压拓扑

### 7.1 VMODEL

```text
FU local iMEM/PRIME not ready
  <- endpoint arbiter / commit pending
  <- relay head cannot dequeue
  <- relay buffer fills（本级 registered-ready）
  <- distributor branch stalls；BOTH 等待另一分支 done
  <- ingress elastic FIFO fills
  <- Program Word FIFO fills
  <- reassembler cannot emit word，64B reservoir fills
  <- DMA RX FIFO fills
  <- DDR response ready deasserts / outstanding credit最终耗尽
  <- controller LOAD transaction cannot retire
```

Decoder/issue 侧：

```text
FU not ready / timed deadline constraint
  <- timed queue cannot dequeue
  <- scheduler cannot place more issue state
  <- active contexts/context FIFO fill
  <- Macro decoder context_valid holds
  <- word FIFO / iMEM response FIFO fill
  <- local iMEM fetch stops
  <- PRIME frontend_ready remains low
  <- global program start remains gated
```

其中真正的解耦边界是 DMA ID/RX FIFO、config FIFO、64B reassembler、word FIFO、ingress elastic、每级 relay、local iMEM response FIFO、decoder word/context FIFO 和 timed issue queue。Relay 用 registered occupancy 切断组合 ready 链；其他基于 `lpu_queue_fifo` 的 FIFO 允许 full+dequeue replacement。

### 7.2 CModel

```text
local commit admission guard / word II
  <- commits_ 未到 due cycle
  <- reassembly_ 无 12/16B 空间
  <- route_.back() 保持
  <- upstream route slot 保持
  <- C2cDmaEngine receive queue 不被 pop
  <- DMA credits / DDR request推进受限
```

Frontend：

```text
consumer / queue control 阻塞
  <- active context不能 issue
  <- active-context credit耗尽
  <- DDB不能 admission
  <- parser/reservoir停止
  <- pending fetch不再发起
```

CModel 这条链是有限行为模型，但它没有复刻 VMODEL 的分支拓扑，因此只能用于“某 image 的总体带宽与本地 queue 消费”研究，不能直接给出 RTL relay/endpoint FIFO 的最优深度。

## 8. 功能差异分类

### A. CModel 缺少 RTL 已实现功能

- Program command RAM、opcode/PC、WAIT/JUMP/END 控制状态机。
- 专用 Instruction DMA 的 descriptor FIFO、outstanding-ID FIFO 和 ordered response checking。
- scope/BOTH 两半区复制、两条 7-relay 物理分发链、endpoint round-robin 和一拍 commit acknowledgement。
- RTL local iMEM 两级同步读与独立 response FIFO 的精确周期。
- MEM/MXM 末级 16-entry timed issue queue 的独立 occupancy/deadline 边界。

### B. 功能相近但时序不同

- 64B burst reassembly。
- local iMEM 容量和 word 宽（MEM/MXM 对得上）。
- 3-word Macro reservoir 和 context 展开。
- 有限 DMA/FU backpressure；但状态边界与传播延迟不同。

### C. Buffer 结构不同

- RTL config FIFO/word FIFO/relay FIFOs 被 CModel 的 `current image + route_ + commits_` 合并。
- RTL MEM/MXM active contexts=4，CModel=32/40。
- RTL timed issue queue=16，CModel 没有直接一一对应结构。
- CModel native IQ=16；当前 RTL Macro frontends 不是同名 IQ 结构。

### D. 控制机制不同

- RTL 以 valid/ready 和正沿提交为主；CModel 以 `tick()`、due cycle 和直接函数调用组织。
- RTL 写入目标可在多分支并行，CModel image 严格串行。
- RTL VXM/SXM Program endpoint 不可接受；CModel SXM 可完整装载，VXM Native 可装载但 Macro 被拒绝。

### E. 可合理保留的软件抽象

- `ProgramImage` 元数据、统计字段和 SHA/manifest 类信息不需要硬件化。
- 若研究只比较“Program Word 被某 queue 接受”“Macro context 出现”“FU issue”这些抽象事件，CModel 无需复制每个 RTL 内部变量；但必须补足会改变事件周期、occupancy、吞吐或反压的 P0/P1 边界。

### F. 暂无法确认

- 当前脏工作树尚未用相同 binary 对 RTL 与 CModel 做逐周期回归，因此 Macro 展开内容/顺序虽可从 codec 结构静态映射，尚不能声明完全一致。
- standalone VXM/SXM frontend 是否计划按现有接口接入 ProgramLoad，源码没有形成完成的集成证据。
- ASIC 最终 SRAM macro 端口、物理读延迟和容量仍未由本调查确定；此处只记录当前 RTL/CModel 参数。

## 9. 测试证据边界

本轮未运行测试。源码中存在以下覆盖点：

- RTL config FIFO 的 full replacement、underflow：`sim/tb/program_load/lpu_program_config_fifo_tb.sv:90-124`。
- RTL reassembler 的 96-bit、跨 burst、downstream stall 和非法尾部：`lpu_program_reassembler_tb.sv:174-245`。
- RTL elastic stage 的两级填满：`lpu_program_load_elastic_stage_tb.sv:63-83`。
- RTL relay 的 local/forward backpressure、满、II=1、fault/reset：`lpu_program_load_relay_tb.sv:317-517`；relay-chain 目标与 blocked record：`lpu_program_load_relay_chain_tb.sv:242-300`。
- CModel ProgramLoader 测试源码覆盖 DMA credit、prefetch、write II 和 image ordering：`tests/subsystem/system/program_loader_test.cpp:113-119,185,221-250`。
- CModel raw Macro decoder 测试源码覆盖 fetch latency、reservoir/DDB、context full 与 MEM/MXM integration：`tests/unit/icu/raw_macro_decoder_test.cpp:149-150,224-240,322-430`。

这些证据最多支持“各自实现有测试意图/既有测试”，不支持“双方同激励逐周期相等”。

## 10. 后续同步建议与迁移顺序

1. **冻结可执行事件契约**：统一定义并记录 `DMA request/response`、`Program Word formed`、`route accepted`、`local iMEM write committed`、`fetch request/response`、`context admitted`、`FU issue`。跨模型只比较这些事件，不机械比较内部类/寄存器名。
2. **先同步 ProgramLoad P0 边界**：建立专用 instruction-load DMA 行为路径；加入 descriptor/config、bounded RX、64B reassembly、8-entry word FIFO，以及明确的 valid/ready/Current-Next 更新。不要继续把未集成 `InstructionFetchDma` 当成当前路径已经完成。
3. **同步真实 distribution**：至少建模 2-entry ingress、BOTH 独立完成、每半区 7×2 relay occupancy、endpoint arbitration 和 commit acknowledgement。若先保留抽象模型，也必须能产生等价的分支 backpressure/延迟事件。
4. **先解决覆盖范围不一致**：明确 Golden 的暂定范围是 MEM/MXM，还是先把 RTL VXM/SXM adapters 接通。未做决定前，不应拿 CModel SXM Macro 与当前 RTL 比周期。
5. **同步 local iMEM fetch**：从 CP7B 的 1-cycle pending fetch 调整到 RTL 的两级请求/返回和 4-entry response FIFO语义；保持 Program reset 不清 SRAM 数据、仅 descriptor 限定可见性。
6. **同步 Macro 资源边界**：将 MEM/MXM active context 深度与 RTL 4-entry 基线对齐，显式表示 word/context FIFO 和末级 timed queue；再验证 stall 下的输出保持、展开顺序与 deadline。
7. **建立分层 Golden**：先 ProgramLoad 单 image，再 BOTH/多 endpoint，再 local iMEM fetch，再单 Macro，再多 context/FU stall；最后才做完整程序。每层均需相同输入和逐周期事件 trace。
8. **深度扫描顺序**：先扫 P0（RX/reassembly/word/distribution/local fetch），再扫 P1（decoder word/context/active/timed issue）。P2 控制缓冲最后；P3 软件元数据不进入硬件 PPA 结论。

在完成上述同步和逐周期验证前，当前 CModel 适合做功能与高层带宽探索，但还不能作为 VMODEL ICU 的周期精确替代模型。
