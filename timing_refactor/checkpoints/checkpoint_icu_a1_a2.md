# Checkpoint ICU A.1/A.2：Relay Mapping 与 Instruction DMA

## 1. 范围与停止边界

本检查点只完成两项：

1. A.1：按本地 VMODEL 修正 MEM Queue 到七级 ProgramLoad Relay 的映射。
2. A.2：让 ProgramLoader 使用独立 Instruction DMA，经共享 DDR 取回指令。

本轮没有修改 `D:\LPU\FTLPU-VMODEL-main`，没有启动 FIFO depth sweep、
完整 attention workload、共享 iMEM 或 Checkpoint 7C。

## 2. RTL 对照基线

只读对照本地 VMODEL 当前磁盘源码：

- `rtl/core/lpu_pkg.sv`
- `rtl/program_load/lpu_program_distribute_system.sv`
- `rtl/program_load/lpu_program_distribute_relay.sv`
- `rtl/program_load/lpu_program_distribute_relay_chain.sv`
- `rtl/program_load/lpu_instruction_dma.sv`
- `rtl/program_load/lpu_program_load_path.sv`
- `rtl/queue/lpu_queue_fifo.sv`

没有用远端 GitHub 版本替代本地 RTL。当前 RTL 没有独立
`lpu_instruction_dma` 功能 testbench，因此该部分只能做静态对照和编译核对。

## 3. A.1 最终映射

RTL 几何关系：

```text
2 banks/slice × 4 slices/group × 2 groups/relay = 16 banks/relay
104 MEM queues/hemisphere = 6 × 16 + 8
```

| Relay | 半区内 MEM Queue |
|---|---|
| R0 | 0–15 |
| R1 | 16–31 |
| R2 | 32–47 |
| R3 | 48–63 |
| R4 | 64–79 |
| R5 | 80–95 |
| R6 | 96–103 |

全局 Queue 先由 `queue_index / 104` 选择 West/East 半区，再由半区内编号
除以 16 选择 Relay。MXM Load、Dequant、Compute 在各自半区均走尾端 R6。

配置和约束：

- `config/ftlpu-lpu32.json`：`mem_groups_per_relay=2`。
- `cmake/FTLPUHardwareConfig.cmake`、`cmake/hardware_config.hpp.in`：生成配置。
- `include/ftlpu/core/hardware_params.hpp`：派生 banks/relay、relay count，并以
  `static_assert` 检查 104 条 Queue 必须由配置的 7 个 Relay 覆盖。
- `include/ftlpu/system/program_loader.hpp`：使用派生常量计算 destination。

验证：`subsystem.program.relay_mapping` 穷举 East/West 共 208 条 MEM Queue，
并检查两半区三种 MXM stream 均落在 R6。ProgramLoader 代表 Trace 为：

- Queue 4：R0，9 words 对应 9 次 relay write/read。
- Queue 16：R1，9 words 对应 18 次 relay write/read。
- Queue 96：R6，9 words 对应 63 次 relay write/read。

## 4. A.2 最终指令装载链路

```text
ProgramLoader descriptor
  -> InstructionFetchDma command
  -> shared Ddr4Model request/completion
  -> Instruction DMA RX FIFO(16)
  -> 32/64/96/128-bit record assembler
  -> ProgramLoader bus-latency adapter
  -> reassembly(64B)
  -> Program Word FIFO(8)
  -> ingress FIFO(2)
  -> West/East Relay(7 stages × 2 entries)
  -> FU-local iMEM endpoint
```

Instruction DMA 与 C2C Data DMA 是两个独立对象；二者只共享
`Ddr4Model`，因而会竞争同一 DDR bandwidth/round-robin，而不会共享 command、
outstanding、RX 或 completion 状态。

Instruction DMA 当前契约：

| 项目 | 实现 |
|---|---|
| DDR beat | 256 bit / 32 B |
| ID | 8 bit |
| Outstanding | ordered FIFO，深度 16 |
| RX FIFO | 深度 16，full+dequeue replacement |
| record width | 32/64/96/128 bit |
| 检查 | response ID、last、error；sticky fault |
| 反压 | RX full 或 assembler/downstream stall 时停止接收完成项 |
| reset | 清本地状态，并保留 orphan request 直到共享 DDR completion 被安全排空 |
| 时钟 | LOW 准备 Next，POSEDGE 统一提交，HIGH 只观察 |

## 5. 有限容量和可观测性

当前模型不是“无限 DMA FIFO”：

| 边界 | 当前深度 | 可观测量 |
|---|---:|---|
| Instruction DMA outstanding | 16 | current/peak、credit wait、request/response |
| Instruction DMA RX FIFO | 16 | current/peak、full cycles、backpressure |
| record assembler | 有界 | occupancy peak、稳定 output |
| reassembly | 64 B | occupancy peak |
| Program Word FIFO | 8 | occupancy peak、full、read/write |
| ingress FIFO | 2 | occupancy peak、full、read/write |
| Relay | 每半区 7×2 | 每级 occupancy、full、read/write |

因此下一阶段可以从真实程序的消费/反压 Trace 中读取这些边界，但本检查点不据此
推导最优 SRAM/FIFO 深度。BOTH、endpoint RR/commit-valid、独立 config FIFO 和
PRIME FIFO 尚未一一建模，若要纳入深度结论必须先补齐范围。

## 6. 验证结果

### CModel

- Release 全量编译：PASS。
- 定向测试 9/9 PASS：
  - `unit.core.timing`
  - `unit.icu.clocked`
  - `unit.icu.instruction_fetch_dma`
  - `subsystem.c2c.dma_ddr4`
  - `subsystem.c2c.dma_clocked`
  - `subsystem.system.timing`
  - `subsystem.program.loader`
  - `subsystem.program.relay_mapping`
  - `subsystem.program.configuration`
- ProgramLoader configured load：170 cycles。
- 全量 CTest：72/79 PASS；7 个失败均为此前冻结类别，没有新增 ICU、DMA、
  ProgramLoader 或 timing 失败。

### 本地 VMODEL

- `lpu_program_load_relay_chain_tb`：`TEST_PASS checks=80`。
- `lpu_queue_fifo.sv` + `lpu_instruction_dma.sv`：ModelSim 编译 PASS。
- Instruction DMA 功能同激励 RTL/CModel Golden：PARTIAL，因为本地 RTL 没有
  独立功能 TB；未将 compile PASS 冒充功能 PASS。

## 7. 结论

A.1 已由本地 RTL 结构、208 条 Queue 穷举测试和 Relay 自检共同支持。
A.2 已在 CModel 中真实接入 ProgramLoader，并与 C2C Data DMA 保持独立、共享 DDR；
CModel 功能和时序回归通过。剩余证据缺口是 Instruction DMA 的 RTL 功能 Golden，
不是 CModel 仍在使用 C2C DMA。到此按范围停止。
