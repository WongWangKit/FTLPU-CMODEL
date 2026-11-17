# Checkpoint B0：Attention Workload 审计

日期：2026-11-15  
状态：B0 调查完成；未实现 Unbounded FIFO，未执行 FIFO 深度扫描  
证据标记：`CONFIRMED`=源码/配置明确支持；`TESTED`=本轮在 Stage A 现有构建中复验；`HISTORICAL`=只存在于历史提交或历史工件；`UNKNOWN`=现有证据不足。

## 1. 结论

`CONFIRMED`：当前 Stage A 正式基线中，**不存在一条已经验证、可完整运行，并同时覆盖完整 Attention 数值计算与 ProgramLoader→Instruction DMA→Reassembly→Word FIFO→Ingress→Relay→local iMEM→Macro Frontend→FU 的端到端 Attention Layer 工作负载**。

第一轮 FIFO 实验必须拆成两个互补但不可拼接的基线：

1. **ProgramLoader Delivery Baseline**：使用 `subsystem.program.loader`，观察装载、DMA、重组、分发和 local iMEM 供给链路；其执行负载主要是 NOP/BARR，不能代表 Attention FU 消费压力。
2. **Direct ICU FU/Macro Baseline**：优先使用 `kernel.prefill_weight_streams`，辅以 `subsystem.sxm.mem_transpose`；它们具有真实 MEM/MXM/SXM 周期行为和数值检查，但绕过 ProgramLoader、Instruction DMA、Ingress 与 Relay。

两组 Trace 不能拼接成一条端到端 Occupancy Trace。完整 Attention 保留为 Level 3，待恢复/移植当前基线可运行的完整程序后再进入正式实验。该结论已提交“CMODEL 建模”聊天并获批准。

## 2. Stage A 基线核对

| 项目 | 结果 | 证据 |
|---|---|---|
| 正式提交 | `ff1aa41321e07667ebd37f3ebefdf8053f0ee0e5`，`FTLPU-CMODEL-DEV/main` | 远端递归 tree 与本地 388 个受控 blob 逐项核对，0 mismatch |
| 本地工作区 | `D:\LPU\FTLPU-CMODEL-clock` 不是 Git worktree，但受控源码与上述提交一致 | 本轮只读 blob 核对；未覆盖已有未跟踪构建/日志 |
| DMA Golden | 108 rows，`mismatches=0`，`semantic_failures=0` | `timing_refactor/rtl_golden/compare_instruction_dma_traces.ps1`；`out/icu_dma_golden/*.csv` |
| ProgramLoader | `configured_load_cycles=170` | `tests/subsystem/system/program_loader_test.cpp:75-123`；本轮复验通过 |
| 统一时钟 | Clock 与 SystemPhase 独立；LOW→validate→commit→HIGH→NEGEDGE | `include/ftlpu/core/timing.hpp:54-217`；`include/ftlpu/system/tsp_slice_system.hpp:156-177,534-546` |
| 当前硬件配置 | target=`ftlpu-lpu32`；Program load：Outstanding/RX=16、Word=8、Ingress=2、Relay=2、Reassembly=64 | `config/ftlpu-lpu32.json:257-271`；`out/icu_rtl_alignment/generated/include/ftlpu/core/hardware_config.hpp:88` |
| ICU frontend | local response=4、macro word=4、context=4、timed issue=16、reservoir=3、DDB=8 | `config/ftlpu-lpu32.json:4-39` |
| MXM accumulator | 4 contexts × 32 rows = 128 rows | `include/ftlpu/core/hardware_params.hpp:115-118`；generated config line 118 |

`CONFIRMED`：Stage A 之后没有发现受控源码差异；现有构建输出不构成源码变更。本轮没有修改 Stage A、VMODEL、Clock 或 InstructionDmaProtocolCore。

## 3. 当前 Attention 相关入口

### 3.1 SmolLM2 Prefill Attention

入口：`tests/model/smollm2/prefill_attention_test.cpp`；CTest：`model.smollm2.prefill_attention`。

| 审计项 | 结论 | 证据 |
|---|---|---|
| 默认规模 | Sequence=8、Heads=4、Head Dimension=8、Hidden=32 | `prefill_attention_test.cpp:20-24` |
| 指令注入 | 直接调用 `enqueue_mem/mxm/sxm/vxm`，不经 ProgramLoader | `prefill_attention_test.cpp:152-258` |
| Q/K/V | 合成或调用者提供后直接写 MEM SRAM | `prefill_attention_test.cpp:261-289,367-467` |
| Q/K/V Projection | 默认 tile 不包含 | `run_attention_tile` 调用链，`prefill_attention_test.cpp:1258-1313` |
| RoPE/QK/Softmax/P×V | 均有真实 TspSliceSystem 调度；数值检查覆盖这四段 | `prefill_attention_test.cpp:1085-1249,1258-1313` |
| Output Projection | 不包含 | 同上 |
| Prefill helper | 接收 `[128,576]`，按 8×32 tile 调用；Q/K/V tile 都取自输入，并非全局 Attention | `prefill_attention_test.cpp:1380-1431` |
| Decode helper | 使用前 7 个 cache token + 新 token；仍调用局部 tile | `prefill_attention_test.cpp:1434-1492` |
| `FTLPU_FULL_INTEGRATION` | CMake 未设置；当前 helper 的 `hidden_limit` 三元两侧相同，不能扩大执行覆盖 | `prefill_attention_test.cpp:1329-1332,1398-1399,1458-1459`；`CMakeLists.txt:233-255` |
| Trace | 只写单行 phase 摘要，不是完整逐周期 FIFO/Issue Trace | `prefill_attention_test.cpp:1316-1327` |
| 当前可运行性 | `TESTED FAIL`：setup 阶段使用 accumulator address 512/1024，超过当前 128-row SRAM | `prefill_attention_test.cpp:80-81`；运行错误原文已在终端记录 |

因此，该文件提供有价值的 Attention tile 语义和数值 checker，但当前正式配置下连默认入口都不能完成；即使修复地址，它也不是完整的 128×576 SmolLM2 Attention Layer。

### 3.2 Decoder Layer Decode

入口：`tests/model/decoder/decode_layer_test.cpp`。

- `CONFIRMED` 默认模型规模包含 Prefill=128、Decode position=128、Sequence=129、Hidden=32、Intermediate=64（`decode_layer_test.cpp:22-29`）。
- `CONFIRMED` GEMV（可用于 Q/K/V/O/gate/up/down projection）与 QK scores 通过 TspSliceSystem、MEM/MXM 和真实 tick 执行（`decode_layer_test.cpp:190-383`）。
- `CONFIRMED` Prefill KV cache 与 append cache 直接写 SRAM，不经过硬件 cache fill 指令流（`decode_layer_test.cpp:386-463`）。
- `CONFIRMED` RMSNorm、RoPE、Softmax、P×V 概率组合、Residual、SwiGLU 等关键段由 C++ reference 直接计算；源码也显式标为 functional composition/reference（`decode_layer_test.cpp:588-595,665-718,787-863`）。
- `CONFIRMED` 指令全部经 direct `icu.enqueue_*`，不经过 ProgramLoader/DMA/Relay（`decode_layer_test.cpp:73-173`）。
- `TESTED FAIL` 当前配置在 accumulator address 128 处越过 128-row SRAM。

因此它是“真实硬件 GEMV/QK + C++ 功能拼接”的混合测试，不是完整硬件 Decoder Layer，也不适合作为第一轮端到端 FIFO 工作负载。

### 3.3 其他当前 Model/Kernel 入口

| 测试 | 本轮结果 | 价值与边界 |
|---|---|---|
| `model.smollm2.decode_attention` | `TESTED FAIL`：VXM input Buffer 在 Bundle active 时被重配置 | Direct ICU；没有外部指令供给链路 |
| `model.smollm2.decode_ffn` | `TESTED FAIL`：MEM SRAM row 越界 | Direct ICU；不是 Attention |
| `kernel.w8a16_projection` | `TESTED FAIL`：accumulator address 1536 越过 128 rows | Projection kernel 源码可复用，但当前配置不可运行 |
| `kernel.w8a16_swiglu` | `TESTED FAIL`：同上 | FFN kernel，不是完整 Decoder/Attention |
| `kernel.rmsnorm` | `TESTED FAIL`：VXM active 重配置 | Direct ICU；当前不可运行 |
| `kernel.prefill_weight_streams` | `TESTED PASS`：6 个 case 数值全通过 | 真实 MEM+MXM Load/Dequant/Compute；Direct ICU；最佳 FU 消费基线 |
| `subsystem.sxm.mem_transpose` | `TESTED PASS`：96 matrices、双半区 | MEM + packed Macro SXM + MEM；Direct load；最佳 Macro/SXM 基线 |
| `subsystem.program.loader` | `TESTED PASS`：170 load cycles | 最完整的 ProgramLoader/DMA/Relay 供给链路；执行内容主要 NOP/BARR |

`kernel.prefill_weight_streams` 的 6 个 end cycles 分别为 220、360、380、408、700、728，均 `max_abs_error=0`。其源码明确限定为 resident-MEM projection、单物理 MXM，并非 whole-layer compiler lowering（`tests/kernel/prefill_weight_streams_test.cpp:17-20,180-290`）。

`subsystem.sxm.mem_transpose` 直接装入 MEM 指令与 packed Macro SXM fixture，真实 tick 后比较 96 个矩阵（`tests/subsystem/sxm/sxm_mem_transpose_test.cpp:11-50`）。

## 4. 历史 SmolLM2 Attention Schedule

`docs/attention_pipeline_optimization.md:3-28` 描述 Sequence=128、Hidden=576、Q Heads=9、KV Heads=3、Head Dimension=64、W8A16 Q/K/V+O Projection、FP16 RoPE、three-pass Softmax、GQA，并声称 optimized=81,273 cycles。

审计结论：

1. `HISTORICAL` 对应的大型 C++ schedule 源码可从本地 Git 历史恢复：`35e87de:tests/integration/smollm2_attention_test.cpp` 或 `e84c228:tests/model/smollm2/prefill_attention_test.cpp`；它**不在当前 Stage A 正式 tree**。
2. `CONFIRMED` 当前 `docs/traces/smollm2_attention_layer_baseline.csv` 与 `smollm2_attention_schedule.csv` 字节完全相同，SHA-256 为 `57656A57F26570F0D2CDCF2A6FDB30F882B83EE8BBDA3C886FB313CFD2C61955`。
3. 两个 CSV 各有 284,622 个事件，最大 `end` 为 95,159，不等于文档的 81,273；它们是历史离线 schedule 事件，不是当前 Runtime FIFO/Issue Trace。
4. 文档声称的 `FTLPU_SCHEDULE_TRACE_ONLY=1` 在当前源码和已定位的历史完整源码中均未找到实现。
5. `UNKNOWN/NOT REPRODUCIBLE`：81,273 cycles 不能由当前 Stage A 直接重新生成。恢复它至少需要移植历史程序、重新适配当前硬件配置/地址约束，并建立当前 ProgramImage 与运行入口；这些工作超出 B0。

历史 CSV 可用于恢复时的参考，不得作为当前正式、可重复的实验结果。

## 5. 候选工作负载真实链路矩阵

`Y`=实际经过；`N`=绕过/未使用；`P`=部分或仅辅助路径；`FAIL`=当前配置未完成。

| Workload / Entry | Loader | DMA | Reasm | Word | Ingress | Relay | local iMEM | Macro FE | FU | Cycle | Golden | Trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `subsystem.program.loader` / `program_loader_test.exe` | Y | Y | Y | Y | Y | Y | Y | P | NOP/BARR；另有 mixed MEM/MXM image 装载 | 170 load cycles | 结构/计数断言 | 聚合 stats，非统一逐周期 FIFO trace |
| `kernel.prefill_weight_streams` | N | N | N | N | N | N | Y（direct enqueue） | N/native | MEM+MXM Load/Dequant/Compute | 220/360/380/408/700/728 | Y，6 cases | 可选 workload CSV；不含上游链路 |
| `subsystem.sxm.mem_transpose` | N | N | N | N | N | N | Y（direct/raw load） | Y，packed Macro | MEM+SXM | 未由测试打印 | Y，96 matrices | 无完整统一 trace |
| 当前 SmolLM2 Attention tile | N | N | N | N | N | N | Y（direct enqueue） | P | MEM/MXM/VXM/SXM | FAIL/setup | checker 存在但未完成 | 仅单行 phase 摘要 |
| Decoder Layer Decode | N | N | N | N | N | N | Y（direct enqueue） | N | MEM/MXM 部分；其余 C++ | FAIL/setup | 混合 reference | 无端到端 FIFO trace |

ProgramLoader 的实际调用链由以下实现建立：

```text
ProgramLoader::start
  -> InstructionFetchDma / shared DDR
  -> response assembler
  -> program word FIFO
  -> distribution ingress
  -> East/West relay chain
  -> destination commit
  -> Icu local raw/native iMEM
  -> queue frontend/prefetch
```

证据：`include/ftlpu/system/c2c_dma_system.hpp:14-45`、`include/ftlpu/system/tsp_slice_system.hpp:178-186,282-301,773-809`、`include/ftlpu/system/program_loader.hpp:409-438,580-699`、`include/ftlpu/system/icu.hpp:450-469`。当前 ProgramLoader 的 RTL-style destination decode 只覆盖 Macro MEM、MXM Load/Dequant/Compute（`program_loader.hpp:580-585`），不能据此推定所有 VXM/SXM Attention 指令已经具备同一路径。

## 6. 当前统计能力

- `CONFIRMED` Instruction DMA 已有 burst/input/assemble/enqueue 事件周期与 peak/full/backpressure 聚合统计（`include/ftlpu/icu/instruction_fetch_dma.hpp:154-174`）。
- `CONFIRMED` ProgramLoader 已统计 DMA/Word/Ingress/Relay 的 peak/full/wait/read/write（`include/ftlpu/system/program_loader.hpp:41-102,502-510,608-675`）。
- `CONFIRMED` ICU frontend 有 response/word/context/timed/DDB/active samples、peaks 与 full 聚合统计（`include/ftlpu/system/icu.hpp:37-125`）。
- `CONFIRMED` DistributedQueue 暴露当前/峰值 occupancy（`include/ftlpu/icu/distributed_queue.hpp:1427-1478`）。
- `PARTIAL` `IcuQueueCycleTrace` 只记录 fetch/issue PC、IQ before/after/action（`distributed_queue.hpp:205-215`），并未覆盖所有待研究 FIFO 的逐周期 occupancy、直方图和 stall cause。

因此当前统计足以做 B0 路径审计，尚不足以直接完成正式 Unbounded/Finite 对比；B1 应先加入统一且不改变时序的观测层，并用默认 Finite 配置证明零行为变化。

## 7. 工作负载分层建议

### Level 1：可立即复现的互补基线

- **L1-A Delivery**：`subsystem.program.loader`。输入为确定性 ProgramImage；覆盖 DMA→Reassembly→Word→Ingress→Relay→local iMEM，完成周期 170；不用于推断真实 FU 消费压力。
- **L1-B FU consumption**：`kernel.prefill_weight_streams`。覆盖真实 MEM/MXM Load/Dequant/Compute，6 个数值 Golden 全通过；不用于推断上游装载积压。
- **L1-C Macro/SXM**：`subsystem.sxm.mem_transpose`。覆盖 packed Macro、双半区 MEM/SXM 与数值结果。

### Level 2：需要后续建立的混合并发基线

以 ProgramLoader mixed MEM/MXM image 为起点，增加真实可消费的 FU 程序，并覆盖多 destination、East/West 并行、连续装载与消费。在其出现前，不得把现有 loader test 或 direct ICU test描述为 Level 2 端到端负载。

### Level 3：完整 Attention

目标仍为 Sequence=128、Hidden=576、Q Heads=9、KV Heads=3、Head Dimension=64，包含 Q/K/V Projection、RoPE、QK、Softmax、P×V、Output Projection，并具有 ProgramImage、完整供给链路、数值 Golden 与逐周期 Trace。当前缺少：

1. 当前 tree 可构建、可运行的完整 Attention 源码；
2. 与当前 128-row MXM/MEM/VXM 约束兼容的地址/分块方案；
3. 能表达所有目标 FU 指令并经过 ProgramLoader 的 ProgramImage/路由；
4. 不使用 C++ reference 替代关键硬件阶段的执行入口；
5. 统一的 FIFO occupancy、issue 与数值结果 Trace。

## 8. B0 停止结论

B0 已回答“当前是否已有可用于正式 FIFO 深度实验的完整 Attention”：**没有**。第一轮只能采用经批准的两组互补基线，且必须分别解释。下一步应先完成 B1 观测层与默认 Finite 零回归，再考虑 Unbounded；本报告不授权实现、深度扫描或 Attention 新开发。
