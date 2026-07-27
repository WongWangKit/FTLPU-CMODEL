# FTLPU-CMODEL

[English](README.md) | [简体中文](README.zh-CN.md)

FTLPU-CMODEL 是一个面向 FTLPU/TSP 风格处理器的 C++20 cycle model。它建模
stream-register 时序、逐队列 ICU 控制、两个镜像 MEM hemisphere、四个 MXM、
中心 VXM，以及每个 hemisphere 各一个 SXM。

项目参考公开的 Groq LPU/TSP 资料，但不追求与私有 Groq 硬件或 ISA
bit-accurate。当前目标是提供一个可验证的数据流调度目标，并为后续编译器开发
建立清晰的硬件行为边界。

## 架构概览

| 模块 | 当前模型 |
| --- | --- |
| 向量形态 | 4 个 tile/superlane x 8 lane = 32 个元素 |
| Stream | 32 条 eastward + 32 条 westward，每个寄存器 1 byte |
| MEM | 每个 hemisphere 44 个 slice，全芯片 88 条 ICU queue |
| SRAM | 每个 slice 256 KiB，每侧 11 MiB，全芯片 22 MiB |
| Accumulator | 每侧两个四-slice FP32 accumulator group |
| MXM | 四个 32 x 32 FP16 GEMM 阵列，每侧两个 |
| MXM 权重 | 每个 supercell 两个对等 buffer，由 `IW`/`Compute` 选择 |
| VXM | 中心一个 slice，每个 lane 有 16 个独立控制的 ALU |
| SXM | 每个 hemisphere 一个四-tile Transpose/Permute slice |
| ICU | 88 MEM、4 MXM load、4 MXM compute、16 VXM、4 SXM queue |

固定的完整芯片拓扑为：

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W <-> VXM <-> MEM.E <-> SXM.E <-> MXM0/MXM1
```

每个 hemisphere 使用局部 `sreg0..sreg12`。`sreg0` 靠近 VXM，MEM 的 11 个
group 位于 `sreg0..sreg11`，SXM 将 `sreg11` 连接到 MXM 边界 `sreg12`。

Stream 支持广播读取：同一拍内多个功能单元可以消费同一个寄存器值。只要有一个
消费者，该值就不再被动传播；多个生产者仍不能向同一个 stream register 写入
不同数据。

## 执行模型

整系统 workload 遵循严格的离线控制契约：

1. cycle 0 前把外部输入初始化到 MEM。
2. 离线生成完整指令时间线，并装入所有 ICU queue。
3. 启动时钟后只调用 `TspSliceSystem::tick()`。
4. 结束后读取最终 MEM，并与软件 golden model 比较。

MEM、MXM、VXM 和 SXM 指令必须在正确周期与 stream operand 相遇。每条 queue
通过 `NOP N` 表示延迟，通过 `Repeat n,d` 表示规则指令序列；MEM Repeat 还可以
携带 signed address stride。

## 已验证 Workload

| 测试 | Workload | 验证内容 |
| --- | --- | --- |
| `w8a16_projection_test` | `[128,576] x [576,1536]` W8A16 projection | 196,608 个 FP32 输出 |
| `w8a16_swiglu_test` | gate/up projection + SwiGLU | 196,608 个 FP16 输出 |
| `dual_hemisphere_w8a16_swiglu_test` | 完整 gate/up、SwiGLU、down FFN | `[128,576]` 最终 FP16 输出 |
| `rmsnorm_test` | `[32,32]` FP16 RMSNorm | 全部存回的 FP16 输出 |
| `smollm2_attention_test` | Q/K/V、RoPE、QK、softmax、P x V、`o_proj` | `[128,576]` attention 输出 |
| `sxm_mem_transpose_test` | 连续 MEM -> SXM -> MEM FP16 transpose | 四个 32 x 32 矩阵 |

完整 FFN 使用四个 MXM，当前调度为 90,817 拍。gate/up 的最终 reduction 会把
accumulator 结果直接送入共享 VXM SwiGLU 流水线。SmolLM2 attention 使用
sequence length 128、hidden size 576、9 个 query head、3 个 KV head 和
64 维 head，完整验证调度为 81,273 拍。

## 构建

当前 Windows 配置使用 Visual Studio 2026 Community：

```powershell
cmake -S . -B build-vs2026 `
  -G "Visual Studio 18 2026" `
  -A x64
cmake --build build-vs2026 --config Release
ctest --test-dir build-vs2026 -C Release --output-on-failure
```

使用其他 CMake generator：

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

直接运行两个最大的整系统回归：

```powershell
build-vs2026\Release\dual_hemisphere_w8a16_swiglu_test.exe
build-vs2026\Release\smollm2_attention_test.exe
```

整系统日志默认关闭，因为逐拍 trace 会显著拖慢模拟。小型测试和 demo 可以通过
`TspSliceSystem::LogSinks` 分别输出 ICU、MEM、MXM、VXM、SXM 和 system 日志。

## 调度图

- [W8A16 projection pipeline](docs/w8a16_projection_pipeline.svg)
- [完整 FFN pipeline](docs/w8a16_swiglu_pipeline.svg)
- [完整 FFN 详细 ICU 调度](docs/w8a16_swiglu_schedule_detail.svg)
- [SmolLM2 attention pipeline](docs/smollm2_attention_pipeline.svg)
- [Attention 优化对比](docs/smollm2_attention_pipeline_optimization.svg)
- [Attention 详细 ICU 调度](docs/smollm2_attention_schedule_detail.svg)

重新生成 FFN 调度图：

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\w8a16_swiglu\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\dual_hemisphere_w8a16_swiglu_test.exe
python scripts\render_swiglu_schedule_trace.py `
  logs\w8a16_swiglu\schedule.csv `
  docs\w8a16_swiglu_schedule_detail.svg
```

重新生成 attention 调度图：

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\smollm2_attention\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\smollm2_attention_test.exe
python scripts\render_schedule_trace.py `
  logs\smollm2_attention\schedule.csv `
  docs\smollm2_attention_schedule_detail.svg
```

详细图中，紫色 accumulator 条带表示 partial sum 保留在 SRAM；红色表示最终
`stream+clear`。

## 仓库布局

- `include/ftlpu/core/`：硬件常量、stream、FP16 和 ISA codec。
- `include/ftlpu/mem/`：SRAM、MEM 指令流水线和 accumulator。
- `include/ftlpu/mxm/`：supercell、阵列、控制 slice 和 GEMM datapath。
- `include/ftlpu/vxm/`：ALU、lane、superlane 和中心 VXM slice。
- `include/ftlpu/sxm/`：Shift/Distribute/Transpose/Permute 模型。
- `include/ftlpu/system/`：ICU、stream topology 和完整芯片集成。
- `tests/`：单元测试和离线整系统数值回归。
- `examples/`：用于观察 trace 的小型 demo。
- `scripts/`：调度可视化工具。
- `docs/`：架构说明、优化分析和调度图。

## 更多文档

- [架构说明](docs/architecture.zh-CN.md)
- [English architecture reference](docs/architecture.md)
- [Attention pipeline 优化分析](docs/attention_pipeline_optimization.md)
- [可编辑拓扑图](docs/FTLPU.drawio)
