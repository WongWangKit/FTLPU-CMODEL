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
| MEM | 每个 hemisphere 52 个 slice，每 slice 两条 bank queue，全芯片 208 条 ICU queue |
| SRAM | 每 slice 两个 256 KiB 单口 bank，每侧 26 MiB，全芯片 52 MiB |
| Accumulator | 每个 MXM 的完整 32x32 FP32 block 数由 JSON 配置，默认 256 blocks / 1 MiB |
| MXM | 四个 32 x 32 FP16 GEMM 阵列，每侧两个 |
| MXM 权重 | 每个 supercell 两个对等 buffer，由 `IW`/`Compute` 选择 |
| MXM decode | 可选择 `Linear1x16` 或 activation-stationary `Native4x4` |
| VXM | 中心一个 slice；每 lane 两条镜像的 8 级链，共 16 个物理 ALU |
| SXM | 每个 hemisphere 一个四-tile Transpose/Permute slice |
| ICU | 208 MEM、12 MXM、8 VXM 紧凑控制、4 SXM、6 C2C/DMA queue |

固定的完整芯片拓扑为：

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W <-> VXM <-> MEM.E <-> SXM.E <-> MXM0/MXM1
```

每个 hemisphere 使用局部 `sreg0..sreg15`。`sreg0` 靠近 VXM，MEM 的 13 个
group 位于 `sreg0..sreg13`，C2C 接在 `sreg13`，SXM 跨接 `sreg14` 到 MXM
边界 `sreg15`。

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
| `smollm2_attention_test` | ICU 驱动的 Q/K RoPE、QK、softmax 和 P x V hardware tile | RoPE 逐 bit 检查及 `8 x 32` attention 输出 |
| `mxm_decode_layout_comparison_test` | 用两种 decode layout 执行相同 `K=128, N=32` GEMV | BF16 逐 bit 一致并比较周期数 |
| `smollm2_decode_ffn_test` | 原生 4 x 4 weight-streaming decode FFN | `[1,576]` 最终 BF16 输出 |
| `sxm_mem_transpose_test` | 连续 MEM -> SXM -> MEM FP16 transpose | 四个 32 x 32 矩阵 |

完整 FFN 使用四个 MXM，当前调度为 90,817 拍。gate/up 的最终 reduction 会把
accumulator 结果直接送入共享 VXM SwiGLU 流水线。standalone attention 可执行文件
严格验证一个 `8-token x 4-head x 8-dimension` 物理 tile；layer-phase harness
再把这一 tile 映射重复到 `[128,576]` API 外形。

MXM Decode 指令带有显式 layout bit。`Linear1x16` 从 8 条 stream 为每个 tile
装入 4 个独立的 8 元素 activation vector，让一个 partial sum 依次走过全部
16 个 supercell。`Native4x4` 从 2 条 BF16 stream 为每个 tile 装入一个 8 元素
vector，并在同一物理行的 4 个 supercell 间广播；阵列并行形成 4 条纵向 reduction
链。32 条 INT8 权重 stream 分成 4 个 8-stream 物理列，第 `c` 列比第 0 列晚
`c` 拍到达 MXM，构成 7 拍对角波前。对比测试用同一个 golden GEMV 锁定两种
实现的数值一致性。当前 SmolLM2 decode FFN 选择 `Native4x4`，decode attention
的 resident layout 暂时明确保留 `Linear1x16`。

## 构建

硬件 target 统一定义在
[`config/ftlpu-lpu32.json`](config/ftlpu-lpu32.json)。CMake 会严格校验该文件，
并为需要固定数组尺寸的 CModel 结构生成编译期常量。可在配置阶段通过
`-DFTLPU_HARDWARE_CONFIG=<target.json>` 选择其他兼容配置。FTLPU-SOFTWARE 使用
同一个 cache 变量和同一份 JSON，因此修改物理几何参数后需要重新配置并构建两个工程。

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
SRAM 几何参数来自硬件 target JSON。默认配置的每个 bank 有 8192 个本地 row，
每 row 32 byte，因此每个 bank 为 256 KiB、每个 MEM slice 为 512 KiB。
后备存储按 4 KiB page 延迟分配。

## 调度图

- [W8A16 projection pipeline](docs/figures/w8a16_projection_pipeline.svg)
- [完整 FFN pipeline](docs/figures/w8a16_swiglu_pipeline.svg)
- [完整 FFN 详细 ICU 调度](docs/figures/w8a16_swiglu_schedule_detail.svg)
- [SmolLM2 attention pipeline](docs/figures/smollm2_attention_pipeline.svg)
- [Attention 优化对比](docs/figures/smollm2_attention_pipeline_optimization.svg)
- [Attention 详细 ICU 调度](docs/figures/smollm2_attention_schedule_detail.svg)

重新生成 FFN 调度图：

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\w8a16_swiglu\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\dual_hemisphere_w8a16_swiglu_test.exe
python scripts\render_swiglu_schedule_trace.py `
  logs\w8a16_swiglu\schedule.csv `
  docs\figures\w8a16_swiglu_schedule_detail.svg
```

重新生成 attention 调度图：

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\smollm2_attention\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\smollm2_attention_test.exe
python scripts\render_schedule_trace.py `
  logs\smollm2_attention\schedule.csv `
  docs\figures\smollm2_attention_schedule_detail.svg
```

详细图中，紫色 accumulator 条带表示 partial sum 保留在 MXM 本地 accumulator；红色表示最终
`stream+clear`。

## 仓库布局

- `include/ftlpu/core/`：硬件常量、stream、FP16 和 ISA codec。
- `include/ftlpu/mem/`：同构 SRAM slice 和 MEM 指令流水线。
- `include/ftlpu/mxm/`：supercell、阵列、控制 slice、GEMM datapath 和 MXM 本地
  accumulator。
- `include/ftlpu/vxm/`：ALU、lane、superlane 和中心 VXM slice。
- `include/ftlpu/sxm/`：Shift/Distribute/Transpose/Permute 模型。
- `include/ftlpu/system/`：ICU、stream topology 和完整芯片集成。
- `tests/`：单元测试和离线整系统数值回归。
- `examples/`：用于观察 trace 的小型 demo。
- `scripts/`：调度可视化工具。
- `docs/`：架构说明、优化分析和调度图。

## 更多文档

- [JSON 硬件配置指南](docs/hardware_configuration.zh-CN.md)
- [架构说明](docs/architecture.zh-CN.md)
- [English architecture reference](docs/architecture.md)
- [MEM ICU N 维流描述符](docs/icu_mem_stream_nd.zh-CN.md)
- [MXM ICU N 维流描述符](docs/icu_mxm_stream_nd.zh-CN.md)
- [VXM_STREAM_ND 与 N 维启动描述符](docs/icu_vxm_stream_nd.zh-CN.md)
- [SXM TILE_PROGRAM](docs/icu_sxm_tile_program.zh-CN.md)
- [Attention pipeline 优化分析](docs/attention_pipeline_optimization.md)
- [可编辑拓扑图](docs/FTLPU.drawio)
