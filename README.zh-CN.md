# FTLPU-CMODEL

[English](README.md) | [简体中文](README.zh-CN.md)

FTLPU-CMODEL 是一个面向 cycle 的 C++20 模型，用于实验 FTLPU/TSP 风格的数据流：
MEM stream 穿过 stream register；MXM 消费流式 FP16 向量完成矩阵乘；VXM 执行
16-ALU lane 程序；SXM 对 eastward 向量执行 transpose 或 permute；ICU 将指令分发到
各功能模块。

该模型参考公开的 Groq TSP/LPU 描述，但不是 bit-accurate 的 Groq 实现。当前目标
是构建一个实用的编译器与调度实验平台，可以逐 cycle 验证指令、stream 时序和
功能单元之间的数据交接。

## 当前状态

仓库当前建模了：

- `MEM`：两个镜像 hemisphere，每侧 44 个 slice column（全芯片共 88 个）、4 个 tile row、每个 tile 8 lane，每 lane 有
  32 条 east stream 和 32 条 west stream。每个 stream register 宽 1 byte。
  每个 tile-local SRAM 有 2 个 bank，每个 bank 为 4096 x 8 byte。最靠近 MXM
  的两个四-slice group 还支持 `Accumulate(address, stream)`：每组四片组成
  FP32，与 stream FP32 相加后原址写回。
- `MXM`：共四个 MXM unit，每个 hemisphere 外侧各两个。每个 unit 是 4 x 4 个 8 x 8 supercell。
  每个 supercell 有两个对等 weight buffer；`IW` 选择接收右移 weight stream 的
  buffer，`Compute` 同时选择 weight buffer 和 output stream base，执行带 FP32
  accumulation 的 32 x 32 FP16 GEMM datapath。INT8 权重先由 VXM 反量化并 cast
  为 FP16，再通过 16 条 byte stream 送入 MXM 的 `IW`。
- `VXM`：中心位置一个 VXM slice，包含 4 个 superlane/tile，每个 superlane 有
  8 lane。每个 lane 有 16 个 ALU 和 16 条独立 ICU queue。操作数可来自
  INT8/FP16/FP32 stream、立即数或前一拍 ALU 输出，结果可保留或写到指定 stream
  与 hemisphere。离线程序把 dequant 展开为 8 组 multiply/cast，把 SwiGLU
  展开为带寄存延迟的乘法、sigmoid 分解和 FP16 cast 流水线。
- `SXM`：每个 hemisphere 的 MEM 与 MXM 之间各有一个数据移动 slice。它消费和生成 32 条 east
  stream，支持 8 x 8 transpose 和 32-lane `Permute`；队列没有发射时被动
  forward 数据。
- `ICU`：按 queue 分发指令，支持 `NOP N` 和 `Repeat n,d`，包括 MEM address
  stride。全芯片有 88 条 MEM queue、4 条 MXM load queue、4 条 MXM compute
  queue、16 条 VXM ALU queue，并为两个 SXM 分别提供 Transpose/Permute queue。
- `TspSliceSystem`：VXM 位于中心，两侧镜像地连接 `MEM <-> SXM <-> 两个 MXM`。
  每侧 fabric 都有 13 个 stream-register column。全局 MXM 0..1、MEM queue 0..43
  属于 East；MXM 2..3、MEM queue 44..87 属于 West。VXM 指令分别指定输入和输出 hemisphere。
- 用于 MEM、MXM、VXM ALU 和 ICU queue command 的紧凑模型 ISA codec。

`w8a16_projection_test` 是当前标准整系统测试。它计算
`[128,576] x [576,1536]`，权重采用按输出列对称 W8 量化。ICU 控制 MEM 将
INT8 权重送到 VXM，VXM 乘 scale 并 cast 为 FP16，再由 `IW` 装入两个 MXM。
activation 由 ICU MEM Read 放到 east stream，连续 `Compute` 指令产生 FP32
部分和，ICU MEM `Accumulate` 指令在 slice36..43 中原址累加 18 个 K tile。
两个 MXM 各自使用一个四片 SRAM read-modify-write group，可以保持并行输出。

`w8a16_swiglu_test` 在此基础上完成
`X[128,576] -> gate/up[128,1536] -> SwiGLU[128,1536]`。两个 MXM 生成 gate/up
partial sum，两个 MEM ACC group 累加全部 18 个 K tile，ICU 再用 MEM Read 将
两路 FP32 operand 向西送入 VXM ALU 流水线，FP16 结果向东写回 MEM，并验证最终
196,608 个元素。

`dual_hemisphere_w8a16_swiglu_test` 是使用相同完整参数的全芯片回归：
`X[128,576]`、gate/up 权重 `[576,1536]`。相邻的 32-column 输出 block 在两个
hemisphere 间交替分配，East MXM0/1 与 West MXM2/3 在相同 compute window 工作。
中心 VXM 交错执行 East/West SwiGLU ALU 程序，并把两半 FP16 结果分别写回本侧 MEM。
测试对照软件 golden 验证全部 accumulator 和 196,608 个输出。

`smollm2_kv_projection_test` 开始实现 SmolLM2 attention 路径。测试把输入视为
已经完成 normalization 的 FP16 `X[128,576]`，对 K/V 权重 `[576,192]` 做按输出
列的 W8 对称量化，并使用四个 MXM 计算 `K/V[128,192]`。相邻 32-column block
交替分配给 East/West，K 和 V 的 FP32 accumulator 分别保存在最靠近 MXM 的两个
MEM group 中，并验证全部 49,152 个结果。RMSNorm、FP32→FP16、RoPE 和 KV cache
布局属于后续阶段。

整系统测试遵守严格的离线契约：cycle 0 前只允许初始化 MEM SRAM，并把完整指令
workload 装入 ICU；时钟启动后只能调用 `tick()`，不能直接
操作 MEM、MXM、VXM 或 SXM。`TspSliceSystem` 因此只公开命名明确的初始化/结果
接口、`icu()`、`tick()` 和 `cycle()`。

## 仓库布局

- `include/ftlpu/core/`：硬件参数、stream word、拓扑 helper、instruction
  pipeline primitive 和指令编码。
- `include/ftlpu/mem/`：MEM slice 和完整 tile-array 模型。
- `include/ftlpu/mxm/`：MXM supercell、array、control slice、wrapper 和系统持有
  的 datapath state。
- `include/ftlpu/vxm/`：VXM ALU、lane、superlane 和 slice 模型。
- `include/ftlpu/sxm/`：SXM shift、distribute、permute、transpose 和集成的
  stream-facing slice 模型。
- `include/ftlpu/system/`：ICU 和 whole-slice system 集成。
- `tests/core/`、`tests/mem/`、`tests/mxm/`、`tests/vxm/`：子系统测试。
- `tests/integration/`：MEM/MXM/VXM/ICU 跨单元测试。
- `examples/`：小型 trace demo。
- `docs/architecture.zh-CN.md`：详细项目说明。

## 构建

Windows Visual Studio generator：

```powershell
cmake -S . -B build-vs2026
cmake --build build-vs2026 --config Release
ctest --test-dir build-vs2026 -C Release --output-on-failure
```

Single-config generator：

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## 常用测试

运行 offline ICU W8A16 projection 测试：

```powershell
ctest --test-dir build-vs2026 -C Release -R w8a16_projection_test --output-on-failure
```

运行完整 W8A16 SwiGLU：

```powershell
ctest --test-dir build-vs2026 -C Release -R w8a16_swiglu_test --output-on-failure
```

运行 VXM 测试：

```powershell
ctest --test-dir build-vs2026 -C Release -R "vxm_alu|vxm_lane|vxm_superlane|vxm_slice" --output-on-failure
```

## 日志和图示

长整系统 workload 默认关闭日志，避免文件 I/O 主导模拟时间。单元级 trace demo
仍保留在 `examples/`。当前离线 projection 调度见
[docs/w8a16_projection_pipeline.svg](docs/w8a16_projection_pipeline.svg)。

## Demo 可执行程序

构建后可在输出目录运行 demo，例如：

```powershell
.\build-vs2026\Release\tile_array_trace_demo.exe tile_array_trace.log
.\build-vs2026\Release\vector_roundtrip_demo.exe vector_roundtrip.log
.\build-vs2026\Release\mxm_control_trace_demo.exe mxm_control_trace.log
.\build-vs2026\Release\mem_mxm_trace_demo.exe mem_mxm_mem.log mem_mxm_mxm.log
```

## 更多文档

当前架构、时序模型、指令队列、离线 workload、数据 layout 和已知限制见
[docs/architecture.zh-CN.md](docs/architecture.zh-CN.md)。
