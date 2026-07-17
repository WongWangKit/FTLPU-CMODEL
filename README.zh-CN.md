# FTLPU-CMODEL

[English](README.md) | [简体中文](README.zh-CN.md)

FTLPU-CMODEL 是一个面向 cycle 的 C++17 模型，用于实验 FTLPU/TSP 风格的数据流：
MEM stream 穿过 stream register；MXM 消费流式 int8 向量完成矩阵乘；VXM 执行
ALU 指令队列；SXM 对 eastward 向量执行 transpose 或 permute；ICU 将指令分发到
各功能模块。

该模型参考公开的 Groq TSP/LPU 描述，但不是 bit-accurate 的 Groq 实现。当前目标
是构建一个实用的编译器与调度实验平台，可以逐 cycle 验证指令、stream 时序和
功能单元之间的数据交接。

## 当前状态

仓库当前建模了：

- `MEM`：44 个 slice column、4 个 tile row、每个 tile 16 lane，每 lane 有
  32 条 east stream 和 32 条 west stream。每个 stream register 宽 1 byte。
  每个 tile-local SRAM 有 2 个 bank，每个 bank 为 4096 x 16 byte。
- `MXM`：east 侧两个 MXM unit。每个 unit 是 4 x 4 个 16 x 16 supercell。
  每个 supercell 有两个对等 weight buffer；`IW` 选择接收右移 weight stream 的
  buffer，`Compute` 同时选择 weight buffer 和 output stream base，执行带 int32
  accumulation 的 64 x 64 int8 GEMM datapath。
- `VXM`：west 侧一个 VXM slice，包含 4 个 superlane/tile。每个 superlane
  有 16 lane，每 lane 有 16 个 ALU issue queue。ALU 输出可向 stream 写入 int8
  或 fp16 byte。
- `SXM`：MEM 与 MXM 之间 east 侧的数据移动 slice。它消费和生成 32 条 east
  stream，支持 `Transpose sg16` 和 64-lane `Permute`；队列没有发射时被动
  forward 数据。
- `ICU`：按 queue 分发指令，支持 `NOP N` 和 `Repeat n,d`，包括 MEM address
  stride。两个 SXM queue 分别发射 Transpose 和 Permute 指令。
- `TspSliceSystem`：固定局部拓扑，VXM 位于 MEM west 侧，SXM 和两个 MXM 位于
  MEM east 侧。Fabric 有 13 个 stream-register column：MEM 边界为
  `sreg0..sreg11`，`sreg12` 是 SXM-to-MXM 边界；west 路径具有相同额外 hop。
- 用于 MEM、MXM、VXM ALU 和 ICU queue command 的紧凑模型 ISA codec。

最大的集成测试建模完整 FFN 路径：

1. 从 MEM 将 gate/up weight 加载到两个 MXM。
2. 从 MEM 向两个 MXM 发送 activation stream。
3. gate/up int32 输出通过 MEM west stream 进入 VXM。
4. 使用 ALU 指令执行 SwiGLU。
5. 将 int8 hidden result 写回 MEM。
6. 将 down-projection weight 加载到两个 MXM。
7. 让 hidden result 流经 MXM。
8. 将两个 int32 partial result 送入 VXM，执行 add + quant。
9. 将最终 int8 result 写回 MEM并与 golden data 比较。

`mem_dual_mxm_swiglu_offline_icu_test` 是标准 FFN 集成测试。所有 MEM、MXM 和
VXM 指令在 cycle 0 之前离线生成并加载到 ICU。MXM output 由 `Compute` 指令流
控制；runtime loop 只推进时钟并桥接数据。这正是未来编译器后端应生成的形态。

`single_head_attention_test` 建模完整 single-head attention datapath。它在 MEM
初始化 `seq_len=32`、`hidden=64` 的输入和 Wq/Wk/Wv 矩阵，将 Wq/Wk 加载到
两个 MXM，使 X 流经两个 MXM，并将 Q/K int32 result 送入 VXM。ALU 通过
`Multiply` + `Cast(Int8)` requantize，随后将 Q/K int8 stream 写回 MEM，再将
Q 加载到 MXM0、发送 K，计算采样的 `K * Q^T` score 并与 golden data 比较。

Raw score 不写入 MEM：MXM west-stream score output 直接进入 VXM softmax pass 1。
该 pass 缩放为 fp32，在 MEM 保存 intermediate，并通过 ALU self-feedback `Max`
计算每个 query row 的 maximum。Pass 2 重新加载 scaled score 和 maximum，计算
`exp(x - max)` 并通过 self-feedback `Add` 累加 row sum。Pass 2/3 将 key position
分到四个独立 MEM group，使用四条 VXM ALU pipeline 并行执行。Pass 2 将四个
8-element partial sum 合并为 32-element row sum。Pass 3 重新加载 exponential
和 sum，执行 divide、scale、`Cast(Int8)`，并保存最终 attention probability。

MXM 每 cycle 输出一个 key position，而 VXM lane 表示 query，因此 reduction 无需
物理 transpose 或 host reduction。Pass 1 仍需 32 个 data cycle；条带化的
Pass 2 和 Pass 3 各需 8 个 data cycle。

X row 跨 16 个 MEM slice 条带存储。MXM1 每 cycle 将 16 行原始 X 作为 weight
column 加载，因此无需物化转置矩阵即可在 buffer 中形成逻辑 `X^T`。Softmax
pass 1 排空后，ICU 在 `E16..E31` 按 column 发送 Wv，MXM1 每 cycle 生成一行
`V^T = Wv^T * X^T`。ALU3..5 requantize 后将 V 条带写回 MEM。最终 phase 直接
用 16 条 weight stream 在 4 cycle 内加载 V，无需对 V 执行 SXM transpose。
随后 SXM 组装 softmax query row，MXM1 计算 `softmax * V`。`32 x 64` int32
result 通过 `W0..W3` 返回四个 MEM slice，并与 golden data 比较。

测试将 ICU dispatch trace 写入
`build-vs2019/logs/single_head_attention/icu.log`。

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
cmake -S . -B build-vs2019
cmake --build build-vs2019 --config Debug
ctest --test-dir build-vs2019 -C Debug --output-on-failure
```

Single-config generator：

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## 常用测试

运行 offline ICU FFN 测试：

```powershell
ctest --test-dir build-vs2019 -C Debug -R mem_dual_mxm_swiglu_offline_icu --output-on-failure
```

运行 VXM 测试：

```powershell
ctest --test-dir build-vs2019 -C Debug -R "vxm_alu|vxm_lane|vxm_superlane|vxm_slice" --output-on-failure
```

运行 single-head attention 测试：

```powershell
ctest --test-dir build-vs2019 -C Debug -R single_head_attention_test --output-on-failure
```

运行 stream-facing SXM `softmax * V` 测试：

```powershell
ctest --test-dir build-vs2019 -C Debug -R sxm_softmax_v_test --output-on-failure
```

该测试只在 MEM SRAM 初始化 softmax 和 V，随后执行 ICU 驱动的 MEM Read、
MXM `IW`、SXM Transpose、非 identity Permute、MXM Compute 和四 byte MEM
Write queue。最终从四个 MEM slice 还原 `32 x 64` int32 matrix，并与 golden
data 比较。ICU trace 位于当前 CTest build 目录下的 `logs/sxm_softmax_v/icu.log`。

## 日志和图示

FFN 集成测试默认不生成日志，避免长时间 workload 被文件 I/O 主导。调试时启用：

```powershell
$env:FTLPU_FFN_LOG = "1"
ctest --test-dir build-vs2019 -C Debug -R mem_dual_mxm_swiglu_offline_icu --output-on-failure
Remove-Item Env:\FTLPU_FFN_LOG
```

启用后，日志写入 build 目录：

- `build-vs2019/logs/mem_mxm/`
- `build-vs2019/logs/mem_dual_mxm_swiglu_offline_icu/`
- `build-vs2019/logs/mem_dual_mxm_swiglu_early_compute_icu/`

FFN 测试生成四个功能单元日志：

- `icu.log`
- `mem.log`
- `mxm.log`
- `vxm.log`

同时生成 pipeline 图：

- `build-vs2019/logs/mem_dual_mxm_swiglu_offline_icu/pipeline.svg`

该图分别展示 `MEM W read`、`MEM A read`、`MEM write`、`MXM0 load`、
`MXM0 compute`、`MXM1 load`、`MXM1 compute` 和 `VXM`。不存在独立 `LW`
阶段；`IW` 填充指定 buffer，`Compute` 指定消费的 buffer 和 output stream base。

## Demo 可执行程序

构建后可在输出目录运行 demo，例如：

```powershell
.\build-vs2019\Debug\tile_array_trace_demo.exe tile_array_trace.log
.\build-vs2019\Debug\vector_roundtrip_demo.exe vector_roundtrip.log
.\build-vs2019\Debug\mxm_control_trace_demo.exe mxm_control_trace.log
.\build-vs2019\Debug\mem_mxm_trace_demo.exe mem_mxm_mem.log mem_mxm_mxm.log
.\build-vs2019\Debug\vxm_lane_trace_demo.exe vxm_lane_trace.log
```

## 更多文档

当前架构、时序模型、指令队列、FFN workload、数据 layout、日志和已知限制见
[docs/architecture.zh-CN.md](docs/architecture.zh-CN.md)。
