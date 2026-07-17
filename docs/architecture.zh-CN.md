# FTLPU-CMODEL 架构说明

[English](architecture.md) | [简体中文](architecture.zh-CN.md)

本文描述仓库中当前 C++ 模型的实现，作为后续编译器和 scheduler 工作的实现指南。

## 设计目标

该模型是面向 cycle 的 FTLPU/TSP 类架构实验平台，重点关注三个方面：

- Stream 显式化：数据通过具名 eastward/westward stream 移动，stream register
  宽度为 1 byte。
- 指令显式化：MEM、MXM 和 VXM 由 instruction queue 驱动，而不是直接调用
  单个 tile。
- 时序准确：测试检查指令和数据是否在 consumer 功能单元所需的同一个 cycle
  到达。

本模型不是 bit-accurate 的 Groq LPU/TSP 模型。公开资料只作为架构参考，当前
指令编码和时序规则均为 CModel 自有定义。

## 硬件形态

模型 slice 使用 `include/ftlpu/core/hardware_params.hpp` 中的常量。

当前拓扑：

- 4 个 tile row；
- 44 个 MEM slice column；
- 每个 stream-register group 包含 4 个 MEM slice；
- 11 个 MEM slice group；
- 13 个系统 stream-register column：MEM 覆盖 `sreg0..sreg11`，SXM 位于
  MEM east 边界 `sreg11` 和 MXM 边界 `sreg12` 之间；
- 每个 tile/superlane 16 lane；
- 每 lane 64 条 stream：32 eastward、32 westward；
- 每个 stream register 1 byte；
- 当前建模一个 hemisphere，包含 44 个 MEM/SRAM slice column；
- 公开的双 hemisphere 总计 88 个 MEM/SRAM slice；
- 每个 slice column 有 4 个 tile-local SRAM block；
- 每个 tile-local SRAM block 有两个 bank；
- 每个 bank 为 4096 word x 16 byte；
- 完整 slice 容量为 4 x 2 x 4096 x 16 byte = 512 KiB。

软件可见 MEM 地址遵循公开风格的布局：

```text
[39:24] TSP chip
[23]    hemisphere，East=1，West=0
[22:17] slice number 0..43
[16]    bank
[15:4]  所选 bank 内的 4096-word offset
[3:0]   16-byte SRAM word 内的 byte offset
```

当前模型实例化一个 hemisphere。每个 MEM slice 接收一条 instruction stream；
对于 `Read`/`Write`，编码的硬件地址是从软件地址 `[16:4]` 复制的 13-bit
slice-local word address。`set_sram_byte` 和 `sram_byte` 为测试和初始化保留
byte-level access；MEM `Read`/`Write` 要求 16-byte 对齐，每 cycle 移动一个
完整 SRAM word。

## Stream 方向约定

模型把 stream 视为 lane-local channel：

- `E0..E31`：eastward stream；
- `W0..W31`：westward stream；
- combined stream ID `0..31` 表示 east stream；
- combined stream ID `32..63` 表示 west stream。

MEM 指令的 `stream` 字段使用 `0..63` combined 编号。VXM operand
`StreamInt32(base)` 消费连续四条 byte stream，并按 little-endian 打包成一个
int32 operand。

## MEM 模型

MEM tile-array 模型位于 `include/ftlpu/mem/tile_array.hpp`。

重要行为：

- 有 44 条独立 MEM instruction queue，每个 slice column 一条；
- 指令从 tile 0 进入，每 cycle 向北移动一个 tile；
- 每个 MEM slice column 建模为单 instruction port，同一 cycle 不能同时发射
  `Read` 和 `Write`，需要两者的 schedule 必须安排到不同 cycle；
- `Read(address, stream)` 从 address bit 16 指定的 bank 读取一个对齐的 16-byte
  SRAM word，每 lane 一个 byte，并写入该 MEM slice 相邻 stream register 上的
  目标 stream；
- `Write(address, stream)` 从目标 stream 消费 16 byte，每 lane 一个 byte，并
  写入 address bit 16 指定 bank 的一个对齐 SRAM word；
- instruction enum 中包含 `Gather` 和 `Scatter`，当前测试主要覆盖 `Read` 和
  `Write`。

每个 cycle、每个 tile，一条 MEM 指令可以为一个 stream ID 在 16 lane 上移动
16 byte。

## MXM 模型

MXM 模型位于 `include/ftlpu/mxm/`。

当前组件：

- `MxmSupercell`：一个 16 x 16 int8 weight block；
- `MxmArray`：4 x 4 supercell grid；
- `MxmControlSlice`：`IW` 和 `Compute` 的南到北 control pipeline；
- `Mxm`：包含 array、control slice，以及 activation flow、int32 accumulation
  和 output stream injection datapath state 的 wrapper。

系统在 MEM east 侧包含两个 MXM。

### Weight 加载

每个 supercell 有两个对等 weight buffer。`IW(buffer)` 将一个 16 x 16 weight
block 注入所选 row buffer 的 west 侧。该 row 上每个有效 `IW` cycle 都使所选
buffer 向 east 移动一 column，新 block 进入 column 0。为了最终按 column 0..3
得到预期顺序，MEM 以反序读取 weight column block：先读 3，再读 2，直到 0。
不存在单独的 `LW` commit 指令；`Compute(buffer, activation_stream,
output_stream)` 直接选择为当前 activation token 提供 weight 的 buffer。

`IW` 所需 weight byte 从 MEM 到达 east handoff stream register。每个 MXM 每
cycle 使用 16 条 stream：

- MXM0 消费 `E0..E15`；
- MXM1 消费 `E16..E31`。

完整 64 x 64 weight matrix 包含 4 个 column block。当前测试向一个 buffer
连续发射 4 个 `IW` pulse，同时另一个 buffer 仍可供 in-flight compute 使用。
两个 weight buffer 有独立的 row shift state，因此加载一个 buffer 不会扰动另一个。

### Compute

`Compute(buffer, activation_stream, output_stream)` 是 one-cycle pulse。ICU 在
有效 compute window 的每个 cycle 发射一条 `Compute`。选定的 buffer id 和
output stream base 随 activation event 在 MXM 中移动，后续 `IW` 因而可以覆盖
另一个 buffer，而不改变 in-flight work。

系统持有的 MXM runtime 对 activation flow 建模：

- activation 从 MEM 进入 tile row，具有 one-cycle 的南到北 skew；
- 每个有效 supercell 消费一个 16-byte activation vector；
- supercell 计算该向量与 16 个本地 weight column 的 16 个 dot product；
- activation 向 east 穿过 supercell column；
- partial sum 累加成 int32 result column。

Runtime 由 `TspSliceSystem` 持有。ICU `Compute` pulse 通过 MXM datapath 消费
MEM east handoff stream。当 tile row 3 的 contribution 完成一个 result column
block 时，MXM 自动将 int32 result byte 注入 `Compute` 指令指定的 MEM west
stream。

## VXM 模型

VXM 模型位于 `include/ftlpu/vxm/`。

层级：

- `VxmAlu`：支持 ALU op 的 vector helper；
- `VxmLane`：一个 lane，包含 16 个 ALU，每个 ALU 一条 instruction queue；
- `VxmSuperlane`：16 lane；
- `VxmSlice`：4 个 superlane/tile，指令南到北流动。

当前支持的 ALU opcode：

- `Pass`
- `Add`
- `Subtract`
- `Multiply`
- `Divide`
- `Negate`
- `Abs`
- `Min`
- `Max`
- `Clamp`
- `Square`
- `Sqrt`
- `Exp`
- `Log`
- `Relu`
- `Cast`

Quantization/dequantization 不是特殊 ALU opcode，而由 primitive ALU op 表达：

- 将 int32 stream data cast 为 fp32；
- 乘 scale 进行 dequantize；
- 乘输出 scale 的倒数；
- 必要时加 zero point；
- cast 为 int8。

VXM 也可将 fp32 cast 为 fp16 stream output。Fp16 路径将 IEEE half-precision
bit pattern 按 little-endian 写到两条 byte stream；旧 int8 路径仍写一条 stream。

Stream output 与 opcode 无关。任意 ALU instruction 都可以指定 output stream，
根据 cast target 将 result 序列化为 int8、fp16 或 fp32。如果 arithmetic result
已经是 fp32 且只需进入 stream，就不需要额外 `Cast(Float32)`。

SwiGLU 由 primitive op 组成：

```text
gate_i32 -> cast fp32 -> multiply gate_scale
up_i32   -> cast fp32 -> multiply up_scale
sigmoid(gate) = 1 / (1 + exp(-gate))
hidden = gate * sigmoid(gate) * up
hidden_i8 = cast_int8(hidden / output_scale + zero_point)
```

## ICU 模型

ICU 实现在 `include/ftlpu/system/icu.hpp`。

Queue 数量：

- 44 条 MEM queue，每个 MEM slice column 一条；
- 2 条 MXM load queue，每个 MXM 一条；
- 2 条 MXM compute queue，每个 MXM 一条；
- 2 条 SXM queue：Transpose 和 Permute 各一条；
- 16 条 VXM queue，每个 ALU 一条。

ICU 每条 queue 支持：

- `Instruction`：分发一条 MEM/MXM/VXM 指令；
- `NOP N`：仅将当前 queue 延迟 `N` cycle；
- `Repeat n,d`：将上一条指令重复 `n` 次，每次 dispatch 间隔 `d` cycle；
- MEM `Repeat` 额外支持 address stride。

这对编译器很重要：编译器可以把每条 queue 构造成静态时间线，并用 NOP/Repeat
压缩长而规则的 schedule。

## 指令编码

模型 ISA encoder 位于 `include/ftlpu/core/instruction_codec.hpp`。

当前编码：

- MEM instruction：32 bit；
- MXM control instruction：32 bit；
- VXM ALU instruction：3 x 32-bit word；
- ICU queue command：32 bit。

MEM instruction address field 不是完整软件地址，只编码 slice-local SRAM word
address：bit 12 是 bank，bit 11:0 是 4096-word offset。`Read`/`Write` 要求
软件地址低位 byte offset 为 0。

这是紧凑的 FTLPU CModel 编码，不是 Groq 硬件二进制编码。它有意拒绝 VXM
operand scale 和 output zero point 等 model-only metadata，因为这些内容应由
显式 ALU instruction 合成。

Codec 由 `tests/core/instruction_codec_test.cpp` 覆盖。

## 整体系统拓扑

`TspSliceSystem` 固定当前局部拓扑：

```text
VXM <-> sreg0 ... 44 MEM slices ... sreg11 <-> SXM <-> sreg12 <-> MXM0/MXM1
```

重要路径：

- MEM east stream 穿过 SXM，再进入 MXM weight/activation input；
- SXM 只接收 `E0..E31`。没有 SXM 指令发射时，east data 以普通 one-cycle
  register hop 从 `sreg11` 移动到 `sreg12`；
- 对称的 west register hop 将 MXM output 从 `sreg12` 移到 `sreg11`，SXM
  不转换 west stream；
- MXM int32 output 被写入 `sreg12` 的 west stream；
- west stream 穿过 MEM 到达 west edge；
- VXM 根据 ALU operand 消费选中的 stream；
- VXM output stream 被重新注入 MEM，可写入 SRAM。

System tick 处理 ICU dispatch、MEM tick、SXM evaluation、VXM stream bridge 和
MXM control slice。MXM compute/output 由 `TspSliceSystem` 内部的 `Mxm` datapath
持有；测试不再使用独立 software GEMM engine。

## FFN 集成测试

主要集成测试文件：

```text
tests/integration/mem_dual_mxm_swiglu_test.cpp
```

它构造 32 x 64 activation matrix 和三个 128 x 64 weight matrix：

- gate projection weight；
- up projection weight；
- down projection weight。

测试计算：

```text
hidden = swiglu(A * W_gate, A * W_up)
out = quant((hidden_left * W_down_left) + (hidden_right * W_down_right))
```

Shape：

- Activation：`32 x 64`，int8；
- Gate weight：等效 `64 x 128`，分两个 64-column pass 加载；
- Up weight：等效 `64 x 128`，分两个 64-column pass 加载；
- Hidden：`32 x 128`，int8；
- Down weight：`128 x 64`，拆分到两个 MXM；
- Final output：`32 x 64`，int8。

生成数据使用非平凡 symmetric quantization scale，避免输出大多为零。测试将 SRAM
内容与测试内计算的 golden reference 比较。

### Offline ICU 测试

`mem_dual_mxm_swiglu_offline_icu_test` 使用同一源文件，并启用
`FTLPU_OFFLINE_ICU_FFN_TEST`。

该测试在 cycle 0 前构建 `OfflineIcuProgram`，其中包含全部 MEM、MXM 和 VXM
ALU instruction。MXM result stream placement 编码在 `Compute` 指令中。程序将
指令一次性加载到 ICU queue 后开始 tick。

这是未来编译器应生成的形态：

```text
高层 workload -> compiler/scheduler -> OfflineIcuProgram -> ICU queue
```

### Attention Softmax

`single_head_attention_test` 通过三个 ICU-scheduled VXM pass 实现稳定 softmax。
MXM score output 直接通过 MEM stream register 向 west 进入 VXM，不先写 SRAM。
每条 VXM lane 表示一个 query，连续 cycle 表示 key position，因此无需物理
transpose 即可沿时间维 reduction：

```text
pass 1: Cast -> Multiply(output fp32) -> Max(self feedback, output final max)
pass 2: 4 x [Subtract(max) -> Exp(output fp32) -> Add(self feedback)]
pass 3: 4 x [Divide(sum) -> Multiply(127) -> Cast(Int8)]
```

第一条 `Max` 以 negative infinity 为 seed，每个并行 group 的第一条 `Add` 以 0
为 seed。Pass 1 reduction 全部 32 个 key position。Pass 2/3 按 `key % 4`
将 key 分到四组互不相交的四-slice MEM group 和四条 VXM pipeline。每个 pass-2
feedback `Add` reduction 8 个 value，再用三条 `Add` 合并四个 partial sum。
保存的 row maximum 和 row sum 每个 pass 只读一次，并保持在 ALU output 中复用。
Scaled score、exponential、final maximum 和 final sum 均由 ICU `Read`/`Write`
在 MEM 中移动。测试侧代码只读取完成后的 SRAM 并计算 post-run golden value，
不会向 VXM 注入 maximum 或 sum。

X 以 row-major 形式跨 16 个 MEM slice 按 row 条带存储。MXM1 每 cycle 读取 16
行 X 作为 16 条 weight stream，使原始 X row 成为 MXM output column；buffer
因此在不执行 SXM transpose 的情况下形成逻辑 `X^T`。Softmax pass 1 及其 stream
排空后，ICU 在 `E16..E31` 按 column 读取 Wv。MXM1 每 cycle 计算一行
`V^T = Wv^T * X^T`。ALU3..5 将结果 requantize 为 int8，并跨 16 个 MEM slice
条带保存。测试验证 Q、K、V、QK score、softmax maximum/sum 和最终 softmax byte。

最终 attention phase 也由 ICU 调度。MEM 并行读取 16 条带化 `V^T` row，对每个
hidden-column block 直接在 `E16..E31` 发射。由于 MXM `IW` 每次把新 block
移入 column 0，这些 block 按 hidden-block 反序读取，4 cycle 后 MXM1 buffer 0
被 V 完整替换，无需对 V 执行 SXM Transpose 或 Permute。

MEM 随后在每个 16-key block 内按反序 lane 读取四个条带化 softmax column。
SXM Transpose+Permute 恢复 key order，每 cycle 发射一条 zero-padded 64-byte
query row。MXM1 计算 32 行 `softmax * V`；`W0..W3` 将 little-endian int32
output 送回四个 MEM slice。测试将完整 MXM weight buffer、所有 MXM output
block 和最终 `32 x 64` SRAM matrix 与 golden data 比较。

`sxm_attention_transpose_test` 验证该 GEMM 前所需的 block algorithm。它将 16
个 key column 对齐到 16 条 stream，对 `16 x 16` block 的 `2 x 2` grid 执行
4 条真实 `Transpose sg16`，暂存生成的 row chunk，再用非 identity
`Permute320` map 将每个 32-element query row 组装成 zero-padded 64-lane
MXM activation。测试检查全部 1,024 个 value 和每 row 的 32 个 padding lane。

`sxm_softmax_v_test` 验证 `seq_len=32` 的完整 stream-register 路径。测试代码
只初始化 SRAM 和 ICU queue。ICU MEM Read 通过 `E16..E31` 移动 V，MXM1 同时
接收连续 4 个 `IW` pulse；另一条 MEM Read queue 在 `E1` 移动 softmax column。
由于 MEM instruction 向北传播，SXM 为每个 tile 独立收集 16 个 phase；Transpose
连续发射 `32 + 4 - 1` cycle，覆盖完整 diagonal wavefront。每个局部
`16 x 16` block 仍为 16-cycle latency。

连续 Permute 先捕获本地 Transpose segment，再组装每个 query 对应的两个 key
block。MEM reader 有意反序读取 block 内 16 个 key，非 identity lane-reversal
`Permute320` 恢复 canonical key order。收集完整 wavefront 后，Permute 按 MXM
Compute 所需 tile skew 每 cycle 发射一条 zero-padded 64-byte query row。

MXM1 消费连续 32 行 transposed row，通过 `W0..W3` 输出 int32 result。四条
MEM Write queue 将四个 little-endian result byte 存入相邻 MEM slice。测试从
SRAM 还原所有 result，与 scalar golden model 比较完整 `32 x 64` matrix，
并在 writeback 前检查每个 MXM output block。

当前 offline test 只加载 ICU queue、初始化外部 MEM 内容、tick 系统 datapath，
最后检查 MEM 内容。

测试还覆盖 MXM 双 buffer 路径。`IW(buffer)` 通过每 row 的右移路径填充指定
buffer，同时 `Compute(buffer, activation_stream, output_stream)` 可继续使用
另一个 buffer。MXM control 分开 load 和 compute queue，使 scheduler 可以重叠
下一组 weight transfer 与当前 compute。第二个 gate/up pass 和 down pass 使用
ping-pong schedule：

- 前一个 GEMM 启动时 MXM0 立即开始 `IW`。最初几行的共享 activation traffic
  使用替代 stream ID，避免与 `E0..E15` 冲突；
- MXM1 weight transfer 等 activation traffic 离开 `E16..E31` 后启动。实际
  `IW` payload 只需 4 cycle；其余间隔是 stream scheduling constraint，而不是
  MXM array size constraint；
- Early-compute 变体在 compute queue 可用时立即启动第二个 gate/up GEMM。
  Down weight 分两段加载：stream conflict 清除后 MXM1 可提前开始，MXM0 则等待
  `E0` activation stream 不再使用。由于 MEM 为 single-port，down GEMM 仍需等
  第二次 SwiGLU writeback 完成后，才能从同一个 MEM slice 读取 hidden activation；
- `Compute` 指定消费的 buffer 和 int32 result 的 MEM west stream base；不发射
  `LW`、active-weight commit 或独立 MXM output instruction。

`mxm.log` 包含每 phase 和总计的 MXM performance counter：

```text
offline_gate_up_p0_mxm0 perf cycles=32 active_cycles=32 ...
offline_total_mxm0 perf cycles=... active_cycles=96 ...
```

每 phase utilization 使用 tile0 compute window。Total utilization 同样使用 tile0
scheduling time，并包含 weight-load、writeback 和 wait cycle，因此修改 scheduler
时应重点观察该数值。

## 日志

FFN 集成测试默认不生成日志。需要 trace 或 pipeline diagram 时设置
`FTLPU_FFN_LOG=1`，日志会写到 build 目录。

Offline ICU FFN：

```text
build-vs2019/logs/mem_dual_mxm_swiglu_offline_icu/
```

目录内容：

- `icu.log`：queue depth 和已 dispatch 指令；
- `mem.log`：MEM stream/register/SRAM activity；
- `mxm.log`：MXM load/compute state；
- `vxm.log`：tile 0 的 VXM ALU state；
- `pipeline.svg`：分别显示 MEM weight-read、MEM activation-read、MEM write、
  MXM load、MXM compute 和 VXM row 的 phase timeline。

Offline ICU log 会打印关键 phase start，例如：

```text
offline ICU FFN program loaded before cycle 0
  schedule=baseline ...
  load0.mxm0_iw=... gemm0=... gemm0_output=...
```

Early-compute 变体打印按当前 array size 计算的 schedule：

```text
offline ICU FFN program loaded before cycle 0
  schedule=early_mxm_compute p1 compute starts as soon as the compute queue is free
  load0.mxm0_iw=... gemm0=... gemm0_output=...
```

## 数据 Layout 汇总

FFN 测试在 cycle 0 之前直接把数据放入 SRAM。此后单元间的数据移动完全通过
ICU 控制的 MEM Read/Write 和 stream 完成。

Activation：

- 存储在 MEM column `32`；
- row `r`、lane `l` 的地址为 `r * 16 + l`；
- tile 选择 K 维的 16-element block。

Weight：

- 分布在 MEM column `0..31`；
- MXM0 从 column `0..15` 读取 weight stream；
- MXM1 从 column `16..31` 读取 weight stream；
- 每个 pass 反序读取 4 个 column block，因为 `IW` 会将每行所选 weight buffer
  向 east 移动。

SwiGLU hidden output：

- Pass 0 存在 MEM column `40`；
- Pass 1 存在 MEM column `41`。

Final output：

- 存在 MEM column `42`。

## 已知限制

- 模型不与任何私有 Groq ISA bit-accurate；
- `Gather` 和 `Scatter` 不是当前测试重点；
- VXM op 列表有意保持较小，只包含当前 FFN/SwiGLU 实验和简单 scalar primitive
  所需 op；
- CModel 仓库本身尚不包含编译器，`OfflineIcuProgram` 是编译器输出容器的原型。

## 建议的下一步

近期工程工作：

- 根据更细致的硬件 pipeline 假设收紧 MXM runtime timing；
- 将 `OfflineIcuProgram` 从测试局部 helper 提取成可复用 module；
- 使用 ISA codec 增加文本或二进制 program dump/load 格式；
- 增加简单 compiler pass，为 FFN workload 生成 MEM Read/Write、MXM IW/Compute
  和 VXM ALU timeline；
- 为 stream-register 和 queue collision 增加资源冲突诊断。
