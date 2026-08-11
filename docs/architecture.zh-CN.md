# FTLPU-CMODEL 架构说明

本文描述 `4x8_fp16` 分支的当前实现。公开 Groq LPU/TSP 资料只作为架构参考；
本文中的字段宽度、时序规则和功能行为均属于本 CModel。

## 1. 当前配置

中心向量形态为：

```text
4 个 tile/superlane x 8 lane = 32 个元素
```

| 属性 | 当前值 |
| --- | ---: |
| Hemisphere | 2 |
| MEM slice | 每侧 52 个，全芯片 104 个 |
| MEM group | 每侧 13 个，每组 4 个 slice |
| Stream-register column | 每侧 15 个（`sreg0..sreg14`） |
| 每 lane stream | 32 条 eastward + 32 条 westward |
| Stream-register 位宽 | 1 byte |
| SRAM 容量 | 每 slice 2 MiB，全芯片 208 MiB |
| MXM | 共 4 个，每侧 2 个 |
| MXM 阵列 | 32 x 32 FP16/BF16 乘法、FP32 累加 |
| VXM | 中心 1 个 slice，每 lane 16 个 ALU |
| SXM | 每侧 1 个四-tile slice |

一个 MEM slice 拥有一个 `65536 x 32-byte` SRAM block。每个 row 横跨 4 个
tile；指令波到达某个 tile 时，该 tile 只访问自己的 8-byte segment。两个
hemisphere 共 104 个同构 slice，总容量为 `104 x 2 MiB = 208 MiB`，不再划分
logical bank。

## 2. 完整芯片拓扑

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W(52) <-> VXM <-> MEM.E(52) <-> SXM.E <-> MXM0/MXM1
```

两个 hemisphere 使用相同的局部朝向：

- `sreg0` 靠近 VXM；
- 13 个 MEM group 位于 `sreg0..sreg13`；
- SXM 把 MEM 边界 `sreg13` 连接到 MXM 边界 `sreg14`；
- east stream 从 VXM 流向 MXM；
- west stream 从 MXM 流向 VXM。

全局 MEM queue `0..51`、MXM `0..1` 属于 East；MEM queue `52..103`、MXM
`2..3` 属于 West。

共享 stream fabric 采用逐拍 current/next state：功能单元读取当前状态并暂存输出，
整系统最后统一 commit。当前拍写入的数据到下一拍才可见。

### 广播消费

同一拍内，多个功能单元可以消费同一个物理 stream-register cell，并观察到相同的
数据和 vector tag。消费是幂等的，只表示“至少有一个消费者”：

- 任一消费者都会阻止该值继续被动传播；
- 多个读取者合法；
- 多个生产者向同一个 next-state cell 写不同数据仍然非法。

广播允许本地两个 MXM 共用一对 activation stream，但不能让权重和 activation
占用相同 stream ID。

## 3. 时钟与控制流

MEM、MXM control 和 SXM Transpose 指令从南侧 tile0 进入，每拍向北推进一个
tile。workload 必须在每个 tile 对齐数据和控制，测试不能直接操作运行中的某个 tile。

一次 `TspSliceSystem::tick()` 完成：

1. ICU 从每条 queue 分发下一条 command；
2. MEM、SXM、VXM、MXM 读取当前 stream-register state；
3. 功能单元消费 operand，并暂存输出；
4. 未被消费的数据暂存 passive link；
5. 共享 stream fabric commit next state。

整系统测试在 cycle 0 前初始化输入并装入 ICU queue。时钟开始后只调用 `tick()`，
直到离线 schedule 完成。

## 4. MEM

### 组织方式

每个 hemisphere 有 52 个 MEM slice column，每个 slice 一条指令队列。相邻四个
slice 组成一个 group，位于两个 stream-register boundary 之间。全部 52 个 slice
都是同构 SRAM，不再有专用 accumulator group。

每个 slice 是单端口：同一拍即使地址不同，也不能同时 Read 和 Write。

### 指令

- `Read(address, stream)` 读取 tile-local 8-byte SRAM segment，写入一条 stream；
- `Write(address, stream)` 消费一条 8-byte stream segment 并写入 SRAM；
- `Gather`/`Scatter` 已编码，但因为尚未实现 address-stream datapath，执行时会拒绝。

一条指令波最终经过四个 tile，因此完整指令波以四个错拍的 8-byte segment 搬运
一个 32-byte 物理向量 row。

### 地址

参考的公开风格软件地址为：

```text
[39:27] chip
[26]    hemisphere
[25:20] slice
[19:4]  slice 内 row offset
[3:0]   软件 byte offset
```

`MemInstruction::address` 只保存对应软件位 `[19:4]` 的 16-bit slice-local row，
范围为 `0..65535`。测试的初始化/结果 API 另外指定 tile 和 lane byte。

## 5. MXM

每个 `Mxm` 包含：

- `4 x 4` supercell 阵列；
- 每个 supercell 一个 `8 x 8` 原始 16-bit weight block；
- 每个 supercell 两个对等 weight buffer；
- 从南向北传播的 control slice；
- activation flow 和 FP32 output state。

### 权重装载

`IW(buffer, column)` 把一个显式 supercell-column block 写入指定 buffer。
两位 `column` 字段选择 `0..3`，装载顺序不会隐式改变最终 layout。不存在 `LW`。

默认一个 IW pulse 消费 8 条 INT8 east stream：

```text
每个 lane 8 个 INT8 值 x 每值 1 条 byte stream = 8 条 stream
```

每个 MXM 新增一条独立 Dequant 指令队列，由该队列提供 BF16 scale immediate。
Dequant 与对应 IW 同拍发射并从南到北同步传播；每个 tile 在写 weight buffer 前执行
`BF16(INT8 * BF16(scale))`。Dequant pulse 缺失或与 IW 错拍都会报执行错误。

本地 MXM0 保留 `E0..E15`，MXM1 保留 `E16..E31`；默认 INT8 load 使用各自半区的前
8 条 stream。连续四个 IW pulse 填满一个 32-column weight tile。`IWColumn` 默认只
消费 1 条 INT8 stream；Direct16 兼容模式消费 2 条 stream。向空闲 buffer 执行 IW
时，另一个 buffer 仍可支持在途 Compute。

W8 权重可以采用按输出列的对称 scale：

```text
scale[n] = max_k(abs(W[k,n])) / 127
```

严格按输出列缩放时使用 8 组 `IWColumn`/Dequant；一次完整 `8 x 8` IW 会把一个
scale immediate 广播到本拍装入的 8 列。`IWDirect16` 和 `IWColumnDirect16` 保留
直接写入 FP16/BF16 原始位的兼容路径。

### Compute

`Compute(buffer, activation_stream_base, output_stream_base, ..., data_format,
compute_mode)` 是一拍 control pulse。buffer、stream base、FP16/BF16 格式和 compute
mode 会随 activation 波传播。

| 模式 | 每个 tile 的 activation 输入 | 每个 MXM pulse 的结果 | 每个 supercell 的 MAC |
|---|---:|---:|---:|
| `Vector` | `1 x 8`，2 条 byte stream | `1 x 32` | 64 |
| `Block8` | `8 x 8`，16 条 byte stream | `8 x 32` | 512 |

两种模式都按 Compute 指定的格式解释原始 weight bits。`Vector` 保持原行为：连续
pulse 注入连续输出行，Stream destination 通过四条 west byte stream 输出一行
FP32。

`Block8` 把每对 stream 解释成一行 activation、把 lane 解释成 K 元素。一拍计算八行
输出，row counter 前进 8，并选择一个逻辑 `8 x 32 FP32` accumulator 宽行；每个
supercell column 更新该宽行地址中的一个 `8 x 8` segment。Block8 只允许 accumulator
destination。Block8 `AccumulatorRead` 每拍使用全部 32 条 west byte stream 输出一个
`8 x 8` 列分块，四拍读完一个 `8 x 32` 宽行。连续四拍 Block8 构成一个 `32 x 32`
output tile。

activation 向东穿过四个 supercell column，partial sum 向北传播并按 FP32 累加。
不存在独立 MXM output 指令，也不存在软件输出队列。

全部 MXM runtime state 由整系统持有；整系统测试不使用独立 GEMM engine 或 runtime
helper。

## 6. VXM

中心 VXM 有四个 superlane、每个 superlane 8 lane、每个 lane 16 个 ALU。每个 ALU
有独立 ICU queue，同一条 ALU 指令会作用到所有物理 lane。

支持的 opcode：

```text
Pass Add Subtract Multiply Divide Negate Abs Min Max Clamp
Square Sqrt Exp Log Relu Cast
```

operand 可以来自 INT8、FP16、BF16、INT32、FP32 stream，FP32 immediate，或前一拍
ALU output。结果可以保留在 ALU register，也可以写入指定 stream 和目标 hemisphere。
Cast 可以把 FP16 或 BF16 按 little-endian 拆成两条 byte stream 输出。

量化/反量化不是独立 opcode，而是 ALU 指令图。例如 W8 dequant 由 Multiply 和 Cast
组成；SwiGLU 由算术、Exp、用于流水延迟的 Pass 和 Cast 组成。RMSNorm 使用 ALU
feedback 计算 `sum(x^2)`，并在 `x`、`gamma` 流过时保留 inverse RMS。

## 7. SXM

系统包含两个独立 SXM，每个 hemisphere 一个。SXM 只转换 east stream；west stream
只经过对称寄存器 hop。

每个 SXM 有四个 tile row。Transpose 指令每拍从南向北推进一个 tile，使每个 tile
捕获与自身匹配的对角波前。FP16 low/high byte 作为两个 plane，tile-local Transpose
完成一个 `8 x 8` block 的行列交换。Transpose 和 Permute 只有一种物理宽度：
16 条 byte stream 输入、16 条 byte stream 输出，一拍携带全部 8 行 FP16 数据；
不再支持原来的 2-stream 串行模式。

Transpose 输出先打一拍，再由 Permute 消费。Permute 在 4 个 superlane、32 个 lane
之间重排完整 block。当前实现使用一个 transpose buffer；相同 destination 的 block
可按 `II=4` 流水。

每个 hemisphere 有两条 SXM ICU queue：Transpose 和 Permute 各一条。没有 SXM
指令时，east stream 以普通一拍 link 从 `sreg13` pass 到 `sreg14`。

## 8. ICU 与 ISA

ICU 共拥有 136 条独立 queue：

| Queue 类别 | 数量 |
| --- | ---: |
| MEM | 104 |
| MXM load | 4 |
| MXM Dequant | 4 |
| MXM compute | 4 |
| VXM ALU | 16 |
| SXM Transpose/Permute | 4 |

已实现的 queue command：

- `NOP N`：该 queue 延迟 N 拍；
- `Repeat n,d`：以间隔 d 重复上一条指令 n 次；
- MEM Repeat 可附带 signed address stride。

`Sync`、`Notify`、`Ifetch` 和低功耗配置尚未实现。

当前紧凑 codec 覆盖：

- 32-bit MEM 指令；
- 32-bit MXM control 指令；
- 16-bit MXM Dequant BF16 scale immediate；
- 四个 32-bit word 的 VXM ALU 指令；
- 32-bit ICU NOP/Repeat command。

SXM 指令使用固定的 13 x 32-bit packet 编码，包含 header、16 个输入/输出
stream selector、tile 内 lane map 和完整跨 tile permute map。字段范围和保留位由
`tests/core/instruction_codec_test.cpp` 验证。

## 9. 调度模式

### W8A16 权重与 Activation 共存

原始 INT8 权重从 MEM 沿 east stream 直接进入 MXM 本地反量化模块；scale 由独立
Dequant queue 提供，默认权重装载路径不再使用 VXM。

| 活跃操作 | INT8 IW stream | 空闲半区 |
| --- | --- | --- |
| 装载本地 MXM0 | `E0..E7` | `E16..E31` |
| 装载本地 MXM1 | `E16..E23` | `E0..E15` |
| 没有 IW | 无 | 通常为 `E0..E1` |

两个本地 MXM 可以同拍装载，总计使用 16 条 east stream。activation 可以使用未占用
范围，但仍需遵守 stream collision 规则。Direct16 兼容 load 仍会占满一个 16-stream
半区。

### 权重乒乓

projection reduction 中，Compute 使用 buffer `k mod 2`，MEM、Dequant 和 IW 同时把
reduction `k+1` 准备到另一个 buffer。SRAM slice、stream ID、MXM Dequant queue 和
load queue 都是显式调度资源。

### Accumulator 生命周期

每个 MXM 内置两个按模式区分的 FP32 accumulator SRAM：

| 模式 | SRAM 组织 | 行宽 | 容量 |
|---|---:|---:|---:|
| `Vector` | `(block_count * 32) x 32 FP32` | 128 bytes | 32 blocks 时 128 KiB |
| `Block8` | `(block_count * 4) x (8 x 32 FP32)` | 1024 bytes | 32 blocks 时 128 KiB |

`FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT` 是 CMake cache 参数，默认容量为 32 个完整
32x32 部分和 block。Vector Compute 每次写窄行中的一个 8 列 segment；
Block8 Compute 同时写相同 8 列上的 8 个输出行，四个列 segment 共用一个逻辑宽行
地址。Compute/AccumulatorRead 的 mode bit 选择目标 SRAM。Block8 read 会占满 32
条 west stream，因此输出窗口需要与本地另一个 MXM 协同调度。只有最终结果输出并
清零后，地址才可复用。

### 单端口 MEM

地址不同也不能消除 slice 冲突。只要 Read、Write 占用同一物理 slice，其时间窗口
就必须错开。

## 10. 已验证整系统 Workload

### W8A16 Projection

`w8a16_projection_test` 计算：

```text
A[128,576] fp16 x W[576,1536] int8 -> C[128,1536] fp32
```

权重采用按输出列的 W8 对称 scale。VXM 反量化权重，MXM0/1 计算相邻 output block，
两个 MXM 本地 accumulator 累加 18 个 K tile。全部 196,608 个输出与考虑 FP16
舍入的 scalar golden model 比较。

### 完整 FFN

`dual_hemisphere_w8a16_swiglu_test` 计算：

```text
X[128,576]
  -> gate/up[128,1536]
  -> SwiGLU[128,1536]
  -> down[128,576]
```

四个 MXM 全部参与。gate/up 的非最终 reduction 中，两个 hemisphere 并行工作；
最终 reduction 中，East/West 的 32-row block 交替进入唯一的共享 VXM SwiGLU
流水线，因此每个交替 block 都会让另一侧 MXM 空闲。这是明确的吞吐取舍：它删除了
独立 accumulator readback/SwiGLU 阶段，使验证调度从 93,642 拍降到 90,817 拍。

SwiGLU 结果在两个 MEM hemisphere 都保存一份。down projection 读取本地副本，
使用全部四个 MXM，累加 48 个 K tile，把最终和 cast 为 FP16，并验证全部 73,728
个输出值。

### Block8 + MXM Dequant FFN

`smollm2_block8_dequant_ffn_test` 使用相同的 SmolLM2 参数：

```text
X[128,576] BF16
  -> gate/up[128,1536]
  -> BF16 SwiGLU[128,1536]
  -> down[128,576] FP32
```

权重以 8 条 INT8 stream 进入每个 MXM；独立 Dequant queue 为每个 8 列 group 提供
一个 BF16 scale，转换后的 BF16 权重直接写入 weight buffer。gate、up 和 down 全部
使用 Block8 Compute 与可配置的 `(block_count * 4) x (8 x 32 FP32)` 宽 accumulator。

连续 4 个 Block8 pulse 覆盖一个物理 32-row tile。128 行输入分成 4 组调度，组间只
保留最后一个 column wave 离开内部 partial-sum row 所需的间隔。gate/up accumulator
结果逐位检查，并在显式测试 phase boundary materialize 为 FP32 VXM 输入。六级 VXM
SwiGLU 流水每拍接收一个 32-element vector，将结果 cast 为 BF16，并把 16-stream
packed activation layout 同时写入两个半球。down 直接消费该 VXM 输出并完成全部
48 次 reduction。down 使用带尺度的 FP32 容差，因为硬件按 K tile partial sum
累加，与标量 golden 的线性结合顺序不同。

accumulator 状态直接编码在所属的 `MXM.*.Compute` event 上，不再单独画成功能单元
lane。非末次 reduction 用紫色表示 partial sum 留在 SRAM；末次 reduction 用深紫色
表示结果已经 ready、但仍保留在 SRAM。红色只表示 Compute 或 AccumulatorRead 显式
送到 stream 并清零 slot。

生成的调度文件为
[`smollm2_block8_dequant_ffn_schedule.csv`](traces/smollm2_block8_dequant_ffn_schedule.csv)
和
[`smollm2_block8_dequant_ffn_schedule_detail.svg`](figures/smollm2_block8_dequant_ffn_schedule_detail.svg)。

### RMSNorm

`rmsnorm_test` 完全通过 MEM 和 VXM 计算 `[32,32]` FP16 RMSNorm。ALU0 每拍对一个
hidden column 求平方，ALU1 通过 feedback 在 32 个物理 lane 中并行累加
`sum(x^2)`。随后 VXM 计算 inverse RMS 并将其保留在 ALU 中。MEM 让每个 `x`
vector 比对应的 `gamma` 早一拍到达，两个 multiply ALU 可以直接消费，不再需要
Pass stage。该测试不使用 MXM 或 MEM accumulator。

### SmolLM2 Attention

`smollm2_attention_test` 验证：

```text
Q/K/V projection -> Q/K RoPE -> QK score -> scaled 三-pass softmax
-> P x V -> o_proj[128,576]
```

配置为 sequence length 128、hidden size 576、9 个 query head、3 个 KV head、
head dimension 64。QK、P x V 和 o_proj 在 stream/accumulator 资源允许时把独立
work 分配到四个 MXM。SXM 为 attention replay 准备 packed/transpose layout。
完整 numerical golden 在 94,761 个调度周期通过。

各 phase 时序、MXM 利用率和后续 overlap 机会见
[attention_pipeline_optimization.md](attention_pipeline_optimization.md)。

### Decoder Layer Decode

`decoder_layer_decode_test` 从已经物化的 128-token K/V cache 开始，让 token 128
完整经过一个 tile-scale decoder layer：

```text
RMSNorm -> Q/K/V -> RoPE -> append K/V
        -> QK -> softmax -> P x V -> O -> residual
        -> RMSNorm -> Gate/Up -> SwiGLU -> Down -> residual
```

测试采用 hidden/head dimension 32、intermediate dimension 64，使每条矩阵边界
恰好对应一个 MXM tile。所有 GEMV、QK 和 P x V 都通过系统 MEM/MXM 路径发射。
Decode Q 使用两条 stream 的 `IWColumn` 写入一个 weight column，随后把 cache 中
的 129 条 K 连续送入 MXM。非线性组合遵循 RMSNorm、VXM、attention 和 SwiGLU
独立测试已经验证的 FP16 边界。扩展到 SmolLM2 的 576/1536/9-head 配置时，只需
把相同 tile 映射平铺到四个 MXM。

Decode control 显式编码两种 layout：

- `Linear1x16` 保留原有的 16-supercell 串行 reduction。每个 tile 从 8 条 BF16
  stream 装入 4 个独立 activation vector，一个 wave 经过 16 个 cell stage 后
  产生 8 个输出。
- `Native4x4` 按物理结构使用 4 列、每列 4 个 cell。每个 tile 从 2 条 BF16
  stream 装入一个 8 元素 activation vector，并广播到该行 4 个 cell。权重第
  `c` 列比第 0 列晚 `c` 拍发射，因此 `(tile,column)` 在
  `launch + tile + column` 执行；4 条 8-output 纵向 reduction 在 7 个 stage 后
  同时完成。

`mxm_decode_layout_comparison_test` 用两种 layout 执行同一个 `K=128, N=32`
GEMV，并要求 BF16 输出逐 bit 相同。完整 decode FFN 使用 `Native4x4`；decode
attention 在 RoPE/cache resident layout 整体迁移前继续使用 `Linear1x16`。

### 多 executable 边界

`TspSliceSystem::reset_execution_state()` 在 command binary 之间建立干净的
cycle-0 边界。它会 reset ICU queue、MEM 指令与 stream 状态、MXM datapath
和 accumulator、VXM、SXM 以及系统 cycle，但保留 MEM SRAM 内容。模型级执行
需要区分这两类状态：decoder activation 可以继续驻留在 SRAM，而每一层都不会
继承上一层残留的 pipeline 或 stream 状态。

## 11. 日志与调度图

长回归默认关闭日志，因为逐拍 stream dump 会主导运行时间。
`TspSliceSystem::LogSinks` 可分别记录 ICU、MEM、MXM、VXM、SXM 和 system。

调度 CSV 使用以下环境变量：

- `FTLPU_SCHEDULE_TRACE=<path>`
- `FTLPU_SCHEDULE_TRACE_ONLY=1`
- `FTLPU_SCHEDULE_REPORT=1`

详细图由 `scripts/render_schedule_trace.py` 和
`scripts/render_swiglu_schedule_trace.py` 生成。Accumulator 颜色为：

- 紫色：partial sum 保留在 MXM accumulator；
- 红色：最终结果送流并清零 slot。

## 12. 已知限制

- 模型不与私有 Groq 硬件或 ISA bit-accurate；
- Gather/Scatter 尚无 address-stream 执行 datapath；
- 离线 schedule 仍在 integration test 内构造，没有独立编译器或可复用 program
  文件格式；
- 资源分配仍使用 workload-specific SRAM slice 和 stream ID，没有通用 allocator；
- 模拟器每拍仍扫描较多 inactive state，长 workload 运行较慢。

## 13. 后续工程方向

1. 提取可复用的 offline program 和 resource-calendar scheduler；
2. 增加 SRAM/stream 生命周期分配与冲突诊断；
3. 基于现有 ISA codec 增加 program serialization；
4. 用 data-ready event 替代全局 phase barrier，继续流水 attention；
5. 对 queue NOP span 和整系统 idle interval 增加 fast-forward。
