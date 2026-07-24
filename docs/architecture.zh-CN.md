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
| MEM slice | 每侧 44 个，全芯片 88 个 |
| MEM group | 每侧 11 个，每组 4 个 slice |
| Stream-register column | 每侧 13 个（`sreg0..sreg12`） |
| 每 lane stream | 32 条 eastward + 32 条 westward |
| Stream-register 位宽 | 1 byte |
| SRAM 容量 | 每 slice 256 KiB，全芯片 22 MiB |
| MXM | 共 4 个，每侧 2 个 |
| MXM 阵列 | 32 x 32 FP16 乘法、FP32 累加 |
| VXM | 中心 1 个 slice，每 lane 16 个 ALU |
| SXM | 每侧 1 个四-tile slice |

一个 MEM slice 拥有一个 `8192 x 32-byte` SRAM block，逻辑上分成两个
4096-row bank。每个 row 横跨 4 个 tile；指令波到达某个 tile 时，该 tile 只访问
自己的 8-byte segment。

## 2. 完整芯片拓扑

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W(44) <-> VXM <-> MEM.E(44) <-> SXM.E <-> MXM0/MXM1
```

两个 hemisphere 使用相同的局部朝向：

- `sreg0` 靠近 VXM；
- 11 个 MEM group 位于 `sreg0..sreg11`；
- SXM 把 MEM 边界 `sreg11` 连接到 MXM 边界 `sreg12`；
- east stream 从 VXM 流向 MXM；
- west stream 从 MXM 流向 VXM。

全局 MEM queue `0..43`、MXM `0..1` 属于 East；MEM queue `44..87`、MXM
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

每个 hemisphere 有 44 个 MEM slice column，每个 slice 一条指令队列。相邻四个
slice 组成一个 group，位于两个 stream-register boundary 之间。group9 和 group10
对应 slice `36..39`、`40..43`，既保留普通 SRAM 功能，也支持 FP32 accumulation。

每个 slice 是单端口：同一拍即使地址不同，也不能同时 Read 和 Write。Accumulator
操作会在该 tile/周期占用整组四个 slice。

### 指令

- `Read(address, stream)` 读取 tile-local 8-byte SRAM segment，写入一条 stream；
- `Write(address, stream)` 消费一条 8-byte stream segment 并写入 SRAM；
- `Accumulate(address, west_stream_base, destination)` 消费连续四条 west stream
  组成 FP32，与四-slice SRAM 值相加，然后：
  - 写回 SRAM；或
  - 从相同四条 west stream 输出，并清零该 SRAM slot；
- `Gather`/`Scatter` 已编码，但因为尚未实现 address-stream datapath，执行时会拒绝。

一条指令波最终经过四个 tile，因此完整指令波以四个错拍的 8-byte segment 搬运
一个 32-byte 物理向量 row。

### 地址

参考的公开风格软件地址为：

```text
[39:24] chip
[23]    hemisphere
[22:17] slice
[16]    logical SRAM bank
[15:4]  4096-row bank 内的 row offset
[3:0]   软件 byte offset
```

`MemInstruction::address` 只保存对应软件位 `[16:4]` 的 13-bit slice-local row，
范围为 `0..8191`。测试的初始化/结果 API 另外指定 tile 和 lane byte。

## 5. MXM

每个 `Mxm` 包含：

- `4 x 4` supercell 阵列；
- 每个 supercell 一个 `8 x 8` FP16 weight block；
- 每个 supercell 两个对等 weight buffer；
- 从南向北传播的 control slice；
- activation flow 和 FP32 output state。

### 权重装载

`IW(buffer, column)` 把一个显式 supercell-column block 写入指定 buffer。
两位 `column` 字段选择 `0..3`，装载顺序不会隐式改变最终 layout。不存在 `LW`。

一个 IW pulse 消费 16 条 east stream：

```text
8 个 FP16 值 x 每值 2 条 byte stream = 16 条 stream
```

本地 MXM0 使用 `E0..E15`，MXM1 使用 `E16..E31`。连续四个 IW pulse 填满一个
32-column weight tile。向空闲 buffer 执行 IW 时，另一个 buffer 仍可支持在途 Compute。

W8 权重采用按输出列的对称 scale：

```text
scale[n] = max_k(abs(W[k,n])) / 127
```

IW 前，VXM 先把 INT8 乘对应 scale，再 cast 为 FP16。

### Compute

`Compute(buffer, activation_stream_base, output_stream_base)` 是一拍 control pulse。
连续 pulse 注入连续 activation vector；buffer 和 stream base 会随 activation 波传播。

每个 supercell 将 8 个 FP16 activation 与八列 FP16 权重做点积。activation 向东
穿过四个 supercell column，partial sum 向北传播并按 FP32 累加。完成的输出会自动
写入 Compute 指定的连续四条 west byte stream。不存在独立 MXM output 指令，也
不存在软件输出队列。

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

operand 可以来自 INT8、FP16、INT32、FP32 stream，FP32 immediate，或前一拍 ALU
output。结果可以保留在 ALU register，也可以写入指定 stream 和目标 hemisphere。

量化/反量化不是独立 opcode，而是 ALU 指令图。例如 W8 dequant 由 Multiply 和 Cast
组成；SwiGLU 由算术、Exp、用于流水延迟的 Pass 和 Cast 组成。RMSNorm 使用 ALU
feedback 计算 `sum(x^2)`，并在 `x`、`gamma` 流过时保留 inverse RMS。

## 7. SXM

系统包含两个独立 SXM，每个 hemisphere 一个。SXM 只转换 east stream；west stream
只经过对称寄存器 hop。

每个 SXM 有四个 tile row。Transpose 指令每拍从南向北推进一个 tile，使每个 tile
捕获与自身匹配的对角波前。FP16 low/high byte 作为两个 plane，tile-local Transpose
完成一个 `8 x 8` block 的行列交换。

Transpose 输出先打一拍，再由 Permute 消费。Permute 在 4 个 superlane、32 个 lane
之间重排完整 block。当前实现使用一个 transpose buffer；相同 destination 的 block
可按 `II=4` 流水。

每个 hemisphere 有两条 SXM ICU queue：Transpose 和 Permute 各一条。没有 SXM
指令时，east stream 以普通一拍 link 从 `sreg11` pass 到 `sreg12`。

## 8. ICU 与 ISA

ICU 共拥有 116 条独立 queue：

| Queue 类别 | 数量 |
| --- | ---: |
| MEM | 88 |
| MXM load | 4 |
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
- 四个 32-bit word 的 VXM ALU 指令；
- 32-bit ICU NOP/Repeat command。

SXM 指令仍是 C++ control object，尚无二进制 codec。字段范围和保留位由
`tests/core/instruction_codec_test.cpp` 验证。

## 9. 调度模式

### W8A16 权重与 Activation 共存

原始 INT8 权重从 MEM 沿 west stream 到 VXM；反量化后的 FP16 权重再沿 east stream
到 MXM，后一个阶段才可能与 activation 冲突。

| 活跃操作 | FP16 IW stream | 共享 activation |
| --- | --- | --- |
| 装载本地 MXM0 | `E0..E15` | `E16..E17` |
| 装载本地 MXM1 | `E16..E31` | `E0..E1` |
| 没有 IW | 无 | 通常为 `E0..E1` |

同拍装两个 MXM 会占满 32 条 east stream。因此离线 FFN schedule 每次只装一个
空闲 buffer，并把 activation 放到另一半。本地两个 MXM 广播消费同一对 FP16
activation。

### 权重乒乓

projection reduction 中，Compute 使用 buffer `k mod 2`，VXM 和 IW 同时把
reduction `k+1` 准备到另一个 buffer。SRAM slice、stream ID、VXM ALU 和 MXM
load queue 都是显式调度资源。

### Accumulator 生命周期

非最终 reduction 使用 `Accumulate(..., SRAM)`；最终 reduction 使用
`Accumulate(..., Stream)`，输出并清零 slot。只有最终 stream result 发出后，地址
才可复用。

### 单端口 MEM

地址不同也不能消除 slice 冲突。只要 Read、Write、Accumulate 占用同一物理 slice，
其时间窗口就必须错开。

## 10. 已验证整系统 Workload

### W8A16 Projection

`w8a16_projection_test` 计算：

```text
A[128,576] fp16 x W[576,1536] int8 -> C[128,1536] fp32
```

权重采用按输出列的 W8 对称 scale。VXM 反量化权重，MXM0/1 计算相邻 output block，
两个 MEM accumulator group 累加 18 个 K tile。全部 196,608 个输出与考虑 FP16
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

### RMSNorm

`rmsnorm_test` 完全通过 MEM 和 VXM 计算 `[32,32]` FP16 RMSNorm。ALU0 每拍对一个
hidden column 求平方，ALU1 通过 feedback 在 32 个物理 lane 中并行累加
`sum(x^2)`。随后 VXM 计算 inverse RMS，并应用到流入的 `x` 和 `gamma`。该测试
不使用 MXM 或 MEM accumulator。

### SmolLM2 Attention

`smollm2_attention_test` 验证：

```text
Q/K/V projection -> Q/K RoPE -> QK score -> scaled 三-pass softmax
-> P x V -> o_proj[128,576]
```

配置为 sequence length 128、hidden size 576、9 个 query head、3 个 KV head、
head dimension 64。QK、P x V 和 o_proj 在 stream/accumulator 资源允许时把独立
work 分配到四个 MXM。SXM 为 attention replay 准备 packed/transpose layout。
完整 numerical golden 在 81,273 个调度周期通过。

各 phase 时序、MXM 利用率和后续 overlap 机会见
[attention_pipeline_optimization.md](attention_pipeline_optimization.md)。

## 11. 日志与调度图

长回归默认关闭日志，因为逐拍 stream dump 会主导运行时间。
`TspSliceSystem::LogSinks` 可分别记录 ICU、MEM、MXM、VXM、SXM 和 system。

调度 CSV 使用以下环境变量：

- `FTLPU_SCHEDULE_TRACE=<path>`
- `FTLPU_SCHEDULE_TRACE_ONLY=1`
- `FTLPU_SCHEDULE_REPORT=1`

详细图由 `scripts/render_schedule_trace.py` 和
`scripts/render_swiglu_schedule_trace.py` 生成。Accumulator 颜色为：

- 紫色：partial sum 保留在 SRAM；
- 红色：最终结果送流并清零 slot。

## 12. 已知限制

- 模型不与私有 Groq 硬件或 ISA bit-accurate；
- Gather/Scatter 尚无 address-stream 执行 datapath；
- SXM 尚无二进制指令 codec；
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
