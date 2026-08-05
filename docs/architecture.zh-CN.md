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
| MXM 阵列 | 32 x 32 systolic K-block 输出，每侧共享 ACC |
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

正常映射下，每个本地 MXM 固定拥有一个 16-stream 窗口。Compute 从窗口的第一个
byte group 读取 activation。全宽权重装载在该拍独占整个窗口；后台权重装载只使用
固定的后半窗口，因此 activation 与 weight 不会占用相同 stream ID。

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
slice 组成一个 group，位于两个 stream-register boundary 之间。

每个 slice 是单端口：同一拍即使地址不同，也不能同时 Read 和 Write。MXM 的
int32 部分和通过普通四-slice Read/Write 保存和取回，MEM 内不再包含加法单元。

### 指令

- `Read(address, stream)` 读取 tile-local 8-byte SRAM segment，写入一条 stream；
- `Write(address, stream)` 消费一条 8-byte stream segment 并写入 SRAM；
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
- activation 和 systolic partial-sum pipeline。

每侧两个 MXM 共用位于 MEM 边界的一个累加单元。每个输出 lane 有两条固定 int32
加法路径：独立模式下两个 MXM 各使用一条；合并模式固定为 ACC0 计算
`MXM0 + MXM1`，ACC1 计算 `ACC0 + old`。不存在 accumulator crossbar，也不存在
绑定在单个 MXM 上的大结果 bank。

### 权重装载

每个本地 MXM 固定拥有一个 16-stream 窗口：本地 MXM0 使用 `0..15`，本地 MXM1
使用 `16..31`。`IW` 有三种模式：

- `Full` 消费完整 16-stream 窗口并全宽写入；
- `BackgroundLowerHalf` 固定消费窗口后8路，写weight列 `0..7`；
- `BackgroundUpperHalf` 固定消费同样的后8路，写weight列 `8..15`。

两个后台 pulse 完成后，非活跃 buffer 才变为有效。第一半写入会使原来的完整
buffer失效，从而避免Compute看到新旧权重混合。较小的评估配置保持相同规则，按
两个相等的固定半区工作。

W8 权重采用按输出列的对称 scale：

```text
scale[n] = max_k(abs(W[k,n])) / 127
```

### Compute

`Compute(buffer, output_stream_base)` 是一拍 control pulse。Compute不再携带activation
stream选择字段，而是固定从该MXM本地16-stream窗口的第一个byte group读取。
连续pulse注入连续activation vector；weight buffer和output stream base随activation波传播。

每个 supercell 是 weight-stationary 标量 MAC 阵列。一个物理 MAC row 消费一个
activation 分量，并把该标量广播到全部输出列。每周期都可以输入一条完整 activation
row，但第 `k` 个分量延迟 `k` 拍进入 MAC row `k`。稳定状态下，同一周期 MAC row
`k` 处理的是 token row `t-k`，partial sum 则逐级进入下一 MAC row。因此 supercell
延迟等于 block K 维度，启动间隔为一拍。

下一个外层 supercell row 的 Compute control 和 activation 会延迟完整的 MAC-array
周期，以便和对应的 northbound partial sum 对齐；activation vector 在外层
supercell column 之间仍然每拍向东推进一级。完成的 K block 进入每侧共享 ACC。
最终结果经一拍 FP16 Cast 写入固定两-byte stream group，随后进入 VXM Input Buffer；
多上下文的非最终结果以四条 int32 byte stream 交给编译器安排普通 MEM 存储。

全部 MXM runtime state 由整系统持有；整系统测试不使用独立 GEMM engine 或 runtime
helper。

## 6. VXM

中心 VXM 每个 tile row 有一个 superlane，每个 tile lane 对应一条 VXM lane。
每条 lane 有 16 个物理 ALU stage，8 条共享 compact-instruction 通道通过本地 decode
驱动这些 stage，并支持 2、4、8 三种 chain depth。

支持的 opcode：

```text
Pass Add Subtract Multiply Divide Negate Abs Min Max Clamp
Square Sqrt Exp Log Relu Cast
```

物理 stream operand 统一为 FP16；其他 operand 可选择 previous、original、
auxiliary、accumulator、immediate 或 feedback。96-bit compact packet 携带 operation、
operand、precision、output type、chain depth、repeat count 和 accumulator 控制。
输出使用中心 64-stream 编号，compact packet 不再携带 hemisphere 字段。
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

每个 hemisphere 有三条可独立调度的 SXM ICU queue：Stream、Transpose 和
Permute 各一条，因此独立的 transpose 与 permute 数据通路可以在同一逻辑 VLIW
周期发射。没有 SXM 指令时，east stream 以普通一拍 link 从 `sreg11` pass 到
`sreg12`。

## 8. ICU 与 ISA

ICU 共拥有 114 条独立 queue：

| Queue 类别 | 数量 |
| --- | ---: |
| MEM | 88 |
| MXM load | 4 |
| MXM compute | 4 |
| MXM activation dequantize | 4 |
| VXM compact control | 8 |
| SXM Stream | 2 |
| SXM Transpose | 2 |
| SXM Permute | 2 |

每条独立 queue 旁边设置本地 i-MEM，并通过本地取指控制器以每拍一条的速率预取到
有限深度 IQ。运行时取指不占用数据 MEM 容量，也不经过数据 Stream Register。
编译器为每条局部程序提供 `base_pc`、指令数量和 `start_cycle`，确保指令在发射前
已经到达 IQ。i-MEM 位宽、i-MEM 深度和 IQ 深度均按模块类型参数化；当前 MXM 为
32-bit，MEM/VXM/SXM 为 96-bit。

已实现的 queue command：

- `NOP N`：该 queue 延迟 N 拍；
- `Repeat n,d`：以间隔 d 重复上一条指令 n 次；
- MEM Repeat 可附带 signed address stride。

当前紧凑 codec 覆盖：

- 32/96-bit MEM 指令 codec；
- 32-bit MXM control 指令；
- 32-bit ICU control 指令；
- SXM instruction packet。

VXM compact instruction 使用 VXM 本地的 96-bit codec，不再编码为旧 core program
packet。字段范围和保留位由对应单元测试验证。

## 9. 调度模式

### Accumulator 生命周期

单个活动结果可以保留在共享 ACC 的操作数/结果寄存器中。多个 token 上下文使用
普通 MEM `Write` 保存 int32 部分和，再用普通 `Read` 取回。编译器通过 MXM
Compute 控制选择本地或 MEM 的 start/accumulate/finalize 模式。

### 单端口 MEM

地址不同也不能消除 slice 冲突。只要 Read、Write 占用同一物理 slice，其时间窗口
就必须错开。

## 10. 日志与调度图

长回归默认关闭日志，因为逐拍 stream dump 会主导运行时间。
`TspSliceSystem::LogSinks` 可分别记录 ICU、MEM、MXM、VXM、SXM 和 system。

## 11. 已知限制

- 模型不与私有 Groq 硬件或 ISA bit-accurate；
- Gather/Scatter 尚无 address-stream 执行 datapath；
- SXM 尚无二进制指令 codec；
- 离线 schedule 仍在 integration test 内构造，没有独立编译器或可复用 program
  文件格式；
- 分布式本地 i-MEM 的外部装载过程尚未建模；
- 资源分配仍使用 workload-specific SRAM slice 和 stream ID，没有通用 allocator；
- 模拟器每拍仍扫描较多 inactive state，长 workload 运行较慢。

## 12. 后续工程方向

1. 提取可复用的 offline program 和 resource-calendar scheduler；
2. 增加 SRAM/stream 生命周期分配与冲突诊断；
3. 基于现有 ISA codec 增加 program serialization；
4. 用 data-ready event 替代全局 phase barrier，继续流水 attention；
5. 对 queue NOP span 和整系统 idle interval 增加 fast-forward。
