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
- 每个 tile/superlane 8 lane；
- 每 lane 64 条 stream：32 eastward、32 westward；
- 每个 stream register 1 byte；
- 当前完整建模两个 hemisphere，每侧 44 个 MEM/SRAM slice column，共 88 个；
- 每个 slice column 有 4 个 tile-local SRAM block；
- 每个 tile-local SRAM block 有两个 bank；
- 每个 bank 为 4096 word x 8 byte；
- 完整 slice 容量为 4 x 2 x 4096 x 8 byte = 256 KiB。

软件可见 MEM 地址遵循公开风格的布局：

```text
[39:24] TSP chip
[23]    hemisphere，East=1，West=0
[22:17] slice number 0..43
[16]    bank
[15:4]  所选 bank 内的 4096-word offset
[3:0]   16-byte SRAM word 内的 byte offset
```

当前模型实例化两个 hemisphere。每个 MEM slice 接收一条 instruction stream；
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

- 有 88 条独立 MEM instruction queue，每个 hemisphere 的每个 slice column 一条；
- 指令从 tile 0 进入，每 cycle 向北移动一个 tile；
- 每个 MEM slice column 建模为单 instruction port，同一 cycle 不能同时发射
  `Read` 和 `Write`，需要两者的 schedule 必须安排到不同 cycle；
- `Read(address, stream)` 从 address bit 16 指定的 bank 读取一个对齐的 16-byte
  SRAM word，每 lane 一个 byte，并写入该 MEM slice 相邻 stream register 上的
  目标 stream；
- `Write(address, stream)` 从目标 stream 消费 16 byte，每 lane 一个 byte，并
  写入 address bit 16 指定 bank 的一个对齐 SRAM word；
- `Accumulate(address, stream)` 从 MEM queue 36 或 40 发射。选中的四-slice
  group 在相同地址各读一个 byte plane，组成 FP32；再消费连续四条 west stream
  组成另一个 FP32，相加后原址写回；
- instruction enum 中包含 `Gather` 和 `Scatter`，当前测试主要覆盖 `Read` 和
  `Write`。

每个 cycle、每个 tile，一条 MEM 指令可以为一个 stream ID 在 8 lane 上移动
8 byte。

### 最东侧 Accumulator Groups

最靠近 MXM 的 slice36..39 与 slice40..43 都保留普通 `Read`/`Write`。执行
`Accumulate` 时，选中的 group 暂时作为 FP32 SRAM vector 的四个 byte plane，
完成一次单端口 read-modify-write，并在该 tile、该 cycle 占用本 group。两个
group 拥有独立端口，可以同周期执行。被消费的数据不再向西传播，模型中没有
持久 route 或独立 accumulator 存储。

## MXM 模型

MXM 模型位于 `include/ftlpu/mxm/`。

当前组件：

- `MxmSupercell`：一个 8 x 8 FP16 weight block；
- `MxmArray`：4 x 4 supercell grid；
- `MxmControlSlice`：`IW` 和 `Compute` 的南到北 control pipeline；
- `Mxm`：包含 array、control slice，以及 activation flow、FP32 accumulation
  和 output stream injection datapath state 的 wrapper。

系统在每个 hemisphere 外侧包含两个 MXM，全芯片共四个。

### Weight 加载

每个 supercell 有两个对等 weight buffer。`IW(buffer, column)` 将一个 8 x 8
FP16 weight block 直接写入所选 row 和 supercell column。2-bit column ID
允许 column 0..3 按任意数据流顺序加载，不再发生隐式向 east 移列。
不存在单独的 `LW` commit 指令；`Compute(buffer, activation_stream,
output_stream)` 直接选择为当前 activation token 提供 weight 的 buffer。

INT8 权重先送到 VXM。8 条 multiply ALU 指令消费 8 条 W8 stream 并应用按输出列
的 scale，随后 8 条 cast 指令为每个 FP16 权重产生两条 little-endian byte stream。
FP16 weight byte 随后从 MEM 到达 east handoff stream register。
每个 MXM 每 cycle 使用 16 条 stream，承载 8 个 FP16 weight column：

- MXM0 消费 `E0..E15`；
- MXM1 消费 `E16..E31`。

完整 32 x 32 weight matrix 包含 4 个 column block。当前测试向一个 buffer
连续发射 4 个 `IW` pulse，同时另一个 buffer 仍可供 in-flight compute 使用。
两个 weight buffer 有独立的 row shift state，因此加载一个 buffer 不会扰动另一个。

### Compute

`Compute(buffer, activation_stream, output_stream)` 是 one-cycle pulse。ICU 在
有效 compute window 的每个 cycle 发射一条 `Compute`。选定的 buffer id 和
output stream base 随 activation event 在 MXM 中移动，后续 `IW` 因而可以覆盖
另一个 buffer，而不改变 in-flight work。

系统持有的 MXM runtime 对 activation flow 建模：

- activation 从 MEM 进入 tile row，具有 one-cycle 的南到北 skew；
- 每个有效 supercell 从两条 byte stream 消费 8 个 FP16 activation；
- supercell 计算该向量与 8 个本地 FP16 weight column 的 8 个 dot product；
- activation 向 east 穿过 supercell column；
- partial sum 累加成 FP32 result column。

Runtime 由 `TspSliceSystem` 持有。ICU `Compute` pulse 通过 MXM datapath 消费
MEM east handoff stream。当 tile row 3 的 contribution 完成一个 result column
block 时，MXM 自动将 FP32 result byte 注入 `Compute` 指令指定的 MEM west
stream。

## VXM 模型

VXM 模型位于 `include/ftlpu/vxm/`。

层级：

- `VxmAlu`：每个 lane 复用的标量 ALU 行为；
- `VxmLane`：一个 lane，包含 16 个 ALU 和 16 条独立发射队列；
- `VxmSuperlane`：8 lane；
- `VxmSlice`：4 个 superlane/tile，指令南到北流动。

每条 ALU 指令从 stream、立即数或前一拍 ALU 输出中选择两个操作数，并可把
INT8、FP16 或 FP32 结果写到指定 stream 和 hemisphere。ICU 为每个 ALU 提供一条
queue。测试侧的离线程序生成会把 W8 dequant 展开为 8 个并行 multiply 加 8 个
FP16 cast，并把 SwiGLU 展开为带寄存延迟的 ALU 图；运行时不存在 fused Dequant
或 Swish 指令。

## ICU 模型

ICU 实现在 `include/ftlpu/system/icu.hpp`。

Queue 数量：

- 88 条 MEM queue，每个 hemisphere 各 44 条；
- 4 条 MXM load queue，每个 MXM 一条；
- 4 条 MXM compute queue，每个 MXM 一条；
- 4 条 SXM queue：每个 hemisphere 的 Transpose 和 Permute 各一条；
- 16 条 VXM queue：每个 ALU 一条。

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
- VXM ALU instruction：4 x 32-bit word（操作/操作数/输出、两个 FP32 立即数 word
  和 hemisphere routing）；
- ICU queue command：32 bit。

MEM instruction address field 不是完整软件地址，只编码 slice-local SRAM word
address：bit 12 是 bank，bit 11:0 是 4096-word offset。`Read`/`Write` 要求
软件地址低位 byte offset 为 0。

这是紧凑的 FTLPU CModel 编码，不是 Groq 硬件二进制编码。

Codec 由 `tests/core/instruction_codec_test.cpp` 覆盖。

## 整体系统拓扑

`TspSliceSystem` 固定完整镜像拓扑：

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W(44) <-> VXM <-> MEM.E(44) <-> SXM.E <-> MXM0/MXM1
```

重要路径：

- 每个 hemisphere 都采用相同的局部 `sreg0..sreg12` 朝向：`sreg0` 靠近 VXM，
  `sreg12` 靠近本侧 MXM；
- 全局 MEM queue 0..43、MXM 0..1 属于 East；MEM queue 44..87、MXM 2..3 属于 West；

- MEM east stream 穿过 SXM，再进入 MXM weight/activation input；
- SXM 只接收 `E0..E31`。没有 SXM 指令发射时，east data 以普通 one-cycle
  register hop 从 `sreg11` 移动到 `sreg12`；
- 对称的 west register hop 将 MXM output 从 `sreg12` 移到 `sreg11`，SXM
  不转换 west stream；
- MXM int32 output 被写入 `sreg12` 的 west stream；
- west stream 穿过 MEM 到达 west edge；
- VXM 根据各 ALU 指令消费指定 hemisphere 的 stream operand；
- VXM output 可指定目标 hemisphere，支持同侧回写或跨中心路由。

System tick 处理 ICU dispatch、MEM tick、SXM evaluation、VXM stream bridge 和
MXM control slice。MXM compute/output 由 `TspSliceSystem` 内部的 `Mxm` datapath
持有；测试不再使用独立 software GEMM engine。

## 离线整系统 Workload

整系统集成测试采用严格的离线控制契约：

1. cycle 0 前，只把外部输入初始化到 MEM SRAM。
2. 离线生成完整指令时间线，并一次性装入 ICU 的各条队列。
3. 启动时钟后只调用 `TspSliceSystem::tick()`。
4. 运行结束后读取最终 MEM 状态做验证。

运行期测试代码无法从 `TspSliceSystem` 取得 MEM、MXM、VXM 或 SXM 对象。
整系统公开接口只保留命名明确的初始化/结果接口、`icu()`、`tick()` 和
`cycle()`。单元测试仍可直接实例化它所测试的单个功能单元。

面向后续编译器的流程是：

```text
workload -> 放置与调度 -> 各队列 ICU program -> system tick
```

每条指令队列拥有独立时间线。调度器通过 queue-local `NOP N` 对齐数据和指令，
通过 `Repeat n,d` 压缩规则的连续指令；MEM Repeat 还可按 signed stride 更新
slice-local SRAM 地址。

### W8A16 Projection 回归测试

`tests/integration/w8a16_projection_test.cpp` 是当前标准整系统 workload：

```text
A[128,576] fp16 x W[576,1536] int8 -> C[128,1536] fp32
```

权重使用按输出列的对称量化。测试先初始化 W8 权重和 FP16 activation，再预装
所有 ICU 队列。运行时：

- MEM Read 从八个 slice group 向西发出八条 W8 stream。
- 一条 VXM Dequant 指令应用 8 个输出列 scale，并生成包含 8 个 FP16 的 16 条
  east byte stream。
- 四个带显式 column ID 的 `IW` pulse 装载一个 32-column MXM weight tile；MXM0 和
  MXM1 负责相邻的输出 tile。
- MEM Read 在 east stream 上发出 FP16 activation vector。
- 连续 `Compute` pulse 指定 weight buffer、activation stream base 和 west
  output stream base。
- ICU MEM `Accumulate` 指令在 `sreg11` 消费 MXM FP32 输出，并在四片 SRAM
  中原址累加 18 个 K tile 的 partial sum。

两个 MXM 同时装载权重并并行 Compute。MXM0 通过 group9 累加，MXM1 通过
group10 累加；group9 距离更远一个 west register hop，因此 ACC 指令晚1拍。
最终从 SRAM 重建 196,608 个 FP32 值，与考虑 FP16 舍入的 scalar golden model
逐元素比较。

### RMSNorm 回归测试

`tests/integration/rmsnorm_test.cpp` 通过完整的 ICU/MEM/MXM/VXM 路径验证一个
`[32,32]` 的 FP16 RMSNorm 工作负载。VXM 先对每个 `x` 向量求平方；东侧
MXM0 装载 `32 x 32` 的 FP16 全 1 权重矩阵，将平方向量归约为重复的 FP32
行和，再由 MEM accumulator 写回。随后 VXM 完成除以 hidden size、加
epsilon、开方、求倒数，以及用该倒数乘原始 `x` 和 `gamma`，最后转换为 FP16。

倒数在对应的 `x`、`gamma` 之后四拍才到达，因此测试通过 `Pass` 指令显式
延迟这两个操作数，保证连续行流量的对齐，避免后一行覆盖前一行数据。测试会将
所有存回 MEM 的 FP16 元素与考虑 FP16 舍入的标量 golden reference 比较。

### W8A16 SwiGLU 回归测试

`tests/integration/w8a16_swiglu_test.cpp` 执行两路完整 projection：

```text
X[128,576] fp16 x Wgate/Wup[576,1536] int8
    -> gate/up[128,1536] fp32 -> SwiGLU[128,1536] fp16
```

MXM0 负责 gate，MXM1 负责 up；两个 MEM ACC group 分别累加全部 18 个 K tile。
projection 完成后，ICU MEM Read 将 gate 放到 `W0..W3`、up 放到 `W4..W7`，
SwiGLU ALU 程序从 west edge 同拍消费输入，流水线在 `E0..E1` 输出 FP16，随后写入
slice29 和 slice30。测试独立验证两路累加 projection 以及全部最终 FP16 结果。

### 初始化与结果读取

允许的非 ICU API 被限制为：

- `initialize_mem_sram_lane_byte(...)`：只用于外部数据初始化。
- `read_mem_sram_lane_byte(...)`：读取最终 SRAM 状态。

整系统没有公开接口可以直接发射 MEM/MXM/VXM/SXM 指令、运行期注入 stream，
或只推进某一部分 datapath。

## 日志

长整系统回归默认关闭日志，因为逐 cycle stream trace 会主导模拟时间。开发较小
workload 时，可以用 `TspSliceSystem::LogSinks` 分别接收 ICU、MEM、MXM、
VXM、SXM 和 system trace。单元级 trace demo 仍保留在 `examples/`。

## 数据 Layout 汇总

W8A16 projection 把权重放在八个 group 各自的一个 MEM slice：
`0,4,8,...,28`。它们到 VXM 的 westward 距离不同，因此离线调度会提前不同
周期发 MEM Read，使八个值与对应 VXM 指令同时到达。

FP16 activation 在 slice `32..35` 中复制两份：两条 byte stream 输入 MXM0，
另两条输入 MXM1。MXM0 FP32 result 存在 slice `36..39`，MXM1 result 存在
slice `40..43`；每组四片分别保存四个 byte plane，地址为
`row * 48 + output_block`。

每条 `IW` 都携带目标 column ID，因此发射顺序不会改变最终 weight layout。
Activation row 和 output column 均按 MXM 的 32-element 维度分块。

## 已知限制

- 模型不与任何私有 Groq ISA bit-accurate；
- `Gather` 和 `Scatter` 不是当前测试重点；
- VXM 当前每 lane 实现 16 个 ALU，高层功能由离线指令图展开；
- Softmax 仍需定义 datapath 和指令；
- CModel 仓库本身尚不包含编译器，`OfflineIcuProgram` 是编译器输出容器的原型。

## 建议的下一步

近期工程工作：

- 根据更细致的硬件 pipeline 假设收紧 MXM runtime timing；
- 将 `OfflineIcuProgram` 从测试局部 helper 提取成可复用 module；
- 使用 ISA codec 增加文本或二进制 program dump/load 格式；
- 增加简单 compiler pass，为 projection workload 生成 MEM Read/Write、MXM
  IW/Compute 和 VXM 功能单元 timeline；
- 为 stream-register 和 queue collision 增加资源冲突诊断。
