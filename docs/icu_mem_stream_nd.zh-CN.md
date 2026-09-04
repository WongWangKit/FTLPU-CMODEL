# MEM ICU N 维流描述符

## 目的

`MEM_STREAM_ND` 用一条 MEM ICU 队列项描述一组规则的 SRAM 读写。它保持
逐 cycle 精确调度，同时避免为 projection 的每个小块分别保存一条队列指令。

描述符携带一条原生 `MemInstruction` 和以下调度字段：

```text
start_cycle
rank                                  // 1..3
count[3]
cycle_stride[3]
address_stride[3]
```

第 0 维是最内层。坐标 `i[d]` 对应的展开结果为：

```text
cycle   = start_cycle + sum(i[d] * cycle_stride[d])
address = native.address + sum(i[d] * address_stride[d])
```

原生指令中的 opcode、stream register、方向和 write-tap 行为都不改变，只有
SRAM 地址随计数器递增。

## MEM ICU 执行方式

每条描述符在 i-MEM/IQ 中只占一个 entry。译码后，它进入 MEM ICU 的活跃
描述符 calendar。ICU 保存三层计数器，选择 next issue cycle 最早的描述符，
发射一条原生传输，再按里程计方式推进计数器。因此，同一 MEM 队列上的多条
描述符可以交错执行，不需要 host 生成 NOP。

以下情况由 CModel 立即报静态调度错误：

- rank 不在 1..3、count 为 0 或 cycle stride 为 0；
- 高维 cycle stride 与低维完整时间跨度重叠；
- 两条活跃描述符在同一 cycle 向同一 MEM 队列发射；
- 描述符到达 ICU 时已经错过绝对 start cycle；
- 活跃的 N 维描述符区域中混入旧式队列命令。

## 硬件映射

一个直接的 RTL 实现需要为每条活跃描述符保存三组 count/current、三组 cycle
stride、三组有符号 address stride，以及 next-cycle 和 current-address 寄存器；
再由有界 next-issue calendar 或比较器选择本 cycle 到期的描述符。后续应在
target model 中明确支持的最大 rank、描述符 FIFO 深度和最大 in-flight 数量。

它是通用 MEM 访问指令。projection 名称、tensor shape 和模型语义仍留在
编译器 IR 中，不进入 ICU ISA。

