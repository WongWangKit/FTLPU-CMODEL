# MXM ICU N 维流描述符

## 目的

`MXM_STREAM_ND` 用一条 MXM ICU 队列项描述一组规则的权重加载、计算、
accumulator read 或反量化发射点。描述符不编码 projection 名称或 tensor
shape，只是在一条原生 MXM 指令外增加逐 cycle 精确的仿射调度：

```text
start_cycle
rank                                  // 1..3
count[3]
cycle_stride[3]
operand_stride[3]
induction_target
```

坐标 `i[d]` 的实际发射位置为：

```text
cycle         = start_cycle + sum(i[d] * cycle_stride[d])
operand_delta =               sum(i[d] * operand_stride[d])
```

归纳目标由队列和原生 opcode 共同限定：

| 队列/原生 opcode | 递增字段 |
| --- | --- |
| MXM load / `IW` | `weight_column` |
| MXM compute / `Compute` 或 `AccumulatorRead` | `accumulator_address` |
| MXM dequant | 无 |

其他原生字段全部保持不变，特别是 stream destination、accumulator clear 和
BF16 输出转换不会在展开过程中丢失。

## MXM ICU 执行方式

公开接口为 `enqueue_mxm_load_stream_nd`、`enqueue_mxm_compute_stream_nd` 和
`enqueue_mxm_dequant_stream_nd`。每条描述符在 i-MEM/IQ 中占一个 entry，译码
后进入本队列的活跃描述符 calendar。calendar 选择 next issue cycle 最早的
描述符，应用当前 operand delta，发射一条原生指令，再推进嵌套计数器。

第 0 维是硬件最内层计数器。高维 stride 必须大于全部低维的完整时间跨度，
保证计数器按时间单调推进。不同描述符仍然可以交错，由每队列的 calendar
仲裁各自的 next issue cycle。

CModel 会拒绝非法 rank/count/stride、opcode 与 induction 不匹配、dequant
携带 operand induction、错过绝对发射 cycle，或同一队列同一 cycle 有两条
描述符到期。

## 硬件映射

三个 MXM ICU 可以共用描述符前端，同时保留各自的原生 payload 译码器。每条
活跃描述符需要三组 count/current、三组 cycle stride、三组有符号 operand
stride、一个 induction selector、next-cycle 寄存器和原生 payload，再由有界
比较器或 timing wheel 选择本 cycle 到期的描述符。

当前软件 extension-word 编码只用于验证语义，不是最终 wire format。后续应在
target 配置中明确最大 rank、descriptor FIFO 深度和每个 MXM 队列允许同时活跃
的描述符数量。

