# VXM_STREAM_ND

`VXM_STREAM_ND` 在一条 ICU 宏指令中携带原生 96-bit VXM 紧凑配置包，以及
一到三维的绝对 cycle 迭代空间。ICU 锁存一次配置，再用 ND counter 生成全部
启动点，不需要配置槽或 CONFIG/RUN 配对状态。

对于坐标 `i[d]`，ICU 在以下 cycle 发射所选配置包：

```text
cycle = start_cycle + sum(i[d] * cycle_stride[d])
```

`VXM_STREAM_ND` 不支持操作数字段归纳。opcode、立即数、stream group、chain
depth、输出模式或 datapath repeat count 任一不同，都必须使用不同描述符。紧凑
配置包内部的 repeat count 不由 ICU 展开；每次发射后，它决定 Superlane
Current Config Register 连续执行多少 cycle。

CModel 将该宏指令表示为 `IcuVxmStreamNdSchedule` 加
`VxmCompactInstruction`。同队列发射点重叠、错过绝对 cycle、非零 operand
stride，或描述符执行期间混入旧式命令都会立即报错。
