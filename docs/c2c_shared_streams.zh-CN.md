# C2C 共享 SR 接收通路

外部 C2C 接口每方向可配置 1..8 条 lane，每条 lane 每 cycle 搬运一个完整的
32-byte vector。lane ID 只是传输通道号；RX 在 MEM/C2C 边界把它映射到普通
westbound SR。配置 8 条 lane 时默认使用 `W24..W31`。

系统不再提供直写 SRAM 或专用 C2C stream 的接收旁路：

```text
host -> DDR4 -> C2C DMA -> C2C RX -> 普通 west SR -> MEM Write -> SRAM
```

RX 每收到一个 vector，就向目标 `(hemisphere, slice, bank)` 的 MEM ICU C2C
命令上下文发送点对点通知。粗粒度 synchronized-write 描述符保存 vector 数量、
目标起始地址和步长、SR ID，以及固定的 RX-to-MEM 传播延迟。MEM ICU 记录每个
通知的到达周期，并在 `notify_cycle + transport_latency` 发出对应 `Write`。
连续 vector 因而可以每拍写一个；DDR 延迟抖动产生气泡时，MEM 消费端会自动等待
下一次通知，不会读取无效 SR segment。

C2C 命令上下文只用于同步和发令，不是独立数据通路。它与计算侧 MEM queue 共用
同一个物理 MEM 端口；同拍发出两条功能指令会被判为静态调度冲突。C2C 数据同样
受普通 SR 的生产者和路由冲突约束。page ready 表示最后一条 MEM write 已发出，
而不是 DMA 刚把最后一个 vector 送到 RX。

## DDR 时序模型

DDR 是外部 backing store，不是初始化 LPU MEM 的旁路。`initialize_vector` 和
`read_vector` 只表示 host 侧访问 DDR；进入 LPU 的数据仍必须经过 DMA、C2C、SR
和 MEM Write。

默认平台采用 500 MHz LPU 和双通道 DDR4-3200，51.2 GB/s 峰值在模型中表示为
读写共享的 102.4 bytes/cycle 小数 token budget。默认读延迟为 35 cycle 加
0..15 cycle 确定性抖动，写延迟为 25 cycle 加 0..10 cycle。queue depth、通道数、
带宽、基础延迟、抖动范围和 seed 均可配置。
