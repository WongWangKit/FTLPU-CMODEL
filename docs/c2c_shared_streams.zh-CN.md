# C2C 共享 SR 接收通路

外部 C2C 接口每方向可配置 1..8 条 lane，每条 lane 每 cycle 搬运一个完整的
32-byte vector。lane ID 只是传输通道号；RX 在 MEM/C2C 边界把它映射到普通
westbound SR。配置 8 条 lane 时默认使用 `W24..W31`。

系统不再提供直写 SRAM 或专用 C2C stream 的接收旁路：

```text
host -> DDR4 -> C2C DMA -> C2C RX -> 普通 west SR -> MEM Write -> SRAM
```

每个物理 `(hemisphere, slice, bank)` 只有一个 MEM ICU、一个本地 i-MEM、
一个 FIFO 和一个程序计数器。普通 `READ_3D`、`WRITE_3D`、`WRITE_TAP_3D`
与 `MEM_WRITE_SYNC` 共用这一条程序流；不存在 C2C sidecar MEM queue 或第二个
命令上下文。`MEM_WRITE_SYNC` 在 i-MEM 中占连续两个 96-bit 字：第一个字是同步
header，第二个字是原生 MEM write template。

runtime loader 在 launch 之前把这条两字指令插入 MEM 程序；launch 之后不会追加
或修改本地 i-MEM。唯一的程序计数器走到 `MEM_WRITE_SYNC` 后，该指令占据队头，
等待 C2C RX 发给目标 MEM ICU 的匹配点对点通知，后续普通 MEM 指令不能旁路。
描述符保存 vector 数量、目标起始地址和步长、SR ID，以及固定的 RX-to-MEM
传播延迟。每个通知 token 自带一个从 0 开始的相对 age counter；每个 ICU tick
完成发射判断后，该 counter 增加 1。age 达到 `transport_latency + 1` 时，对应的
`Write` 才可发射，其中额外的 1 拍表示寄存后的到达边界。ICU 不保存或比较任何
全局/绝对 cycle。连续 vector 因而可以每拍写一个；DDR 延迟抖动产生气泡时，
唯一的 MEM 命令流会停在当前指令，不会读取无效 SR segment。

每条 `MEM_WRITE_SYNC` 还会为 MEM ICU 保留一个静态调度 ownership window。
如果所有 vector 提前到达并完成写入，该指令仍保持在队头直到窗口结束，保证后续
静态时序不提前；如果数据迟到，该指令则越过窗口末尾继续占用队列，直到最后一次
write 发出。C2C 数据仍受普通 MEM 端口、SR 生产者和路由冲突约束。page ready
表示最后一条 MEM write 已发出，而不是 DMA 刚把最后一个 vector 送到 RX。

## DDR 时序模型

DDR 是外部 backing store，不是初始化 LPU MEM 的旁路。`initialize_vector` 和
`read_vector` 只表示 host 侧访问 DDR；进入 LPU 的数据仍必须经过 DMA、C2C、SR
和 MEM Write。

默认平台采用 500 MHz LPU 和双通道 DDR4-3200，51.2 GB/s 峰值在模型中表示为
读写共享的 102.4 bytes/cycle 小数 token budget。默认读延迟为 35 cycle 加
0..15 cycle 确定性抖动，写延迟为 25 cycle 加 0..10 cycle。queue depth、通道数、
带宽、基础延迟、抖动范围和 seed 均可配置。
