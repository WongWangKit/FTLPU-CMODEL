# C2C 共享 SR 接收通路

外部 C2C 接口每方向可配置 1..8 条 lane，每条 lane 每 cycle 搬运一个完整的
32-byte vector。外部 lane ID 与普通 SR ID 相互独立。

默认配置 `c2c_dedicated_streams=false` 下，RX 不直接写 SRAM。接收到的 vector
在 MEM/C2C 边界注入指定的普通 west stream；4 个 tile segment 形成每拍前进一个
tile 的 wavefront，沿共享 fabric 传播，最终由目标 slice 的普通 MEM `Write` 消费：

```text
DDR4 -> DMA lane -> C2C RX -> west SR -> MEM Write -> SRAM
```

第一个 vector 可以通知目标 `(hemisphere, slice, bank)` ICU queue，使 `Sync` 在
数据开始进入 fabric 时释放。软件随后必须按目标 MEM group 的 SR hop 数插入 NOP，
并在对应 cycle 发出 `Write`/`Repeat`。多条 C2C lane 可以同拍写不同 slice；普通
SR 多生产者冲突和 MEM 端口冲突仍会报错。

`c2c_dedicated_streams=true` 提供 32 条普通 SR 之外的独立外部 lane。该模式下 RX
指令携带目标 hemisphere/slice/bank/row，由目标 MEM 接收端提交 SRAM，不占普通 SR；
当前 `ModelSession` 用它做细粒度权重预取。无论采用专用还是共享模式，数据源都必须
是 DDR DMA/C2C，host 都不能直接写 LPU MEM。`c2c_dma_ddr4_test` 覆盖单 lane、8 lane
共享接收和专用接收，并逐字节检查 SRAM。

## DDR 时序模型

DDR 是外部 backing store，不是初始化 LPU MEM 的旁路。测试或 runtime 可以调用
`Ddr4Model::initialize_vector/read_vector` 表示 host 侧访问 DDR，但数据只有经过
DMA 和 C2C 才能跨越 LPU 边界。建模的输入和输出路径是：

```text
host <-> DDR4 <-> DMA <-> C2C <-> [dedicated RX 或 SR/MEM] <-> SRAM
```

默认平台采用 500 MHz LPU 和双通道 DDR4-3200，总峰值 51.2 GB/s，在模型中表现
为读写共享、可累积小数余量的 102.4 bytes/cycle token budget。默认读延迟为
35 cycle（70 ns）加 0..15 cycle 抖动；写延迟为 25 cycle（50 ns）加 0..10
cycle 抖动。抖动由 request ID、地址、读写方向和 seed 确定性生成，因此不同请求
在不同 cycle 完成，回归测试又能稳定复现。queue depth、channel 数、带宽、基础
延迟、抖动范围和 seed 都可以配置。
