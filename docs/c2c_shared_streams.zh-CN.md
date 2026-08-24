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

旧的直接 SRAM 路径仅由 `c2c_dedicated_streams=true` 保留给对照测试，部署默认
使用共享路径。`c2c_dma_ddr4_test` 覆盖单 lane 和 8 lane 共享接收，并在 MEM 消费
后逐字节检查 SRAM。
