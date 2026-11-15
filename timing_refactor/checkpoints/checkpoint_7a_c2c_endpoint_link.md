# Checkpoint 7A：C2C Endpoint 与 Link

## 范围与结论

本检查点只迁移片上 `C2cEndpoint`（TX/RX）和芯片间 `C2cLink`。ICU 按要求暂缓，
`C2cDmaEngine`、`Ddr4Model` 和跨芯片统一 Clock 不在本批范围。结论为 PASS；既有
RMSNorm/slow 冻结基线除外。

## 状态边界

- TX：待发送 Queue、四 tile gather、完成 FIFO、indexed stream 状态和 MEM read request。
- RX：Receive Queue、多向量 active context、ingress、四 tile replay 和完成计数。
- Link：serializer、flight queue、RX-ready FIFO、outstanding credit 和 cycle。
- TSP：永久 `C2cEndpointClockSlot` 注册到全局 Clock；SystemPhase 独占 Endpoint LOW。

LOW 只从 Current 生成 Next；所有已注册模块先通过 `validate_posedge()`，然后提交。
HIGH 不重复发送、接收、Stream consume/write 或 pipeline 推进。

## RTL 核对边界

本地 VMODEL 的 `c2c_tx_gather.sv`、`c2c_rx_replay.sv`、
`c2c_vector_credit_serializer.sv`、`c2c_rx_ready_fifo.sv` 和
`c2c_peer_transport.sv` 支持 gather/replay、credit、FIFO 和传输级是正沿状态的分类。
`dma_store_lane_context.sv` 只是 descriptor context holder，不足以证明完整 C2C DMA 数据
通路；instruction-load DMA 也未被当作 C2C 数据 DMA 证据。

## 验证结果

| 验证 | 结果 |
|---|---|
| 独立 Release 构建 `out/checkpoint_7a_c2c` | PASS |
| `subsystem.c2c.clocked` | PASS |
| C2C 标签 | 5/5 PASS |
| fast（排除冻结 RMSNorm） | 64/64 PASS，18.82 s |
| RMSNorm | 同一 `VXM input Buffer cannot be reconfigured while a Bundle is active` |
| slow 基线 | 6/6 保持 Checkpoint 5 相同失败文本/类别 |

专项测试验证了 LOW 后 Current 不提前变化、HIGH 无重复推进、validation failure 无部分
提交，以及反向 Clock 注册顺序不改变 SystemPhase 的 fabric→Endpoint 组合执行顺序。

## 未完成项

1. DMA/DDR 仍由 legacy `tick()` 推进。
2. DMA completion 到 ICU notify 仍是 LOW 中的旧回调边界。
3. `DualChipC2cSystem` 仍顺序完成两个 TSP tick，再推进 Link；尚无跨芯片单一 barrier。
4. 尚未建立 CModel/RTL 逐周期 Trace 等价证明。

Checkpoint 7A 到此停止。
