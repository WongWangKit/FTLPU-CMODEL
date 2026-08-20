# C2C 独立 Stream 数据面

CModel 保留原有每方向 32 条计算 stream，并额外提供独立 C2C stream：默认
8 条 eastward 和 8 条 westward。`SystemHardwareConfiguration` 的
`c2c_streams_per_direction` 可选择物理上支持的子集，且不会改变普通
`StreamId` 的 0..31 编码。

每条 C2C lane 每 cycle 搬运一个完整的 32-byte vector，所以默认每个方向峰值
带宽为 `8 x 32 = 256 bytes/cycle`。DDR 请求可以流水并行，DMA 为每条 lane
分别维护 command、outstanding request 和 RX FIFO。

DMA Load 的接收路径为：

```text
DDR -> per-lane DMA FIFO -> dedicated C2C RX -> target MEM slice/bank/row
```

它不经过普通 westward stream-register file。C2C receive instruction 用一个
burst 描述 `base_row + vector_index * row_stride`，接收端每 cycle 最多提交每条
lane 一个 vector。这样 bank 0 的计算读取可与 bank 1 的权重写入重叠，并且
VXM/RMSNorm 可以继续使用全部 32 条普通 stream。

`c2c_dma_ddr4_test` 同时覆盖 8 个 32-byte vector 在一个 cycle 完成、4-lane
参数限制，以及全部 32 条 westward 计算 stream 发射时 C2C 仍能前进并正确写入
另一 SRAM bank。
