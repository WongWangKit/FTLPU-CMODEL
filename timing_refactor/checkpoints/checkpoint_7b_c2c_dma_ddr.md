# Checkpoint 7B：C2C Data DMA 与 DDR4

## 范围与结论

本检查点只迁移 `C2cDmaEngine`、`Ddr4Model` 及它们在单芯片 `TspSliceSystem` 中的时钟
接线。ProgramLoader 的 instruction-load DMA 保持独立软件路径；`DualChipC2cSystem` 的
跨芯片统一 Clock 留到 7C。结论为 PASS；既有 RMSNorm/slow 冻结基线除外。

## 状态与调用边界

- DMA：descriptor Queue、active transfer、TX/RX FIFO、DDR pending、completion token、
  cycle/trace 均由 Current/Next 管理。
- DDR：request/active/completion、latency、带宽、round-robin、cycle/trace 使用 Current/Next。
- DDR Memory：`pages_` 不逐拍复制；LOW 仅暂存局部 byte write，POSEDGE 才写入。
- TSP：East/West DMA 与共享 DDR 使用永久 Clock slot，避免 attach/detach 改变注册对象地址。
- LOW 顺序：Endpoint interaction → data DMA → completion notify ICU Next → DDR → ICU barrier。
- POSEDGE：DMA、DDR、ICU、Stream Fabric 和各 FU 先全部验证，再统一提交。
- HIGH：只观察新 Current，不重复 DMA、DDR、notify 或 Memory write。

`C2cDmaSystem::tick()` 仍先推进 ProgramLoader，使 instruction-load DMA 的 DDR 请求能在
本周期 DDR LOW 被采样；随后 TSP 执行统一周期。旧的尾部 `ddr4.tick()` 已删除。

## 验证结果

| 验证 | 结果 |
|---|---|
| 独立 Release 构建 `out/checkpoint_7b_dma_ddr` | PASS，76 项测试已生成 |
| `subsystem.c2c.dma_clocked` | PASS |
| fast（排除冻结 RMSNorm） | 66/66 PASS，27.72 s |
| 未归类配置/VXM | 3/3 PASS |
| RMSNorm | 同一 `VXM input Buffer cannot be reconfigured while a Bundle is active` |
| slow 基线 | 6/6 保持 Checkpoint 6 相同失败文本/类别 |

专项测试证明 DDR Memory 写和 completion 在 LOW 不可见、DMA/DDR 在同一 POSEDGE 可见、
HIGH 不重复推进、Clock 注册顺序不替代 SystemPhase 顺序，并且任一模块验证失败时 DMA、
DDR counter/completion 和稀疏 Memory 都不发生部分提交。

## 未完成项

1. `DualChipC2cSystem` 仍未形成覆盖两个 TSP 与 Link 的单一验证/提交屏障。
2. ProgramLoader 内部 DMA 是排除项，尚未迁移为统一 ClockedModule；它只通过共享 DDR
   输入窗口与本检查点接轨。
3. 尚未建立 DMA/DDR 与 RTL 的逐周期 Trace 等价证明。

Checkpoint 7B 到此停止。
