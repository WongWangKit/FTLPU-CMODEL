# Checkpoint 3：MXM 统一时序迁移

## 结论

Checkpoint 3 已完成当前集成 MXM 的状态映射、源码迁移和回归。Control、Supercell、
Datapath 和 Accumulator 已进入 TspSliceSystem 的全局 validate/commit 屏障；旧 C++
中 Queue front 在同一个 LOW 直接进入并执行 row0 的旁路已消除。

本结论针对当前 VMODEL 顶层仍在使用的 legacy `lpu_mxm_slice`。仓库中的新
`lpu_mxm_core` 尚未迁移进顶层，具有另一套更深流水和 Backend 约束，不能与本批混为
同一硬件契约。

## RTL 契约依据

- `FTLPU-VMODEL-main/rtl/mxm/lpu_mxm_slice.sv`：当前 slice 实例化
  `lpu_mxm_legacy_control`，使用独立 Load/Dequant/Compute 控制流。
- `FTLPU-VMODEL-main/rtl/mxm/legacy/lpu_mxm_legacy_control.sv`：三条控制流均进入
  4-row pipeline。
- `FTLPU-VMODEL-main/rtl/core/lpu_control_pipeline.sv`：row0 与后续 row 都在
  `posedge clk` 更新。
- `FTLPU-VMODEL-main/rtl/mxm/lpu_mxm_compute.sv`：sum/cursor/fault 为正沿状态，
  本地 dot/result 位于其前的组合路径。
- `FTLPU-VMODEL-main/docs/mxm_supercell.md`：新 frontend/core 独立，legacy slice/top
  尚未迁移。

这些证据用于选择寄存边界，没有执行 RTL 动态 Trace 对照，因此验证等级是源码契约
核对加 C++ 独立不变式，不是完整 L3 RTL Golden。

## 状态迁移

| 对象 | 迁移前 | 迁移后 |
|---|---|---|
| Control Load/Dequant/Compute rows | 普通数组在 `tick()` 内 dispatch、execute、原地 shift | 三个 `ClockedState`；LOW 只执行 Current，POSEDGE 整体提交 |
| Control Queue front | dispatch 时立即 pop | LOW 标记接受，全部模块验证通过后才 pop |
| Weight Buffer 与 valid | IW 中立即修改 | Supercell `ClockedState` Next，正沿后可见 |
| Activation Pipeline | `tick()` 内原地移动 | 从旧 Current 生成 Next，每边沿一级 |
| WS/IS Datapath | 多组容器和 cursor 在函数中立即更新 | `DatapathState` Current/Next，统一提交 |
| Accumulator | `accumulate/clear` 立即改大数组 | LOW 暂存局部 segment 操作，POSEDGE 原地提交 |
| MXM cycle | `tick()` 末立即增加 | `ClockedState<size_t>`，只在正沿增加 |

`MxmArray`/Supercell 的提交由 `MxmControlSlice` 所有；Datapath/Accumulator 由 `Mxm`
所有。Tsp 对两者分别注册，避免重复注册嵌套对象。

## 周期契约

1. ICU 在 LOW 把控制条目交给 MXM Queue；Control 只把它写入 row0 Next。
2. 第一个正沿后 row0 Current 才有效；下一 LOW 执行 row0 组合控制。
3. row0～row3 每个正沿只推进一级，不因 C++ 赋值顺序同周期穿越多级。
4. IW 在 LOW 计算目标 Weight Next，正沿前读取仍是旧 Weight/valid。
5. Datapath LOW 读取 Control Current pulse；Accumulator 同 LOW read 可看到本周期已接受
   操作的顺序投影，但大数组本体只在正沿更新。
6. Queue pop、Control rows、Weight/valid、Datapath 和 Accumulator 先全部验证，再统一
   提交；已知验证错误不会产生半提交。
7. HIGH 不准备 Next、不重复 issue/pop、不重复 MAC 或 Accumulator 写入。
8. 当前 legacy pipeline 没有 stall/hold 接口，本批保持每个有效正沿推进一级。

## 验证

| 验证项 | 结果 |
|---|---|
| Release 配置与全部受影响目标编译 | PASS |
| 新增 `unit.mxm.clocked` | PASS |
| MXM 标签 | 9/9 PASS |
| fast（排除既有 RMSNorm） | 61/61 PASS |
| RMSNorm 冻结基线 | 同一 VXM active-bundle 错误 |
| Checkpoint 2 慢速基线对照 | 6 项均为既有配置/VXM/调度阻塞 |
| Decoder Layer MXM 权重时序 | 旧到达偏移已越过；随后命中既有 128-row ACC 容量边界 |
| 原始 `FTLPU-CMODEL-main` | 文件清单与 SHA-256 无变化 |

`unit.mxm.clocked` 独立覆盖：

- Queue dispatch 后 row0 在正沿前不可见、正沿后可见。
- 指令在 4 个 row 间逐边沿推进。
- Supercell Weight/valid 只在正沿后可见。
- Accumulator LOW 投影与正沿后物理值一致。
- 正序/逆序 Clock 注册得到相同结果。
- 后续未准备模块令 validate barrier 失败时，已验证的 Supercell 也不提交。

慢速测试在 Checkpoint 2 的独立只读基线中已经全部失败：三项超出当前 128-row ACC，
一项超出当前 slice SRAM，一项是已知 VXM active-bundle，Decoder 原先先被 MXM 权重
周期偏移阻塞。本批修正 Decoder 的 legacy row0/Stream 延迟对齐后，它继续运行并到达
同一 128-row ACC 配置边界；未修改硬件配置来掩盖这些非本批问题。

## 保留功能与边界

- MXM 指令格式、数值算法和 FP16/BF16/INT8 精度路径未重新设计。
- 未修改 ISA、VMODEL 或原始 CModel。
- 未把未集成新 core 的 M1～A3/Backend 固定延迟加入 legacy CModel。
- 未确认真实 Weight SRAM/Accumulator Macro 的端口、冲突和物理 PPA。
- 当前 PASS 不代表整个 CModel 或新 MXM core 已经周期精确。

## 停止点

Checkpoint 3 已完成。按用户要求在此暂停；Checkpoint 4 VXM 尚未开始。
