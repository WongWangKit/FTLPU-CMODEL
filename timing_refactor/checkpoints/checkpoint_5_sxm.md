# Checkpoint 5：SXM 统一时序迁移

## 结论

Checkpoint 5 已完成 TSP 主系统 `SxmSlice` 的状态映射、正式迁移和回归。East/West 两个
SXM Slice 现在与 Fabric、MEM、MXM、VXM 处于同一全局验证/提交屏障；LOW 只生成完整
层次 Next，POSEDGE 才整体提交，HIGH 不重复执行 Stream consume、Transpose capture、
Permute route、fanout 或 cycle 推进。

ICU 仍按用户要求暂缓。本结论是 CModel 行为契约验证，不是完整 RTL 逐周期 Golden。

## 状态映射

| 层级 | 跨周期状态 | 迁移方式 |
|---|---|---|
| Slice entry | 当拍 ICU issue | pending 容器；prepare 时合并到 Next |
| UnitGroup | 简单流操作 issue、stream-op count、cycle | Slice 层次 Current/Next |
| Local Transpose | 四 tile instruction wave、两 byte-plane Bank、dst/mask/ready/cycle | Slice 层次 Current/Next |
| Permute | 单拍命令、Bank release、输出路由 | LOW 读取旧 Bank、只修改 Next/Fabric Next |
| Fanout | 延迟输出队列 | 到期项 LOW 发送，队列正沿提交 |
| Wide Transpose | jobs、alignment FIFO、双 Bank payload、last issue | Slice 层次 Current/Next |
| Trace | cycle events、enable | 软件观测状态，不冒充 DFF |

关键同拍规则得到保留：Permute 可读取并释放旧 ready block，Transpose capture 在同一个
LOW 中只写 Next，因此不会提前覆盖 Current；全局验证失败时两者都不提交。

## VMODEL 证据与范围

- `rtl/sxm/lpu_sxm_control.sv:23-36`：Permute 控制一拍寄存。
- `rtl/sxm/lpu_sxm_slice.sv:269-349`：旧 Bank 先供 Permute，随后 capture；Bank、tile ready、
  pass-through 边界数据均在 `always_ff` 更新。
- `rtl/sxm_native/sxm_permute_engine.sv:3-9,53-137`：Native Permute 是组合选择/释放逻辑。

CModel 的行式 capture、fanout skew 和 WideStreaming 试验路径比 integrated RTL 接口更广，
本批只确认状态类别和边沿边界，没有宣称逐对象同构。

独立 `SxmMatrixCore/SxmMatrixMemReference` 是三角矩阵/WriteTap 固定日历参考路径，不是
`TspSliceSystem` 的主 Slice。本批保留它已有独立 Clock/SystemPhase 驱动和执行语义，
其两项专项测试继续通过。

## 验证结果

| 项目 | 结果 |
|---|---|
| 独立 Release 构建 | 74 个实际项目，73 个 `.exe`，0 missing |
| `unit.sxm.clocked` | PASS |
| SXM 标签 | 10/10 PASS |
| fast（排除冻结 RMSNorm） | 63/63 PASS，18.50 s |
| RMSNorm | 同一既有 active-bundle 错误 |
| slow 基线 | 6/6 保持 Checkpoint 4 相同失败文本/类别 |

专项测试覆盖：正沿前后 Current 可见性、正反 Clock 注册顺序等价、HIGH 不重复推进、
validation barrier 失败前无 SXM/Fabric 部分提交。

slow 六项仍分别停在四个 MXM accumulator 配置容量边界、一个 MEM SRAM 配置容量边界和
一个 VXM active-bundle 边界；本批未修改配置来掩盖问题。

## 修改文件

- `CMakeLists.txt`
- `include/ftlpu/sxm/slice.hpp`
- `include/ftlpu/system/tsp_slice_system.hpp`
- `tests/unit/sxm/slice_test.cpp`
- `tests/unit/sxm/local_blocks_timing_test.cpp`
- `tests/unit/sxm/wide_transpose_test.cpp`
- `tests/unit/sxm/sxm_clocked_test.cpp`
- 五份既有统一时序文档和本 Checkpoint 证据/清单。

## 停止点

Checkpoint 5 完成。原始 `FTLPU-CMODEL-main`、SOFTWARE、VMODEL 均未修改；没有 Git
commit/push。ICU 继续暂缓，未开始 Checkpoint 6。
