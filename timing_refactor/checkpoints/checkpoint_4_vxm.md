# Checkpoint 4：VXM 统一时序迁移

## 结论

Checkpoint 4 已完成 VXM 状态映射、正式源码迁移和相关回归。`VxmSlice` 现在加入 TSP
全局 Clock 的统一验证/提交集合；LOW 只准备一个完整层次 Next，POSEDGE 才整体提交，
HIGH 不再重复执行任何 VXM 副作用。

这证明当前 CModel VXM 的已定义行为遵守统一边沿契约，不等于已经与完整 RTL 建立逐周期
Golden。ICU 未在本批迁移。

## 状态映射与设计

| 层级 | 跨周期状态 | 迁移方式 |
|---|---|---|
| Slice | 8 条入口 Queue、4-row instruction wave、cycle | 层次事务 Current/Next |
| Superlane control | 3-entry FIFO、Decoder、Next/Current Config、repeat | 随 Slice Next 深拷贝并求值 |
| Input Buffer | lane bytes、occupied、expected/fill count、Empty/Collecting/Ready | capture/release 只修改 Next |
| Lane datapath | input token、Basic/LUT pipeline、accumulator、output、feedback holding | 随层次 Next 整体提交 |
| Quantizer | request/pending one-cycle state | 随 Lane Next 每拍推进一次 |
| Statistics/Trace | counters、timeline | 软件观测状态，共享；不冒充 DFF |

入口 `issue_south()` 在 VXM prepare 前进入 `pending_issues_`，`prepare_cycle()` 才将其合并
到 Next。这样 ICU dispatch 仍保持原有 SystemPhase 顺序，但不会直接改变已提交 Current。
`evaluate()` 在事务副本中执行 Decoder、Config、Lane 和 Bundle 更新；全局 validation 全部
成功后，`posedge()` 只做不可抛异常的状态交换。

## LOW/POSEDGE/HIGH 接线

1. LOW：Fabric 打开 staging，ICU dispatch，VXM `prepare_cycle()`。
2. LOW：从 Fabric Current 读取完整输入组并写 VXM Next Input Buffer。
3. LOW：VXM `evaluate()` 推进 Decoder/Config/Datapath，产生输出视图。
4. LOW：VXM 输出写 Fabric Next；静态调度错误在提交前抛出。
5. POSEDGE：Clock 先验证 Fabric/MEM/MXM/VXM，全部通过后统一提交。
6. HIGH：只观察提交后的 Current，不再 capture、execute、release 或 issue。

直接 VXM 单元测试保留兼容 `tick()`，该包装内部执行一个完整的
prepare→evaluate→validate→commit，不是系统中的第二套时钟。

## VMODEL 证据和限制

- `rtl/vxm/lpu_vxm_control.sv:53-98`：global 与 8 条 local control pipeline。
- `rtl/vxm/lpu_vxm_slice.sv:117-168`：row config fire、`tile_execute_wave_q` 一拍寄存、
  config load 和 execute wave 分离。
- `rtl/vxm/lpu_vxm_tile_execution.sv:281-412,691-1057`：resident config、pending valid、
  input token、accumulator、feedback、output pending 的正沿状态。

当前硬件配置 `ftlpu-lpu32.json` 为 4 tiles、8 lanes/tile。CModel 使用 Superlane compact
decoder；integrated VMODEL 使用 resident local/global config 和 execute wave。两者状态
类别一致但结构并非逐对象同构，因此本批没有虚构 RTL cycle-by-cycle equivalence。

## 验证结果

| 项目 | 结果 |
|---|---|
| 独立 Release 构建 | 73 个实际项目，72 个 `.exe`，0 missing |
| `unit.vxm.clocked` | PASS |
| VXM 标签 | 14/14 PASS |
| fast（排除冻结 RMSNorm） | 62/62 PASS，16.81 s |
| RMSNorm | 同一既有 active-bundle 错误 |
| slow 基线 | 6/6 保持相同失败类别 |

专项测试覆盖：LOW 前后 Current 可见性、正反 Clock 注册顺序等价、validation barrier
失败前无 VXM 部分提交、失败 LOW 不提交 Current、HIGH 不重复推进。

RMSNorm 仍为：

`VXM input Buffer cannot be reconfigured while a Bundle is active`

该错误文本、测试阶段和 slow 中的 `decode_attention` 类别均与 Checkpoint 3 冻结基线一致；
本批未为消除该问题引入动态握手或修改静态 schedule。

## 修改文件

- `CMakeLists.txt`
- `include/ftlpu/vxm/lane.hpp`
- `include/ftlpu/vxm/superlane.hpp`
- `include/ftlpu/vxm/slice.hpp`
- `include/ftlpu/system/tsp_slice_system.hpp`
- `tests/unit/vxm/vxm_clocked_test.cpp`
- 五份既有统一时序文档和本 Checkpoint 证据/清单。

## 权威仓库与停止点

开始前已调用每日核查 Work。权威 CMODEL 仍为
`WongWangKit/FTLPU-CMODEL main@2351cec89b5ef9c3c73f97b2d2e77c458fb36c95`。
该 Work 后续把 `D:\LPU\FTLPU-CMODEL-main` 建立为这一 SHA 的 clean Git 工作树；本批只
写 `D:\LPU\FTLPU-CMODEL-clock`。

Checkpoint 4 完成。ICU 按用户要求暂缓，下一批为 Checkpoint 5 SXM。
