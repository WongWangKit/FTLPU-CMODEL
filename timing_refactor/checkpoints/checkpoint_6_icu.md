# Checkpoint 6：ICU

## 范围与结论

本检查点迁移 `InstructionControlUnit` 及其 local Queue runtime：IQ、pending fetch、Macro
frontend、QCTL、Barrier、Queue completion、事件和周期状态。local iMEM 程序映像保留为
Memory；instruction-fetch DMA、C2C data DMA/DDR 与跨芯片统一 Clock 不在本批范围。

结论为 PASS。冻结的 RMSNorm 和 slow 资源/容量失败除外，没有新增失败类别。

## 状态与提交边界

- `InstructionControlUnit` 注册到 TSP 全局 Clock，但 LOW 仍只由 SystemPhase 的 ICU dispatch
  节点调用一次。
- LOW 复制 Current 为事务 Next，在 Next 上执行 fetch、decode、Macro 展开、Queue pop、
  QCTL retire、Barrier/Completion 和 FU issue。
- 全部 Clock participant 先通过 `validate_posedge()`，然后 ICU 才原子提交 Next。
- HIGH 只观察提交后的 Current，不重新 fetch、decode、issue 或推进任何 Queue。
- local semantic/raw iMEM 与 Macro raw image 共享不可变 backing；配置写时 Copy-on-Write。
- DDB 与 active Macro context 深拷贝；QCTL shared-pool 指针和 MEM transport 回调在快照及
  提交后重绑定。
- 周期外 notify/completion 进入 Pending-event。LOW 将其应用到 Next；due/ready 查询可以
  无副作用预览 Pending-event，避免丢失同拍 Barrier 安全检查或 Queue 完成握手。

## 专项验证

新增 `unit.icu.clocked`，验证：

1. dispatch LOW 后 ICU Current、Queue cycle 仍保持旧值。
2. POSEDGE 后完整 ICU Next 同时可见。
3. HIGH 不重复推进 Queue 或全局 cycle。
4. 外部 notification 在 LOW/POSEDGE 前不修改 Current，提交后正常释放 Sync。
5. 其他 Clock participant 验证失败时，ICU 不先行提交。

旧 `unit.icu.common_barrier` 与 `unit.icu.queue_control` 继续验证 Pending-event 的组合投影：
ready completion 可供 LOW 前查询，但 controller Current 仍只在正沿改变。

## 回归结果

| 验证 | 结果 |
|---|---|
| Release 全量构建 `out/checkpoint_6_icu` | PASS，75 项测试 |
| ICU 标签 | 8/8 PASS |
| C2C 标签 | 5/5 PASS |
| 未归类配置/VXM 测试 | 3/3 PASS |
| fast（排除冻结 RMSNorm） | 65/65 PASS，14.26 s |
| RMSNorm | 同一 active-bundle 失败 |
| slow | 6 项均保持既有容量/资源失败类别 |

slow 六项具体分类为：四个 MXM accumulator 配置容量、一个 MEM SRAM 配置容量、一个
VXM active-bundle。它们不是本次 ICU 迁移产生的新回归。

## 未完成项

1. 尚未建立 ICU CModel/RTL 的逐周期 fetch/decode/issue Trace。
2. `C2cDmaEngine`、`Ddr4Model` 仍为 legacy 周期推进。
3. instruction-fetch DMA 仍是独立的外部程序装载路径，本批未迁移。
4. 双芯片系统尚无覆盖两个 TSP 与所有 Link 的单一验证/提交 barrier。

Checkpoint 6 到此停止。
