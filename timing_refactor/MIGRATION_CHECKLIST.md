# 统一时序迁移检查清单

## 工作区与 Checkpoint 1

- [x] 从 `D:\LPU\FTLPU-CMODEL-main` 建立完整独立工作区。
- [x] 目标为 `D:\LPU\FTLPU-CMODEL-clock`。
- [x] 原始目录无 `.git` 元数据，不虚构 commit 或 diff。
- [x] 排除旧根目录 `out`，保留全部源码、测试、文档和 clock_experiment 资产。
- [x] 复制前后 861 个基线文件逐文件 SHA-256 一致。
- [x] 23 个 Checkpoint 1 关键文件 SHA-256 一致。
- [x] Clock/SystemPhase 调用权保持冻结架构。
- [x] StreamRegisterFabric 稀疏 Current/Next 与双 Hemisphere 提交屏障保留。
- [x] 新工作区独立 CMake 配置成功。
- [x] Release 全量干净编译成功。
- [x] Checkpoint 1 专项测试 5/5 通过。
- [x] 除既有 RMSNorm 失败外，fast 回归 59/59 通过。
- [x] RMSNorm 仍以冻结基线中的同一错误失败。

## 后续顺序

- [x] Checkpoint 2：MEM 状态映射、迁移、专项测试和系统回归。
- [x] Checkpoint 3：MXM。
- [x] Checkpoint 4：VXM，并重新核对 RMSNorm 基线。
- [x] Checkpoint 5：SXM 主系统 Slice。
- [x] Checkpoint 6：ICU。
- [ ] Checkpoint 7：C2C、DMA 和剩余状态（7A Endpoint/Link、7B data DMA/DDR 已完成；
  7C 跨芯片统一 Clock 尚未开始）。
- [ ] 全系统周期级审计和 FIFO 深度统计准备。

## Checkpoint 2：MEM

- [x] 迁移前完成 SRAM、R1 响应、指令流水、Queue、Trace 和周期计数分类。
- [x] `MemArrayModel` 接入 `ClockedModule` 和全局验证/提交屏障。
- [x] 指令流水、R1 响应和 MEM 周期使用 `ClockedState`。
- [x] SRAM 保持稀疏页式 Memory，仅暂存本周期局部写，未复制完整 Memory。
- [x] SRAM 写、transport commit counter 和 Queue pop 移到 POSEDGE。
- [x] `ReadTiming::Main` 与 `RegisteredR1` 的不同延迟继续保留。
- [x] TspSliceSystem 的 East/West MEM 与 Fabric 进入同一 Clock 提交集合。
- [x] MEM 单元测试 8/8 通过。
- [x] 受影响 TSP/SXM 子系统测试 4/4 通过。
- [x] 排除既有 RMSNorm 后，fast 回归 60/60 通过。
- [x] 原始 `FTLPU-CMODEL-main` 861 个基线文件哈希变化为 0。
- [ ] 用可靠 RTL 或 SRAM Macro 规格确认真实端口数量、冲突规则和物理读延迟。

Checkpoint 2 的停止点已由 Checkpoint 3 取代。

## Checkpoint 3：MXM

- [x] 核对当前顶层集成 legacy `lpu_mxm_slice`，并与未集成 `lpu_mxm_core` 契约分离。
- [x] Load/Dequant/Compute 四级 row pipeline 和 cycle 转为 `ClockedState`。
- [x] Queue front 只在 LOW 接受，统一 POSEDGE 后 pop；row0 不再零拍执行。
- [x] Supercell Weight Buffer、valid 和 Activation Pipeline 使用 Current/Next。
- [x] Datapath WS/IS pipeline、cursor、valid/owner 和输出状态纳入统一提交。
- [x] Accumulator 使用局部 segment 操作暂存，未复制整个大数组。
- [x] MXM Control 与 Datapath 注册到 Tsp 全局 Clock，SystemPhase 调用顺序保持显式。
- [x] 新增 `unit.mxm.clocked`，覆盖 row0/四级传播、边沿可见性、注册顺序无关和无半提交。
- [x] MXM 标签测试 9/9 通过。
- [x] 排除既有 RMSNorm 后，fast 回归 61/61 通过。
- [x] 慢速套件用 Checkpoint 2 独立基线对照；5 项配置/VXM 失败保持一致，Decoder 的旧权重时序失败已越过，随后到达同一 128-row accumulator 配置边界。
- [x] 原始 `FTLPU-CMODEL-main` 文件清单与 SHA-256 复核无变化。
- [ ] 用当前集成 legacy MXM RTL 建立逐周期 Trace 对照；本批只有源码契约核对和 C++ 独立不变式验证。

## Checkpoint 4：VXM

- [x] 核对 CModel Slice/Superlane/Lane、Input Buffer、Decoder/Config 和流水状态。
- [x] 核对本地 VMODEL local/global control wave、execute wave 与 tile resident state。
- [x] `VxmSlice` 接入 `ClockedModule` 和 TSP 全局验证/提交屏障。
- [x] 8 条 local Queue、tile row pipeline、Superlane Config/Decoder、Input Bundle 与 Lane
  datapath 形成一个事务化层次 Current/Next。
- [x] ICU 当拍 issue 在 `prepare_cycle()` 前暂存，LOW 后不再允许追加，避免污染 Current。
- [x] VXM 输出仍在 LOW 写入 Fabric Next；HIGH 不重复 capture、decode、execute 或 release。
- [x] 软件 Statistics 与硬件状态分离，避免复制增长中的 timeline。
- [x] 新增 `unit.vxm.clocked`，覆盖边沿可见性、Clock 注册顺序无关、验证失败无半提交、
  失败 LOW 不污染 Current。
- [x] 独立 Release 构建目录 `out/checkpoint_4_vxm`：73 个实际项目、72 个可执行文件，0 missing。
- [x] VXM 标签 14/14 PASS。
- [x] 排除既有 RMSNorm 后，fast 回归 62/62 PASS。
- [x] RMSNorm 仍为同一 active-bundle 错误；slow 6 项失败类别与 Checkpoint 3 一致。
- [x] 权威 CMODEL 仍为 `main@2351cec89b5ef9c3c73f97b2d2e77c458fb36c95`；每日同步 Work
  随后把原始目录建立为该 SHA 的 clean Git 工作树。Checkpoint 4 未向原始目录写源码。
- [ ] 建立 VXM RTL/CModel 逐周期 Trace；当前仅有 RTL 状态边界源码核对和 C++ 不变式验证。

## Checkpoint 5：SXM

- [x] 先完成 `SxmSlice`、Local Transpose/Permute、UnitGroup、Wide Transpose 和独立
  Matrix 参考路径的状态分类，再确定主系统迁移边界。
- [x] 核对 VMODEL `lpu_sxm_control.sv`、`lpu_sxm_slice.sv` 的控制波、Permute 寄存、
  Transpose Bank/tile ready 和边界寄存状态。
- [x] `SxmSlice` 接入 `ClockedModule`，East/West 两实例加入 TSP 全局验证/提交屏障。
- [x] ICU issue 在 LOW 前进入 pending，prepare 后禁止追加；不直接污染已提交 Current。
- [x] Transpose instruction wave、Bank/metadata、Permute、fanout、UnitGroup cycle 和 Wide
  job/FIFO/双 Bank payload 形成一个层次事务 Current/Next。
- [x] SXM 对 Fabric 的 consume/write 只在 LOW 执行一次；HIGH 不重复副作用。
- [x] `unit.sxm.clocked` 覆盖正沿前后可见性、Clock 注册顺序无关、HIGH 不重复推进和
  validation barrier 失败前无 SXM/Fabric 半提交。
- [x] 独立 Release 构建目录 `out/checkpoint_5_sxm`：74 个实际项目、73 个可执行文件，
  0 missing。
- [x] SXM 标签 10/10 PASS。
- [x] 排除既有 RMSNorm 后，fast 回归 63/63 PASS，18.50 秒。
- [x] RMSNorm 保持同一 active-bundle 错误；slow 6 项失败文本/类别与 Checkpoint 4 一致。
- [x] 独立 `SxmMatrixCore/SxmMatrixMemReference` 保持已有固定 P/WriteTap 日历，作为
  非 TSP 主 Slice 的专项参考路径；相关 2 项测试继续通过。
- [ ] 建立 integrated SXM RTL/CModel 逐周期 Trace；当前只有 RTL 寄存边界核对和 C++
  不变式/系统回归证据。

## Checkpoint 6：ICU

- [x] 对 local iMEM、IQ、pending fetch、Macro reservoir/parser/DDB/active context、QCTL、
  Barrier、Queue completion、事件和 cycle 完成状态所有权分类。
- [x] `InstructionControlUnit` 接入 `ClockedModule` 和 TSP 全局验证/提交屏障；ICU LOW
  仍由 SystemPhase 的显式 dispatch 节点唯一调用。
- [x] LOW 从完整 Current 建立事务 Next，只在 Next 上执行 Queue fetch/decode/issue、QCTL、
  Barrier 和事件推进；POSEDGE 验证后原子提交，HIGH 不重复执行。
- [x] local iMEM 与 Macro 原始映像按 Memory 分类，Current/Next 共享只读 backing；只有
  配置写触发 Copy-on-Write，未把完整程序映像当作触发器逐拍复制。
- [x] Macro DDB 和 active context 存储采用深拷贝快照；QCTL shared-pool 指针及 MEM
  transport executor 在快照/提交后显式重绑定。
- [x] 周期外到达的通知先进入 Pending-event；LOW 采样到 Next。Barrier due 与 Queue
  completion 的组合查询可预览 Pending-event，但不会提前修改 Current。
- [x] 新增 `unit.icu.clocked`，覆盖 LOW/POSEDGE 可见性、HIGH 无重复推进、外部通知采样和
  全局 validation failure 无 ICU 部分提交。
- [x] 独立 Release 构建目录 `out/checkpoint_6_icu` 全量编译成功，共 75 项测试。
- [x] ICU 标签 8/8 PASS；C2C 标签 5/5 PASS；三个未归类配置/VXM 测试 3/3 PASS。
- [x] 排除冻结 RMSNorm 后，最终 fast 回归 65/65 PASS，14.26 秒。
- [x] RMSNorm 保持同一 active-bundle 错误；slow 6 项保持四个 MXM accumulator 容量、
  一个 MEM SRAM 容量和一个 VXM active-bundle 失败类别。
- [x] instruction-fetch DMA 未与 ICU local fetch 或 C2C data DMA 混同；本检查点未迁移
  `C2cDmaEngine`/`Ddr4Model`。
- [ ] 建立 ICU 与当前 RTL 的逐周期 fetch/decode/issue Trace；当前结论仅覆盖 CModel
  时钟事务和既有行为回归。

## Checkpoint 7A：C2C Endpoint 与 Link

- [x] 迁移前区分片上 Endpoint、芯片间 Link、外部 C2C DMA/DDR 三层状态所有权。
- [x] `C2cTxSlice` 的 Queue、四 tile gather、完成 FIFO 和 MEM read request 改为
  Current/Next；ICU issue 先进入 pending。
- [x] `C2cRxSlice` 的 Queue、active receive、ingress、四 tile replay 和完成计数改为
  Current/Next；外部通知仍在 LOW 交给尚未迁移的 ICU。
- [x] 可选 `C2cEndpoint` 通过永久 Clock slot 加入 TSP 全局验证/提交屏障。
- [x] `C2cLink` 的 serializer、flight、RX-ready FIFO、credit 和 cycle 统一在 POSEDGE 提交。
- [x] 新增 `subsystem.c2c.clocked`，覆盖 LOW/POSEDGE/HIGH、验证失败无半提交，以及
  SystemPhase 顺序独立于 Clock 注册顺序。
- [x] C2C 标签 5/5 PASS；fast（排除冻结 RMSNorm）64/64 PASS。
- [x] RMSNorm 和 slow 6 项保持 Checkpoint 5 的同一冻结失败文本/类别。
- [x] `C2cDmaEngine` 与 `Ddr4Model` 已在 Checkpoint 7B 建立 Current/Next，并与 TSP
  统一 POSEDGE；DMA completion/notify 在同一 LOW 写入 ICU Next。
- [ ] Dual-chip 顶层仍以两个 TSP 完整 tick 后再 tick Link；尚未形成跨芯片单一 Clock。

## Checkpoint 7B：C2C Data DMA 与 DDR4

- [x] 明确区分 C2C data DMA 与 ProgramLoader 内部 instruction-load DMA；后者保持现有
  软件控制状态机，未冒充 data DMA 的硬件迁移证据。
- [x] `C2cDmaEngine` 的 descriptor Queue、active transfer、TX/RX FIFO、DDR outstanding、
  completion token、cycle/trace 状态形成 Current/Next。
- [x] Endpoint 在 LOW 产生的 send/pop 先暂存，DMA LOW 只推进 Next；POSEDGE 后才改变
  可观察 FIFO、completion 和 cycle。
- [x] `Ddr4Model` 的 request/active/completion、带宽仲裁、latency、cycle/trace 状态形成
  Current/Next。
- [x] DDR 稀疏 `pages_` 保持 Memory；LOW 只记录局部 byte write，统一 POSEDGE 才原地提交。
- [x] TSP 为 East/West data DMA 和共享 DDR 使用永久 Clock slot；SystemPhase 显式保持
  Endpoint interaction → DMA LOW → completion notify → DDR LOW → ICU barrier 的依赖顺序。
- [x] `C2cDmaSystem` 不再在 TSP 完整周期之后额外 `ddr4.tick()`；共享 DDR 只推进一次。
- [x] ProgramLoader 仍在 TSP LOW 前推进，使其内部 DMA 请求进入同一个共享 DDR LOW；
  instruction loading 与 C2C data transport 没有合并为同一 DMA 对象。
- [x] 新增 `subsystem.c2c.dma_clocked`，覆盖 LOW/POSEDGE/HIGH、反向 Clock 注册顺序、
  DMA/DDR 同一提交边界和 validation failure 时稀疏 Memory 无部分写入。
- [x] 独立 Release 构建 `out/checkpoint_7b_dma_ddr` 成功；fast 排除 RMSNorm 为 66/66，
  未归类配置/VXM 为 3/3。
- [x] RMSNorm 与 slow 6 项保持 Checkpoint 6 相同失败文本/类别。
- [ ] `DualChipC2cSystem` 仍按两个 TSP 和 Link 的分离时钟边界推进；留待 Checkpoint 7C。

当前停止点为 Checkpoint 7B 完成。跨芯片统一 Clock barrier 和最终全系统周期级审计尚未开始。

