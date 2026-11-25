# 时序重构变更记录

## 2026-11-19：FU local iMEM SRAM 同步与正式 MXM 直发

- 新增 `timing::Sram1Rw<T,Depth>`，按本地 RTL `lpu_sram` 建模单端口 1RW、
  同步注册读、写占用端口、输出保持，以及 reset 不清 memory 的合同。
- 新增独立 `timing::SramFifo<T,Depth>`；level、read-pending、SRAM return 和
  output register 分离，最多一个在途读，支持非 2 次幂回绕和输出反压保持。
- `LocalImemFrontendFetchModel` 参数化为 96/128 bit，并将 MEM 208 条与 MXM
  Load/Dequant/Compute 12 条正式 local iMEM 取字路径接到同一 SRAM 原语。
- Response FIFO、Macro Word FIFO、Context FIFO、Stream FIFO 等普通 FIFO 未替换；
  VXM/SXM 当前无正式 local iMEM，保持 `NOT_IMPLEMENTED`。
- 正式 MEM/MXM Scheduler 改为 typed valid/ready 直发；production 的
  `timed_issue_depth/lookahead` 均为 0。Ready 低时锁存 terminal fault，只有
  issue fire 才推进 Active Context。
- 新增 SRAM primitive、SRAM FIFO、MXM 128-bit local-iMEM 与 MXM ready-low
  定向测试；宿主 C++ 测试使用已保存 RTL 合同，不重跑 VCS。
- ProgramLoader 将 160-word 任意字节吞吐 fixture 明确为 Native 装载测试，另用
  合法 3-word MEM Raw-Macro fixture 验证 Program Distribution valid/ready/address/
  strobe/last；不再把任意尾随字节冒充可完整解码的 Macro 程序。
- 本轮 14 项受影响精确回归为 14/14 PASS；保存的 Level1 RTL Trace 与当前
  CModel Trace 比较为 410/410、0 mismatch。正式 Hardware Interface View 覆盖
  Program Load、local-iMEM read request/response、data/valid/last、backpressure、
  FIFO level、Decoder/Context/Scheduler/MEM issue。物理 SRAM macro pins 属于
  RTL-only Diagnostic，不要求 CModel 提供同名输出，也不作为本阶段验收门禁。

## 2026-11-14：独立工作区与 Checkpoint 1 迁移复验

### 工作区

- 源：`D:\LPU\FTLPU-CMODEL-main`
- 目标：`D:\LPU\FTLPU-CMODEL-clock`
- 源目录无 Git 元数据。
- 复制 861 个基线文件，复制前后 SHA-256 差异为 0。
- 唯一复制排除项：源根目录 `out`。新工作区建立独立
  `out/checkpoint_1_revalidation`。
- `clock_experiment` 整体保留，包括实验源码、CSV、比较器、编译日志和 VCS 日志。

### Checkpoint 1

- 未重新设计时钟架构；完整继承当前正式实现。
- 正式接口继续采用 `low()`、`validate_posedge()`、`posedge()`，没有恢复早期
  `begin_cycle()`、`commit_cycle()` API。
- Clock/SystemPhase 职责、双 Hemisphere 提交屏障和 Tsp LOW/HIGH 接线均已核实。

### 验证

- CMake 配置：PASS。
- Release 全量干净编译：PASS。
- Checkpoint 1 专项：5/5 PASS。
- fast 全集：59 PASS、1 个已知基线失败。
- 排除 `kernel.rmsnorm` 后：59/59 PASS。
- 已知失败：`VXM input Buffer cannot be reconfigured while a Bundle is active`。
  错误文本与冻结基线一致，本批未修改 VXM 功能来掩盖问题。

### 证据

- `timing_refactor/manifests/source_manifest_pre_copy.csv`
- `timing_refactor/manifests/workspace_manifest_post_copy.csv`
- `timing_refactor/manifests/checkpoint_1_key_hashes.csv`
- `timing_refactor/checkpoints/checkpoint_1_evidence/`

## 2026-11-14：Checkpoint 2 MEM

### 状态迁移

- `MemArrayModel` 继承 `ClockedModule`。
- `instruction_rows_`、R1 pending response 和 MEM cycle 改为 `ClockedState`。
- SRAM 继续使用稀疏页式专用 Memory；LOW 暂存局部 segment write，POSEDGE 原地提交。
- Queue front 在 LOW 接受，只有 POSEDGE 后才 pop；transport commit counter 与实际 SRAM
  write 在同一 POSEDGE 更新。
- TspSliceSystem 的两个 MEM 和 `SxmMatrixMemReference` 的 MEM 均加入各自 Clock 的统一
  validate/commit 集合。

### 保留的行为

- 指令进入 row0 后仍在本 LOW 执行，以保持原有静态 schedule；提交后的寄存流水位于
  下一 tile，不发生 C++ 循环导致的同周期多级传播。
- `ReadTiming::Main` 保持直接写 Fabric Next；`RegisteredR1` 保持额外一拍 R1。
- 同周期同地址读写保持 read-before-write。
- 未限定深度的 `instruction_queues_` 不冒充已确认的硬件 FIFO。

### 验证

- Release 干净构建：PASS。
- MEM 单元测试：8/8 PASS。
- 受影响 TSP/SXM 子系统：4/4 PASS。
- 新增 `unit.mem.clocked`：验证 SRAM 边沿写、指令逐级流水、R1 响应、注册顺序无关和
  validate barrier 失败前无部分提交。
- fast 全集首次运行暴露一个旧测试仍只提交 Fabric 的适配遗漏；修正调用点后，排除既有
  RMSNorm 基线失败为 60/60 PASS。
- `kernel.rmsnorm` 仍以同一文本失败：
  `VXM input Buffer cannot be reconfigured while a Bundle is active`。
- 原始 `FTLPU-CMODEL-main` 的 861 个基线文件重新校验：0 missing，0 changed。

### 未解决边界

- 缺少可靠 RTL/SRAM Macro 规格，尚不能确认真实 port 数、读写冲突、write mask、物理
  read latency 和 initiation interval。
- 本批证据是行为契约、不变式和独立确定性参考检查，不是 RTL 逐周期对照。

### 证据

- `timing_refactor/checkpoints/checkpoint_2_mem.md`
- `timing_refactor/manifests/checkpoint_2_pre_migration_hashes.csv`
- `timing_refactor/manifests/checkpoint_2_changed_files.csv`
- `timing_refactor/checkpoints/checkpoint_2_evidence/`

## 2026-11-14：Checkpoint 3 MXM

### 契约选择

- 当前顶层实际实例化 VMODEL legacy `lpu_mxm_slice`；其 Load/Dequant/Compute 控制使用
  四级寄存 row pipeline，row0 也在正沿更新。
- 未集成的 `lpu_mxm_core` 使用更深的 M1～A3/Backend 架构。本批没有把另一套架构的
  固定延迟强加给当前 CModel。

### 状态迁移

- `MxmControlSlice`、`MxmSupercell`、`MxmArray`、`Mxm` 和 `MxmAccumulator` 接入
  `ClockedModule` 语义。
- Control 三条 row pipeline、Supercell Weight/valid/Activation、Datapath WS/IS pipeline
  和周期状态改为 Current/Next；Queue front 只有全局 POSEDGE 后才 pop。
- Accumulator 保留大型数组，LOW 仅记录命中的 segment 操作，POSEDGE 原地提交。
- TspSliceSystem 将每个 MXM Control 和 Datapath 加入全局验证/提交集合；LOW 继续按
  ICU dispatch、Control、Datapath 的显式 SystemPhase 顺序执行。

### 周期行为变化

- 新发射控制条目先进入 row0 Next，正沿后才成为 Current；旧 C++ 的 row0 零拍旁路
  被消除。
- 静态 schedule 的 MXM Queue 发射相对目标执行周期提前一拍，MEM 数据路径按实际
  Stream Register 延迟对齐。
- Weight/valid、Activation stage、Datapath state 和 Accumulator 写入只在统一正沿后
  对外成为 Current；HIGH 不重复 issue、pop 或写入。

### 验证

- `unit.mxm.clocked`：PASS；覆盖 row0 与四级逐边沿传播、Supercell 边沿可见、
  Accumulator 同 LOW 投影、Clock 注册顺序无关和 validate barrier 无半提交。
- MXM 标签：9/9 PASS。
- fast 排除既有 RMSNorm：61/61 PASS。
- RMSNorm 保持同一冻结错误：`VXM input Buffer cannot be reconfigured while a Bundle is active`。
- Checkpoint 2 慢速基线与本批均受已有 target configuration/VXM 问题阻塞；其中
  Decoder Layer 的旧 MXM 权重到达偏移已修正，随后到达既有 128-row accumulator
  容量边界。本批未改硬件配置掩盖该问题。
- 原始 `FTLPU-CMODEL-main` 复核无变化。

### 证据

- `timing_refactor/checkpoints/checkpoint_3_mxm.md`
- `timing_refactor/manifests/checkpoint_3_pre_migration_hashes.csv`
- `timing_refactor/manifests/checkpoint_3_changed_files.csv`
- `timing_refactor/checkpoints/checkpoint_3_evidence/`

### 停止点

Checkpoint 3 完成后按用户要求暂停；没有开始 Checkpoint 4 VXM。

## 2026-11-14：Checkpoint 4 VXM

### 权威仓库核查

- 开始前调用“每日核查并同步 FTLPU 三仓库”Work。权威 CMODEL 默认分支仍为
  `main`，SHA 为 `2351cec89b5ef9c3c73f97b2d2e77c458fb36c95`，没有新的远端提交。
- 该 Work 后续把 `D:\LPU\FTLPU-CMODEL-main` 建立为上述 SHA 的 clean Git 工作树；
  Checkpoint 4 只修改 `D:\LPU\FTLPU-CMODEL-clock`，未向权威目录写源码。
- 未执行全仓重复哈希；只在阶段末记录本批 6 个源码/构建文件的 SHA-256。

### 状态迁移

- `VxmSlice` 继承 `ClockedModule`，注册到 `TspSliceSystem` 的全局 Clock。
- Slice 以层次化事务快照统一管理 8 条 instruction Queue、tile row pipeline、Superlane
  Decoder/Next/Current Config、repeat、Input Bundle 和 Lane 内部 ALU/LUT/quantizer/
  accumulator/feedback 状态。
- ICU issue 先进入 `pending_issues_`，`prepare_cycle()` 才合并到 Next；prepare 后禁止
  继续 issue，避免 LOW 修改已提交 Current。
- `evaluate()` 只准备 Next；全局 validate 全部通过后，`posedge()` 以指针交换一次提交
  完整 VXM 层次状态。HIGH 为只读，不重复 capture、decode、execute、release 或 pop。
- `VxmLane::Statistics` 明确为软件观测状态，快照之间共享，避免每拍复制增长中的 timeline。
- `VxmSuperlane` 增加深拷贝 datapath/control、共享只读 LUT/Statistics 的事务快照能力。

### 行为契约与 RTL 边界

- 保留静态调度错误，不新增动态 valid/ready retry 或 backpressure。
- 本地 VMODEL 证明了 local/global config wave、execute wave、resident config、pending
  valid、input token、accumulator 和 feedback/output 状态的正沿寄存性质。
- CModel 的 compact decoder/Superlane 组织与 integrated VMODEL 不完全同构；本批不宣称
  已完成逐周期 RTL 等价。

### 验证

- 独立 `out/checkpoint_4_vxm` Release 配置：73 个实际项目，72 个测试/工具可执行文件，
  0 missing。
- `unit.vxm.clocked`：PASS；覆盖 Current/Next 可见性、正反 Clock 注册顺序、验证屏障
  无部分提交、失败 LOW 不提交 Current、HIGH 不重复推进。
- VXM 标签：14/14 PASS。
- fast 排除冻结 RMSNorm：62/62 PASS，16.81 秒。
- `kernel.rmsnorm` 仍以同一文本失败：
  `VXM input Buffer cannot be reconfigured while a Bundle is active`。
- slow 6 项仍分别停在相同 MXM/MEM 容量边界或同一 VXM active-bundle 基线，未出现
  新失败类别。

### 证据与停止点

- `timing_refactor/checkpoints/checkpoint_4_vxm.md`
- `timing_refactor/manifests/checkpoint_4_changed_files.csv`
- `timing_refactor/checkpoints/checkpoint_4_evidence/`

Checkpoint 4 完成；ICU 按用户要求暂缓，下一批为 Checkpoint 5 SXM。

## 2026-11-14：Checkpoint 5 SXM

### 状态迁移

- `SxmSlice` 继承 `ClockedModule`；East/West 两实例注册到 `TspSliceSystem` 全局 Clock。
- ICU 当拍 issue 先进入 pending，SXM LOW 才建立完整层次 Next 并合并；prepare 后拒绝
  继续 issue。
- Local Transpose instruction wave、结果 Bank/metadata、Permute、fanout 延迟项、
  UnitGroup cycle 和 Wide Transpose job/FIFO/双 Bank payload 均在 Next 上推进一次。
- LOW 对 Stream Fabric 的 consume/write 保持一次性；全局 validation 全部成功后，
  `posedge()` 以状态指针交换提交，HIGH 不重复任何 SXM 副作用。
- 独立 `SxmMatrixCore/SxmMatrixMemReference` 明确为非 TSP 主 Slice 的三角矩阵参考路径；
  保留其自身 Clock/SystemPhase 和固定 P/WriteTap 日历。

### RTL 边界

- 本地 VMODEL 的 `lpu_sxm_control.sv` 提供 Permute 一拍控制寄存证据。
- `lpu_sxm_slice.sv` 提供 Transpose Bank、tile ready、pass-through 边界寄存器的
  `always_ff` 证据；Permute 数据选择为组合路径。
- CModel 的行式捕获、扇出延迟和 WideStreaming 扩展未被冒充为 integrated RTL 的
  逐对象同构实现，完整逐周期 Trace 仍待建立。

### 验证

- 独立 `out/checkpoint_5_sxm` Release 构建：74 个实际项目、73 个 `.exe`、0 missing。
- `unit.sxm.clocked`：PASS；覆盖 Current/Next 可见性、正反注册顺序、HIGH 无重复推进和
  validation barrier 无部分提交。
- SXM 标签：10/10 PASS。
- fast 排除冻结 RMSNorm：63/63 PASS，18.50 秒。
- `kernel.rmsnorm` 保持同一错误：
  `VXM input Buffer cannot be reconfigured while a Bundle is active`。
- slow 6 项继续分别停在 4 个 MXM accumulator 配置容量、1 个 MEM SRAM 配置容量和
  1 个既有 VXM active-bundle 边界，没有出现新失败类别。

### 停止点

Checkpoint 5 完成。ICU 按用户要求继续暂缓，未开始 Checkpoint 6。

## 2026-11-14：Checkpoint 7A C2C Endpoint/Link

### 迁移范围

- 按用户要求继续暂缓 ICU；将 Checkpoint 7 拆成可独立验证的 7A。
- `C2cTxSlice` 和 `C2cRxSlice` 建立层次 Current/Next，issue 先进入 pending；四 tile
  gather/replay、active context、完成 FIFO/计数和 MEM read request 在 POSEDGE 提交。
- `C2cEndpoint` 成为 `ClockedModule`；TSP 使用永久 Clock slot 注册 optional Endpoint，
  避免 attach/detach 后 Clock 保存失效对象地址。
- `C2cLink` 的 serializer、in-flight、RX-ready FIFO、outstanding credit 和 cycle 形成一个
 事务 State；旧 `tick()` 保留为 LOW/validate/commit 兼容包装。
- 不修改 `C2cDmaEngine`、`Ddr4Model` 或 ICU completion/notify 语义。

### 验证

- 独立 `out/checkpoint_7a_c2c` Release 构建成功：74 个测试，新增
  `subsystem.c2c.clocked`。
- C2C 标签 5/5 PASS。
- fast 排除冻结 RMSNorm：64/64 PASS，18.82 秒。
- `kernel.rmsnorm` 保持同一 active-bundle 错误。
- slow 6 项保持 Checkpoint 5 的四个 MXM accumulator 容量、一个 MEM SRAM 容量和一个
  VXM active-bundle 失败类别。

### 停止点

Checkpoint 7A 完成。ICU、C2C DMA/DDR、DMA completion 通知事务化和跨芯片统一 Clock
barrier 尚未开始。

## 2026-11-14：Checkpoint 6 ICU

### 状态迁移

- `InstructionControlUnit` 成为 `ClockedModule`，并注册到 `TspSliceSystem` 全局 Clock；
  SystemPhase 继续独占显式 ICU dispatch LOW，Clock 只负责验证和统一提交。
- LOW 从 ICU Current 建立完整层次 Next，只在 Next 中推进 local Queue、IQ/pending fetch、
  Macro reservoir/parser/DDB/active context、Repeat/NOP/Sync/Event、QCTL pool/frontend、
  Barrier、Queue completion 和 cycle。
- local semantic/raw iMEM 以及 Macro 原始映像按 Memory 分类，共享不可变 backing；配置写
  使用 Copy-on-Write，避免每拍复制完整程序存储。
- Macro DDB 与 active context 存储使用可深拷贝指针；QCTL shared-pool 指针及 MEM transport
  executor 在 snapshot/commit 后重绑定。
- 周期外通知进入 Pending-event；LOW 应用到 Next。Barrier due 与 Queue completion ready
  对 Pending-event 做无副作用投影，既不提前修改 Current，也维持同拍组合安全/握手检查。
- instruction-fetch DMA、C2C data DMA 和 DDR 均未被合并到 ICU local fetch；DMA/DDR 内部
  状态继续留在后续检查点。

### 验证

- 独立 `out/checkpoint_6_icu` Release 全量编译成功，共 75 项测试。
- 新增 `unit.icu.clocked`，覆盖 Current/Next 边沿可见性、外部通知采样、HIGH 不重复推进、
  以及全局 validation failure 前 ICU 不发生部分提交。
- ICU 标签 8/8 PASS；C2C 标签 5/5 PASS；未归类的 program/ICU configuration 与
  `vxm_fp32_input` 3/3 PASS。
- 最终 fast 排除冻结 RMSNorm：65/65 PASS，14.26 秒。
- `kernel.rmsnorm` 保持同一 active-bundle 错误：
  `VXM input Buffer cannot be reconfigured while a Bundle is active`。
- slow 6 项保持四个 MXM accumulator 配置容量、一个 MEM SRAM 配置容量和一个 VXM
  active-bundle 失败类别，没有出现新失败类别。

### 停止点

Checkpoint 6 完成。Checkpoint 7A 已在此前完成；未自动开始 C2C DMA/DDR 或跨芯片统一
Clock barrier。

## 2026-11-14：Checkpoint 7B C2C Data DMA/DDR

### 状态迁移

- `C2cDmaEngine` 建立完整 Current/Next，覆盖 descriptor Queue、active transfer、TX/RX
  FIFO、DDR outstanding、completion token、cycle 和 trace；Endpoint send/pop/issue 先暂存。
- `Ddr4Model` 建立 request/active/completion、latency、bandwidth、round-robin、cycle/trace
  Current/Next；稀疏 `pages_` 只暂存本 LOW 命中的 byte write，在 POSEDGE 原地提交。
- TSP 为两个可选 data DMA 和共享 DDR 注册永久 Clock slot；SystemPhase 明确执行
  Endpoint→DMA→completion notify→DDR→ICU barrier，Clock 只负责统一验证和提交。
- `C2cDmaSystem` 删除 TSP 周期后的第二次 `ddr4.tick()`；DDR 每物理周期只推进一次。
- ProgramLoader 内部 instruction-load DMA 保持现有软件状态机，与 C2C data DMA 明确分离；
  两者只共享同一个 DDR request/带宽模型。

### 验证

- 独立 `out/checkpoint_7b_dma_ddr` Release 全量构建成功，共 76 项测试。
- 新增 `subsystem.c2c.dma_clocked`：PASS；覆盖 LOW 不提前可见、DMA/DDR 同边沿提交、
  HIGH 无重复推进、反向 Clock 注册顺序和 validation failure 无部分 Memory 写入。
- fast 排除冻结 RMSNorm：66/66 PASS，27.72 秒；未归类配置/VXM：3/3 PASS。
- `kernel.rmsnorm` 保持同一 active-bundle 错误；slow 6 项保持四个 MXM accumulator
  配置容量、一个 MEM SRAM 配置容量和一个 VXM active-bundle 失败类别。

### 停止点

Checkpoint 7B 完成。未开始 Checkpoint 7C 跨芯片统一 Clock，也未进行最终全系统周期级
Trace 审计。

## 2026-11-15：ICU 指令链路与本地 VMODEL 对齐

### 权威 RTL 基线

- 只读使用 `D:\LPU\FTLPU-VMODEL-main` 当前磁盘工作树：
  branch `rtl-handshake-naming`，HEAD
  `2cbaa104090c6df099894559794c0f981d4aafda`。
- 冻结和复核状态均为 116 modified、10 deleted、55 untracked；ModelSim
  编译没有在 VMODEL 中产生新文件。
- 实际 ProgramLoad filelist、顶层模式选择、MEM/MXM local frontend、
  VXM/SXM boundary 和 legacy ICU 均重新从源码追踪；没有用 GitHub main
  或旧调查假设替代本地 RTL。

### CModel 实施

- 新增 `RtlSynchronousFifo<T>`，明确 Current/Next occupancy、pointer、
  write/read fire、empty no-bypass、full replacement 和 relay
  registered-ready 两种行为。
- ProgramLoader 接入统一 Clock/SystemPhase；program operations 先暂存到
  ICU Next，ProgramLoader、instruction DMA、data DMA、DDR 和 ICU 在同一个
  validate/commit barrier 提交。
- ProgramLoad 增加 8-entry word FIFO、2-entry ingress FIFO、每半区
  7×2 relay buffers；MEM 按物理 region 停在近端，MXM 到 R6/tail。
- local iMEM fetch 改为 2-cycle，并加入 response FIFO4、word FIFO4。
- MEM/MXM 加入 decoded context FIFO4、active contexts4、timed issue
  FIFO16 和 C+2 lookahead；PRIME 覆盖完整 pipeline。
- 所有新 FIFO 增加 occupancy peak/full/read/write/stall/backpressure
  统计；ProgramLoader test 增加分级写读守恒。
- 保留 CModel VXM/SXM、Native NOP/Repeat、QCTL 和 FU datapath；未启动
  Checkpoint 7C。

修改文件：

- 构建/配置：`CMakeLists.txt`、`config/ftlpu-lpu32.json`、
  `cmake/FTLPUIcuGeometry.cmake`、`cmake/FTLPUHardwareConfig.cmake`、
  `cmake/hardware_config.hpp.in`、`include/ftlpu/core/hardware_params.hpp`。
- 实现：`include/ftlpu/icu/rtl_fifo.hpp`、
  `include/ftlpu/icu/macro_bitstream_decoder.hpp`、
  `include/ftlpu/icu/distributed_queue.hpp`、
  `include/ftlpu/c2c/dma.hpp`、`include/ftlpu/system/icu.hpp`、
  `include/ftlpu/system/program_loader.hpp`、
  `include/ftlpu/system/tsp_slice_system.hpp`、
  `include/ftlpu/system/c2c_dma_system.hpp`。
- 测试：`tests/unit/icu/rtl_fifo_test.cpp`、
  `tests/unit/icu/rtl_fifo_golden_trace.cpp`、
  `tests/unit/icu/macro_schedule_test.cpp`、
  `tests/unit/icu/raw_macro_decoder_test.cpp`、
  `tests/subsystem/system/program_loader_test.cpp`。
- RTL Golden：`timing_refactor/rtl_golden/queue_fifo_trace_tb.sv`、
  `timing_refactor/rtl_golden/compare_queue_fifo_traces.ps1`。
- 文档：`ICU_LOCAL_VMODEL_BASELINE.md`、`ICU_FIFO_INVENTORY.md`、
  `ICU_INSTRUCTION_PATH_ALIGNMENT.md` 和本 changelog。

### CModel 验证

- Release 全量构建：PASS。
- ICU/FIFO/Macro/ProgramLoader/统一时钟六项定向测试：6/6 PASS。
- raw Macro：`prime_cycles=20`；configured ProgramLoad：169 cycles。
- 全量 CTest：70/77 PASS。7 个失败与 Checkpoint 4～7B 的冻结 baseline
  同类：4 个 MXM accumulator 配置容量、1 个 MEM SRAM 配置容量、2 个
  VXM active-bundle；没有新增 ICU 失败。

### RTL 与 Golden

- ModelSim 10.1c 实际执行并通过：通用 FIFO、legacy ICU
  NOP/Repeat/MEM stride、Program config FIFO、reassembler、elastic
  stage、七级 relay chain、MEM Macro frontend、MXM 三类 Macro frontend。
- FIFO 使用同一 12-cycle stimulus，对正沿前事件与正沿后
  level/pointer/head 共 17 fields 逐字段比较：PASS。
- 完整 `filelist_frontend_cluster_top.f` 编译被 ModelSim 10.1c 对较新
  SystemVerilog 语法的支持限制阻塞。完整 ProgramLoad→FU cluster
  cycle Golden 仍为 PARTIAL，未冒充 PASS。

### 尚未对齐

- Program command RAM/PC、独立 config/outstanding-ID/RX FIFO 协议。
- BOTH 分支、VXM direct、endpoint RR/commit-valid、PRIME FIFO。
- distributed 模式的 VXM/SXM Program endpoint 当前 RTL `ready=0`。
- legacy 138-Queue NOP/Repeat 与 CModel Native queue 的完整同激励 cycle
  Trace 尚未建立；两者功能测试各自通过。

### 关键 SHA-256

| 文件 | SHA-256 |
|---|---|
| `config/ftlpu-lpu32.json` | `5D886BBD1676018A873012F8D01E009911162BA21725BD103AC5B2ACA838D588` |
| `include/ftlpu/icu/rtl_fifo.hpp` | `9D17D6B8D1A0CDB28DE4E9E20D7638B1A4ABD410A308D1688D82799F5E1146B1` |
| `include/ftlpu/system/program_loader.hpp` | `4782202489781ED1C3BE5F8678DA43CC0FA9AF34FDFFBA25BBFAF49C4F0D0AE3` |
| `include/ftlpu/icu/macro_bitstream_decoder.hpp` | `69502F83AEF0B52FE60F499C9B1734934D1238897E7800E1D72D77F274EEDC07` |
| `include/ftlpu/icu/distributed_queue.hpp` | `0C329DBD207A3E12843A37A217D4317893ABF563D0324039A2B807CA27A42040` |
| `tests/subsystem/system/program_loader_test.cpp` | `34235021F6E15DC2DC95B40DDC9562CB6756E3D1BD24FA75B53FBA4839B1F043` |
| `timing_refactor/rtl_golden/queue_fifo_trace_tb.sv` | `187C2B506543CB9D2D8800DE3E7536405953D808D035C88B3611584FB1612F7B` |
| `timing_refactor/ICU_INSTRUCTION_PATH_ALIGNMENT.md` | `24A3066E7BD093149AE3772DA8C52670713A13C79BA458AD954FB09831CAB387` |

## 2026-11-15：ICU A.1 Relay Mapping 与 A.2 Instruction DMA

### A.1 MEM Relay Mapping 修正

- 以本地 VMODEL 当前源码为准，确认每个 Relay 管辖
  `4 slices/group × 2 banks/slice × 2 groups/relay = 16 MEM banks`。
- 每半区 104 条 MEM Queue 映射为 7 个 Relay：R0～R5 各覆盖 16 条，R6
  覆盖 96～103；MXM Load/Dequant/Compute 均走该半区尾端 R6。
- 新增 `mem_groups_per_relay=2` 生成配置和编译期几何约束，删除
  ProgramLoader 对旧分组关系的隐式推断。
- 新增 East/West 共 208 条 MEM Queue 穷举映射测试及 MXM tail 测试。
  代表 Trace：Queue 4 从错误 R1 修正为 R0，Queue 16 从错误 R4 修正为 R1，
  Queue 96 保持 R6。

### A.2 Instruction DMA 集成

- ProgramLoader 不再把 `C2cDmaEngine` 当作指令装载 DMA；改为实际持有并驱动
  `InstructionFetchDma`，C2C Data DMA 与 Instruction DMA 成为独立对象，只共享
  同一个 `Ddr4Model` 及其带宽/仲裁。
- Instruction DMA 使用 256-bit DDR beat、8-bit ID、16-entry ordered
  outstanding ID 和 16-entry RX FIFO；实现 ID/last/error 检查、full replacement、
  下游反压、reset orphan drain、稳定输出和 32/64/96/128-bit record 重组。
- Instruction DMA、ProgramLoader 与 DDR 纳入原有 Clock/SystemPhase Current/Next
  边界；HIGH 不重复请求、入队、出队或写 iMEM。
- 新增单命令、多 outstanding、不同 record width、精确 byte order、RX full/
  recovery、ID mismatch、response error、reset/drain 测试。原 ProgramLoader 共享
  DDR 测试仍同时运行独立 C2C Data DMA，证明两条 DMA 语义未合并。

### 验证与边界

- Release 全量构建：PASS。
- 定向 9 项 CTest：9/9 PASS，覆盖 timing、ICU、Instruction DMA、C2C DMA/DDR、
  ProgramLoader、Relay mapping 和硬件配置。
- 全量 CTest：72/79 PASS；7 个失败与既有冻结基线同类（4 个 MXM accumulator
  配置容量、1 个 MEM SRAM 配置容量、2 个 VXM active-bundle），没有新增失败。
- 本地 VMODEL `lpu_program_load_relay_chain_tb`：80 checks PASS；
  `lpu_instruction_dma.sv` 与 `lpu_queue_fifo.sv`：ModelSim 编译 PASS。
- 本地 VMODEL 没有独立 Instruction DMA 功能 TB，因此 A.2 的 RTL/CModel
  同激励功能 Golden 仍为 PARTIAL，不能用“编译通过”替代功能等价证明。
- 未修改 VMODEL，未启动 FIFO depth sweep、完整 attention workload 或
  Checkpoint 7C。

