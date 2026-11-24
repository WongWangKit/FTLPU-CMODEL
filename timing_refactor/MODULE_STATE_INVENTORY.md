# 模块时序状态清单

## 分类

- A：真实硬件时序状态，必须具有明确有效沿更新机制。
- B：纯组合结果，由 SystemPhase 按拓扑计算。
- C：软件控制、解析或统计状态，不因统一接口而自动包装为 ClockedState。
- D：Pending 请求/响应等延迟事件，需要映射全局物理周期，但不一定使用 ClockedState。

## Checkpoint 1 状态

| 模块/类 | 主要状态 | 分类 | 当前机制 | 迁移状态 |
|---|---|---:|---|---|
| Clock | phase_count、phase、faulted、注册集合 | A/C | 专用全局阶段调度 | 已完成并复验 |
| SystemPhase | LOW/HIGH 有序节点和阶段守卫 | C | 显式拓扑回调 | 已完成并复验 |
| ClockedState<T> | Current、Next、prepared | A | 通用 Current/Next | 已完成并复验 |
| Register/RegisterBank | q、d、enable、同步复位 | A | ClockedState | 已完成并复验 |
| StreamRegisterFabric | 稀疏 Current/Next、producer、consume、活动计数 | A/C | 专用稀疏双缓冲 | 已完成并复验 |
| TspSliceSystem | 全局 Clock、SystemPhase、active sinks | A/C | 顶层四阶段接线 | 顶层接线完成；内部 FU 未迁移 |
| SxmMatrixMemReference | 外部 Fabric 的阶段驱动 | C | 独立 Clock/SystemPhase 驱动 | 调用接口适配完成；非 SXM 全迁移 |

## Checkpoint 2：MEM 状态映射

| 类/成员 | 硬件含义 | 分类 | 迁移后机制 | 状态 |
|---|---|---:|---|---|
| `MemArrayModel::instruction_rows_state_` | 每个 MEM slice/bank 的 Read/Write 指令波前流水寄存器 | A | `ClockedState<InstructionRows>`；LOW 从旧 Current 派生 Next，POSEDGE 整体提交 | 已迁移并验证 |
| `instruction_queues_` | 未限定深度的 host/ICU 投递容器；现有源码未证明为片内物理 FIFO | C/待定 | 保留 `std::deque`；LOW 只标记被接受 front，POSEDGE 才 pop | 接口边界已迁移；物理身份待 ICU 阶段确认 |
| `Sram::pages_` | MEM SRAM 数据阵列的稀疏行为存储 | A | 专用 Memory；LOW 采样请求并暂存局部写，POSEDGE 原地更新相应 8-byte segment | 已迁移并验证行为契约 |
| `pending_sram_writes_` | 当前 LOW 已接受、等待正沿提交的写集合 | A 的 Next 表示 | 专用局部写暂存，不复制整个 SRAM | 已迁移并验证 |
| `read_response_state_` | `RegisteredR1` 的同步读输出寄存器 | A | `ClockedState<vector<MemTransfer>>` | 已迁移并验证 |
| `staged_read_responses_` | 本 LOW 从旧 R1 Current 送往 Fabric 的观察快照 | B/C | LOW 临时视图；不跨正沿独立保存 | 已分类 |
| `cycle_state_` | MEM 已提交物理周期计数 | A/诊断 | `ClockedState<size_t>`，只在 POSEDGE 增加 | 已迁移并与 Fabric 锁步验证 |
| `transport_commits_.*.committed` | 实际 SRAM transport segment 提交计数 | A/C | 仅在对应 staged SRAM write 的 POSEDGE 更新 | 已迁移并验证 |
| `executed_mem_` / `executed_instructions_` / `consumed_mem_` | LOW 接受事件与调试 Trace | C | 每个 LOW 重建；不作为硬件 Current | 已分类 |
| `active_fabric_` / `cycle_prepared_` | 阶段调用守卫 | C | 拒绝重复 LOW、无 LOW 提交和边沿中 host mutation | 已迁移并验证 |

MEM 当前无 stall/hold 流水语义：每个有效正沿所有指令级向下一 tile 推进一格。每个
slice/bank 的读、写入口各可在一个 LOW 接受一条指令，这是现有 CModel 行为契约，
不是已经由 SRAM Macro 或 RTL 证实的真实端口带宽。

## Checkpoint 3：MXM 状态映射

| 类/成员 | 硬件含义 | 分类 | 迁移后机制 | 状态 |
|---|---|---:|---|---|
| `MxmControlSlice::*_instruction_rows_state_` | legacy Load/Dequant/Compute 四级控制流水寄存器 | A | 三个 `ClockedState<Rows>`；只执行 Current、统一提交 Next | 已迁移并验证 |
| `MxmControlSlice::*_instruction_queue_` | ICU 到 MXM 的未限定深度 host 投递容器 | C/待定 | LOW 接受 front，POSEDGE 才 pop；深度与物理 FIFO 身份留待 ICU 阶段 | 接口边界已迁移 |
| `MxmControlSlice::cycle_state_` | 已提交的 MXM control 周期 | A/诊断 | `ClockedState<size_t>` | 已迁移并验证 |
| `compute_pulses_` 及 pulse details | Control Current 译码产生的本 LOW 组合控制 | B/C | 每个 LOW 重建，不独立跨正沿保存 | 已分类 |
| `MxmSupercell::weight_*_state_` | 两组 Weight Buffer、column valid、buffer valid | A | `ClockedState`；IW 只写 Next，正沿后可见 | 已迁移并验证 |
| `MxmSupercell::activation_stages_state_` | Supercell activation pipeline | A | `ClockedState<ActivationStages>`；每正沿只推进一级 | 已迁移并验证 |
| `MxmArray::evaluated_` | 防止同一 LOW 重复求值的阶段守卫 | C | LOW 临时标志，POSEDGE 后清除 | 已分类 |
| `Mxm::DatapathState` | WS/IS activation、wave、output、MAC pipeline、valid/owner/cursor 状态 | A | 一个堆上 `ClockedState<DatapathState>`，避免增大栈上 `TspSliceSystem` | 已迁移并验证现有功能 |
| `MxmAccumulator::rows_` | FP32 accumulator SRAM/大数组 | A | 专用 Memory；LOW 暂存局部 segment accumulate/clear，POSEDGE 原地提交 | 已迁移并验证 |
| `MxmAccumulator::pending_operations_` | 本 LOW 接受的 accumulator 局部写序列 | A 的 Next 表示 | 稀疏操作日志；同 LOW read 使用顺序投影 | 已迁移并验证 |
| `last_outputs`、日志、输入 slot | 本 LOW 观察结果或组合输入 | B/C | 不另建硬件寄存器；按现有接口生命周期管理 | 已分类 |

当前集成 RTL 证据支持 legacy Control 的四级寄存波前和 compute/accumulator 正沿状态；
未集成 `lpu_mxm_core` 的深流水/Backend 是不同架构，未作为本 Checkpoint 的周期契约。
未发现当前 legacy pipeline 的 stall/hold 接口，因此本批保持每正沿推进一格，不伪造
动态反压。

## Checkpoint 4：VXM 状态映射

| 类/成员 | 硬件含义 | 分类 | 迁移后机制 | 状态 |
|---|---|---:|---|---|
| `VxmSlice::State::instruction_rows` | 8 条 local control wave 的 tile 行寄存器 | A | Slice 层次事务 Current/Next；每正沿推进一 tile | 已迁移并验证 |
| `pending_issues_` / `State::instruction_queues` | ICU 当拍投递与未限定深度 host Queue | C/待定 | LOW 前投递容器合并到 Next；物理深度留待 ICU | 接口边界已迁移 |
| `VxmSuperlaneInstructionControl::{queues_,decoding_,next_,current_}` | 3-entry local FIFO、1-cycle decoder、Next/Current Config、repeat counter | A | 作为 Superlane 的 Slice Next 快照求值并在统一 POSEDGE 整体提交 | 已迁移并验证 |
| `VxmInputBuffer` lane bytes/occupied/count/state | 每 tile Bundle Buffer 的 Empty/Collecting/Ready 和物理槽数据 | A | capture/release 只修改 Slice Next；失败 LOW 不提交 Current | 已迁移并验证 |
| `VxmLane::stage_inputs_` / ALU/LUT pipelines | Lane token 和 Basic/LUT 多周期流水寄存器 | A | Superlane 深拷贝硬件状态进入 Slice Next，正沿整体提交 | 已迁移并验证 |
| accumulator/output/feedback holding registers | Lane 局部累加、tail output、固定反馈保持 | A | 包含在层次化 Next 中；LOW 冲突检查、POSEDGE 提交 | 已迁移并验证 |
| static quantizer pending/request | 固定一拍量化流水 | A | 包含在 Lane Next 中，每物理周期推进一次 | 已迁移并验证 |
| `output_slots` / `required_streams` | 本 LOW 的路由结果与组合需求视图 | B/A 边界 | 从事务 Next 提供给 Fabric staging；不在 HIGH 重算副作用 | 已分类并验证 |
| `VxmLane::Statistics` / trace timeline | 软件统计和调试历史 | C | `shared_ptr` 共享，不复制为硬件状态；不参与功能提交判定 | 已分类 |
| `low_started_` / `low_complete_` | 阶段调用守卫 | C | 拒绝重复 LOW、无完整 LOW 提交和 prepare 后再 issue | 已迁移并验证 |

RTL 证据只用于确认寄存状态类别和波前边界。当前 VMODEL integrated VXM 的 resident
config/execute wave 与 CModel Superlane decoder 组织并非逐对象同构，因此本 Checkpoint
是明确行为契约与不变式验证，不是完整 RTL Golden 对照。

## Checkpoint 5：SXM 状态映射

| 类/成员 | 硬件含义 | 分类 | 迁移后机制 | 状态 |
|---|---|---:|---|---|
| `pending_issues_` | ICU 在本 LOW 前投递的 SXM 命令 | C/入口边界 | prepare 时合并到 Slice Next；LOW 开始后禁止追加 | 已迁移并验证 |
| `State::transpose_instruction_rows` | Transpose 南到北 tile 控制波 | A | 层次事务 Current/Next；每正沿推进一 tile | 已迁移并验证 |
| `State::permute_instruction` | Slice-global 单拍 Permute 控制寄存 | A | 当拍入口进入 Next，LOW 读取事务副本，正沿后清空 | 已迁移并验证 |
| `State::transpose_bank.block` | 两个 byte plane 的 Transpose 结果寄存/存储 | A | capture 只写 Bank Next；失败 LOW 不污染 Current | 已迁移并验证 |
| `dst_streams/tile_ready/input_row_mask/ready_cycle` | Bank 元数据、占用和最早可消费周期 | A | 与 payload 同一 Slice Next 原子提交 | 已迁移并验证 |
| `State::fanout_outputs` | 多目的 block 输出的 superlane 延迟项 | A | LOW 发送到期 Current 投影并生成 Next；正沿提交 | 已迁移并验证 |
| `SxmUnitGroup::issued_/issued_stream_ops_/cycle_` | 简单流操作的本拍 issue 和周期状态 | A | 作为 Slice State 快照求值并整体提交 | 已迁移并验证 |
| `SxmWideTranspose` jobs/input/output/memory/last_issue/bank | Wide 模式 job、对齐 FIFO、双 Bank payload 与发射间隔 | A | 包含在 Slice 层次 Current/Next；不在 HIGH 重复推进 | 已迁移并验证 |
| `cycle_events_` / `trace_enabled_` | 软件日志和观测开关 | C | 保持在事务状态外，不宣称为 DFF | 已分类 |
| `active_fabric_` / LOW flags | 外部 Fabric 绑定与阶段守卫 | C | 拒绝重复 LOW、无完整 LOW 提交和提交后未清理 | 已迁移并验证 |
| `SxmMatrixCore` / `SxmMatrixMemReference` | 独立三角矩阵和固定 WriteTap 日历参考路径 | 独立试验边界 | 保留自身 Clock/SystemPhase 驱动；非 TSP 主 Slice | 已分类，专项回归通过 |

VMODEL 证据支持 Transpose 控制波、Permute 控制寄存、Transpose Bank/tile ready 和
pass-through 边界寄存。CModel 的行式捕获、扇出延迟和 WideStreaming 扩展没有被描述成
与 integrated RTL 完整逐周期同构。

## Checkpoint 6：ICU 状态映射

| 类/成员 | 硬件含义 | 分类 | 迁移后机制 | 状态 |
|---|---|---:|---|---|
| semantic/raw local iMEM | 每条逻辑 Queue 的基础指令或 Macro 物理程序映像 | Memory | `CopyOnWriteVector` 共享只读 backing；配置写时分离 | 已分类并迁移 |
| IQ / pending fetch response | fetch 返回、16-entry IQ 和读延迟在途状态 | A | 随 ICU 层次事务快照只在 LOW 推进，POSEDGE 整体提交 | 已迁移并验证 |
| Macro reservoir/parser FSM | Packed bit reservoir、packet parser、pending raw fetch | A | 完整 ICU Next 快照 | 已迁移并验证 |
| Macro DDB / active context RAM | Delta Dictionary 与正在展开的 Macro context | A/Memory | `ClonePtr` 深拷贝硬件状态；提交后成为 Current | 已迁移并验证 |
| Repeat/NOP/Sync/Wait/Notify state | Queue-local Sequencer 控制状态 | A | Queue 快照内推进一次 | 已迁移并验证 |
| QCTL pool/cache/frontend | 共享控制 RAM、cache、request/response 和 Queue frontend | A/Memory | pool 与 Queue 同一 ICU Next；复制/提交后重绑定共享指针 | 已迁移并验证 |
| Common Barrier / Queue completion | task generation、completion instance、hold/run 和计数 | A | 外部事件先 pending，LOW 应用到 Next；组合安全查询只预览 | 已迁移并验证 |
| `cycle_` / trace counters | ICU 物理周期和 Queue 观测统计 | A/C | 周期随 Next 提交；观测保持既有语义 | 已迁移并验证 |
| MEM transport executor | Queue-local QCTL 到 transport Queue 的连接 | C/接线 | snapshot/commit 后按新 ICU `this` 重新绑定 lambda | 已迁移并验证 |

ICU 以一个层次 `ClockedModule` 加入 TSP Clock；SystemPhase 仍独占 dispatch LOW，Clock
只负责验证与统一提交。周期外通知不写 Current，但 Barrier due 和 Queue completion 的
组合查询可在 LOW 前看到 Pending-event 的投影，以维持安全检查和 ready 语义。

instruction-fetch DMA 属于外部程序装载路径，不是 local iMEM fetch 状态；本检查点没有
把它纳入 ICU 快照。`C2cDmaEngine`/`Ddr4Model` 也继续留在后续 DMA/DDR 检查点。

## Checkpoint 7A：C2C Endpoint/Link 状态映射

| 类/成员 | 硬件含义 | 分类 | 迁移后机制 | 状态 |
|---|---|---:|---|---|
| `C2cTxSlice::State::queue/stream_queues` | ICU 到发送端的待发送向量命令 | C/入口边界 | LOW 前 issue 暂存；LOW 合并到 Next，POSEDGE 可见 | 已迁移；物理深度待 ICU |
| TX `pipeline/stream_pipelines` | 四 tile 对角 gather 寄存级 | A | 只从 Current/Fabric Current 派生 Next；正沿整体提交 | 已迁移并验证 |
| TX `completed/stream_completed/read_requests` | 完整向量发送保持和 MEM 读取通知 | A/C 边界 | Next 中产生/消费；Endpoint 与 TSP 同一验证屏障 | 已迁移；通知由 ICU Pending-event 采样 |
| RX `queue/stream_queues/stream_active` | Receive 命令和多向量上下文 | A/C 边界 | Slice 层次 Current/Next | 已迁移并验证 |
| RX `shared_ingress/shared_pipelines/pipeline` | 输入向量 ingress 和四 tile replay 寄存级 | A | LOW 只推进 Next，POSEDGE 整体提交 | 已迁移并验证 |
| `C2cEndpointClockSlot` | 可选 Endpoint 的永久 Clock 注册位置 | C/接线 | optional 创建/销毁不改变 Clock 指针生命周期 | 已迁移并验证 |
| Link serializer/in-flight/RX FIFO/credit/cycle | 芯片间序列化、飞行延迟、接收存储和 credit | A | 单一 Link State Current/Next；`tick()` 为兼容包装 | 已迁移并验证 |
| `C2cDmaEngine` / `Ddr4Model` | 外部描述符、DDR request/beat、completion | A/C | Checkpoint 7B Current/Next + 稀疏 Memory 局部写 | 已迁移并验证 |

TX/RX 的 Stream consume/write 仍只发生在 SystemPhase LOW。Endpoint 的 `low()` 不由 Clock
重复调用；Clock slot 只负责全局 validate/commit。RX/TX 对 ICU 的通知回调仍在 LOW，
Checkpoint 6 后由 ICU Next 接收并在同一 TSP POSEDGE 可见。

## Checkpoint 7B：C2C Data DMA/DDR 状态映射

| 类/成员 | 硬件含义 | 分类 | 迁移后机制 | 状态 |
|---|---|---:|---|---|
| DMA descriptor Queue / active transfer | 每 stream 的 DMA 命令和当前向量序列上下文 | A/C 入口边界 | LOW 前 issue 暂存；DMA Next 中启动/推进 | 已迁移并验证 |
| DMA TX/RX FIFO | Endpoint 与外部 DDR transport 之间的向量缓冲 | A | send/pop 先暂存；LOW 应用到 Next；POSEDGE 提交 | 已迁移并验证 |
| DMA DDR pending request | 每 stream 的 request id、vector index/tag | A | DMA Next 中维护；DDR completion pop 在 DDR LOW 同步提交 | 已迁移并验证 |
| DMA completion/token/cycle/trace | 完成历史、ICU 通知 token、周期与观测 | A/C | completion 在 DMA Next 产生；同 LOW 消费 token 并通知 ICU Next | 已迁移并验证 |
| DDR request/active/completion | 请求 Queue、latency、active beat 和 response FIFO | A | 完整 DDR State Current/Next | 已迁移并验证 |
| DDR bandwidth/round-robin/cycle/trace | 每拍 byte budget、lane 仲裁和观测计数 | A/C | DDR Next 中推进一次，POSEDGE 统一提交 | 已迁移并验证 |
| DDR `pages_` | 外部内存的稀疏行为存储 | Memory | 不复制；LOW 记录本拍 byte write，POSEDGE 原地写入 | 已迁移并验证 |
| TSP DMA/DDR Clock slots | optional 外部对象的稳定 Clock 注册位置 | C/接线 | 永久 slot 转发 validate/commit；LOW 仍由 SystemPhase 独占 | 已迁移并验证 |
| ProgramLoader 内部 DMA | instruction-load 软件执行路径 | C/A 混合、排除项 | 保持 legacy 立即状态机；共享 DDR 请求在 TSP DDR LOW 采样 | 明确排除，不与 data DMA 混同 |

DMA/DDR 的组合依赖固定为 Endpoint→data DMA→DDR。data DMA completion 在 DMA LOW 的
Next 中产生后，由同一 TSP LOW 消费 token 并写入 ICU Next，因此 DMA、DDR、ICU 和各 FU
仍由一个 TSP POSEDGE 原子提交。DDR request/pop 可以在 DDR LOW 前由 ProgramLoader 或
data DMA 暂存；DDR LOW 后拒绝新增，以防跨越本拍采样窗口。

## 后续模块

| 模块 | 待审计状态 | 当前状态 |
|---|---|---|
| MEM | SRAM Memory、读输出、Pending Response、指令流水、Bank/Port Queue | Checkpoint 2 已完成；真实 SRAM Macro 端口/冲突规格待外部证据 |
| MXM | Control/Datapath 寄存边界、Activation Pipeline、Accumulator、Weight 状态 | Checkpoint 3 已完成；未集成新 core 的深流水延迟不在本批契约内 |
| VXM | Config、Decoder/Instruction Pipeline、Input/Bundle Buffer | Checkpoint 4 已完成；RTL 逐周期 Trace 待后续证据 |
| SXM | Transpose Buffer、执行状态、Stream 交互 | Checkpoint 5 主系统 Slice 已完成；Matrix 参考路径保持独立驱动 |
| ICU | IQ、DDB、Reservoir、Context、QCTL、Barrier、Pending Fetch/Response | Checkpoint 6 已完成；逐周期 RTL Trace 待建立 |
| C2C/DMA | Endpoint、Credit、Transport Pipeline、在途事务 | 7A Endpoint/Link、7B data DMA/DDR 已完成；跨芯片 barrier 未迁移 |

后续 Checkpoint 必须先扩充本清单，确认每个成员的硬件含义后才能修改代码。ICU 已完成
Pending-event 与统一提交边界；跨芯片统一 Clock 仍需后续审计。

