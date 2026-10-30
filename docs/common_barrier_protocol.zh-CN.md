# 公共分页屏障 CModel 协议

新程序使用[按队列执行的伪控制协议](queue_control_v39.zh-CN.md)。下文继续描述 v37/v38 全局 BARR，旧文件不重新解释为 QCTL。

更新日期：2026-10-27。当前目标 `icu.common_transport_overlap_version=1` 支持
DDR→C2C→MEM→SRAM 与计算重叠；v37/未启用重叠程序仍保留串行语义。
软件容器 v38 的 E（transport_launch_cycle）允许提前启动，C 是实际检查点，
满足 E≤C≤首次消费。BARR encoding 1、packed-v1 编码不变。

每个 `InstructionControlUnit` 对应一个芯片、一个执行任务。公共控制表容量由
`icu.common_barrier_capacity` 指定（当前目标 1024 项），编码版本为 1。
`configure_common_barriers(generation, entries, bindings)` 必须在首次 dispatch
之前完成；空表有效。每项为两个 uint32 字段 `stop_cycle,event_id`，严格递增，
同周期多页依赖必须由链接器绑定成一个事件。没有绑定、孤立绑定、重复事件、
重复完成实例、容量不足或过期周期均明确失败。
完成实例状态另有协议固定上限：所有事件的成员总数最多 65535；配置时先检查
该上限再建立状态，超限明确拒绝，不使用无界事件状态模拟。

`generation` 非零，属于运行时任务；任务重置后应由运行时选择更新的 generation。
每个 binding 记录具体 page-use 实例编号，重复装入相同 SRAM 地址仍须使用新实例。
`notify_common_mem_commit(generation, instance)` 只能在该实例所有 MEM 写入已对
SRAM 可见后调用。重复完成幂等，旧 generation 忽略，未来 generation 或当前
任务中未链接的实例拒绝。DDR 读完成和同步控制包发射完成都不能替代 MEM commit。

状态为 INIT/PRIME、RUN、HOLD。INIT 装载控制表和队列，真实有限 Macro 前端通过
`prime_raw_macro_frontends()` 预热；预热保留原有取指延迟、reservoir、DDB 和
Context RAM 容量。它不推进逻辑计算时间。每次 dispatch 是一个物理周期 P，
公共逻辑时间 L 只在 `compute_enabled()` 为 true 的周期递增。

```text
P                  p        p+1       ...      q        q+1
L                  C         C        ...      C        C+1
事件               未完成    搬运继续           已锁存
状态               HOLD      HOLD              RUN      RUN
计算发射           无        无                发射C    发射C+1
MEM transport      正常      正常              正常     正常
```

控制器先于所有计算队列判断 C。早到事件只被锁存，在 C 才消费；晚到事件在下一次
dispatch 边沿被观察并恢复，从 C 发射，没有额外逻辑周期补偿。当前项和下一项由
控制 RAM 缓存。`logical_cycle()`、`physical_cycle()` 和
`common_barrier_wait_cycles()` 分别用于计算预算、搬运超时和物理等待统计。

HOLD 不调用计算发射、Macro parser、delta/DDB expansion 或 Context admission。
Native PC、NOP/循环计数及 Macro 坐标和地址保持。已发出的 i-MEM 请求在发起时
预留 IQ/reservoir credit；HOLD 只推进其物理响应计时并捕获响应，不产生新请求，
不消费已缓存数据。解码、admission 和 deadline 继续使用原逻辑周期。

`TspSliceSystem` 在第一次到达 C 前以及 HOLD 后事件就绪的恢复边沿检查
计算 VXM/MXM/MEM/SXM 在途状态及 SR 是否排空（含 VXM 内部普通 SR），早到事件也不能
跳过检查。非空 Macro Context 不妨碍暂停；SRAM、MXM
accumulator/weight/activation 和未待输出的 SXM transpose 存储可保持。存在
在途数据时直接失败，必须由最终链接器在 C 之前安排有成本的排空窗口；不会先
冻结再等 C 之后的指令排空。启用重叠后，使用明确来源区分计算和搬运，C 当拍
合法的 transport 可以仍在 MEM/SR 内，继续进入 HOLD；旧串行程序仍检查全片排空。

重叠运行时分别使用 `common_compute_resources_drained()` 与
`common_transport_resources_drained()`，不把无关计算 idle 当作页面提交条件。
旧串行路径保留 `common_barrier_resources_drained()`。一条 MEM 在物理周期 P 发射，其最后 tile 在
P+tile_rows-1 提交，最早在下一 dispatch 边沿恢复。存在额外 SR 路径或其他
搬运时必须检查实际排空状态，不能仅凭该最小延迟推断安全。

独立 `mem_transport_iq(queue)` 使用有限原生控制 RAM，不占用 raw Macro i-MEM。
`push_mem_transport_synchronized_raw(queue, packet)` 装入既有 MEM_SYNC 编码；
原 `notify_mem_synchronized`/`mem_accepts_synchronized_notification` 接口优先匹配
此通道。MEM transport 的通知年龄和 D+1 到达边界使用 P，HOLD 期间继续运行。
计算和 transport 向同 bank 同拍发射会显式失败。计算暂停期间 C2C/DDR、事件处理
和真实 SR→MEM commit 仍正常推进。没有 transport 的默认程序不分配额外控制队列。

## 重叠来源、提交反馈和资源归属

计算到达 E 且前序页已经提交并交还资源后，运行时释放本页 DMA/RX launch；计算继续。
首版保持页级按序，不增加乱序缓冲、MEM 端口、SR 数量或 DMA FIFO。接收仍受实际
同步 MEM 控制与有限 FIFO 的背压约束，计算/搬运同 bank 同拍冲突仍失败。

`configure_mem_transport_owner(queue, pc, generation, launch)` 将 Native transport
控制包映射到实例；发射的 `MemInstruction` 携带控制侧带，不改变 ISA 字段。
`configure_transport_commit(generation, launch, slice, bank, vectors)` 预置有限
scoreboard。owner 表及单 MEM array scoreboard 分别最多 65535 条，计数和乘法
溢出拒绝。期望值为 vectors×tile_rows；只在真实 SRAM `write_segment` 后增加，
最终 tile 未提交、计数不足、未注册实例或过量提交都不能提前 ready。

C2C RX 的 `StreamOutputPort` 显式设置 `transport_ingress=true`，SR cell 保留
transport 来源并沿被动桥传播；来源与调试名称、trace 开关无关。SR fabric 分别
提供 `compute_empty()`、`transport_empty()`，MEM array 分别提供
`compute_idle()`、`transport_idle()`。运行时核对所有目标提交计数并确认搬运
尾部/RX 排空后，才调用 `notify_common_mem_commit`。因此持续计算期间页仍能完成。
同页重装使用新 launch/page-use，任务切换清理状态并更新 generation。

最终链接和装载器必须根据实际 MEM/SR 指令证明 [E,C) 的共享资源预留及 SRAM
生存期；迟到搬运在 HOLD 中继续占有资源，提交并交还资源后才恢复。CModel 的来源
区分不能替代静态证明，也不允许通过临时停计算掩盖资源冲突。

链接器仍须证明分页覆盖不破坏恢复后活跃数据、bank/row/SR 资源无冲突、以及
释放事件不依赖 C 之后被冻结的计算。本地排空检查不能替代这些全程序证明。

`unit.icu.common_barrier` 覆盖聚合/早晚到/重复/旧 generation/重复装入事件、
连续屏障和空表、多个活跃 Context、单 Macro 跨 HOLD、不同解码阶段、未返回的
Native/Macro 取指、Native 与 raw Macro 同停、D+1 transport 握手、共享端口冲突、
真实 SR→SRAM 写入后恢复及不安全边界拒绝。完整分页 decoder 由 SOFTWARE 集成测试
验证，不能以此单测替代。

最终 CMODEL 定向回归 10/10 通过，日志 `out/overlap-final-regressions.log`，
覆盖实际提交、trace 关闭/无关计算持续、显式端口来源及既有前端和搬运协议。
SOFTWARE 的 seq32/seq128 完整 attention+FFN 在 51200/25600 MB/s 两档下，
均实际执行分页并与 CPU 校验后的原生基线逐字节比较通过；同调度串行对照也通过。
主执行 P 分别为 97990/122016（seq32）、411072/427101（seq128），相对同调度串行
减少 10.52%/10.11%、3.00%/5.09%。完整数据、启动和存储成本、CSV 图片及限制见
SOFTWARE main 的 `docs/common_transport_overlap_validation.zh-CN.md`。

当前仅验证 lpu32、非零模式权重、单层 decoder 与两种长度；不代表真实预训练
模型、多层/多芯片或 RTL 验收。仍不支持任意流水线透明抢占、运行时屏障表 refill、
跨任务搬运重叠与跨 invocation lookahead。
