# v39 队列伪控制：CModel 实现说明

当前 main 的默认存储已切换为 v40 共享 QCTL 池（512 条、每队列 4 条缓存），并接入定时 DMA 装载。本文保留 v39 控制语义与历史验收；当前集成范围以 SOFTWARE 的 `docs/queue_control_shared_pool_main.zh-CN.md` 为准。

本 worktree 的默认目标已改为 [共享池＋4条缓存](queue_control_shared_pool.zh-CN.md)，以下独立 RAM 参数描述旧 v39 目标。

对应 SOFTWARE 的 `docs/queue_control_v39.zh-CN.md` 与 `docs/queue_control_v39_validation.zh-CN.md`。旧 v37/v38 BARR 的全局暂停语义保留；新 QCTL 不关闭全片 `compute_enable`。

2026-11-02 缩为 16 条 RAM＋8 条缓存后，定向控制测试及 seq32 Native/Macro 数值、逐拍复验通过；详细结果见 SOFTWARE 的 `docs/queue_control_16x8_validation.zh-CN.md`。历史 seq128 结果不作为本次缩容后的执行证据。

2026-10-30 合入当前 main 后，链式 VXM 的输入捕获/流水由 `VxmSlice::idle()` 检查。定时程序装载器仍仅搬运 packed MEM/MXM i-MEM，不装载 QCTL RAM；SOFTWARE 对 `load_timed(QCTL)` 显式拒绝，QCTL 使用直接装载及有限 PRIME。旧 v38 的显式 SR 资源证明不支持新链式 VXM，QCTL 则复用 Native 相位计划。集成复验见 SOFTWARE 的 `docs/queue_control_main_integration.zh-CN.md`。

- `DistributedIcuQueue::execute_synchronization()` 是 Native 与伪控制的共用实现。WAIT_EVENT 消费匹配令牌后可以同拍执行后继；SYNC 释放和 NOTIFY 占用一拍。普通 NOTIFY 保留多令牌语义。
- Raw Macro 的 `cycle()` 是本队列 `Lq`。控制在 `issue_active_macro()` 前检查，等待时保留活跃 Context、循环、解码状态；未返回请求仍捕获响应。其他队列和所有已发出的 FU 流水继续。
- `IcuQueueControlFrontend` 为每队列有限 RAM/缓存，历史独立 RAM 模式深度 16、缓存 8、读延迟 2、每拍一个读请求。缓存保存 RAM 中已取回的记录，不增加程序容量；超出 16 条明确拒绝，没有运行中控制 RAM 换页。没有全局单游标；连续同拍控制必须已经在缓存中。供给不足显式报错。Native 绑定留在 SOFTWARE 作源指令及释放计划校验，不占伪控制 RAM；其容器分区上限独立为 256 条。
- `MEM_PACKET` 是 QCTL 侧带操作（值 8），不占用 Native ISA opcode。它在原生队列边界激活独立 transport store，沿用 MEM 同步包握手、reservation、退休时序。
- `configure_mem_transport_owner()` 把 generation/launch 标记带到实际 MEM 指令。MEM SRAM 逐 tile 提交计数不依赖 trace 开关。包退休、全部提交、资源释放是不同状态。
- `IcuQueueCompletionEvents` 用 generation＋实例区分页完成，支持 all-of、重复幂等、旧任务隔离及重新装入；与普通通知令牌分开。SOFTWARE 在确认全部实际提交后投递实例完成，并按共享 Native 相位算法逐组释放队列。
- 装载后先完成 Native `launch()` 初始化再允许外部投递事件，防止第一拍前的通知被 lazy initialization 清空。此操作不增加 FU 发射拍。

定向测试 `unit.icu.queue_control` 包括完整指令/物理拍 Native 对照、早晚事件、连续 WAIT、SYNC/NOTIFY、两个活跃 Context、独立队列推进、有限读请求返回、共享 MEM 顺序、尾部 NOP、generation/all-of、消费者在 WAIT 时继续接收已发数据，以及 MEM 包已退休但最后 tile 尚未提交。

`unit.icu.barrier` 和 `unit.icu.common_barrier` 保留旧协议验证。SOFTWARE 的真实 seq32/seq128 attention＋FFN 对照在两档带宽和额外延迟条件下通过；详细物理周期、PRIME、存储、数值及原始证据位置见 SOFTWARE 验证文档。

不支持任意流水线抢占；不对任意手工控制表作全程序安全承诺。有限前端不供给时不插入隐藏空拍；未支持的 coarse-domain/native4 组合由 SOFTWARE 明确拒绝。
