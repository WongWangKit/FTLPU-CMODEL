# ICU Macro Scheduler CModel 说明

本次 CModel 在 `DistributedIcuQueue` 中增加了粗粒度描述符执行路径，覆盖
MEM、MXM load、MXM dequant 和 MXM compute 队列。

每个描述符包含一条原生功能单元指令、绝对 `start_cycle`、内外两层
count/interval/stride，以及需要递增的原生指令字段。ICU 在目标 cycle 才将
展开后的指令送给功能单元，因此功能单元看到的逐 cycle 行为与旧队列一致。

为了支持 FFN 中不同 wave 的穿插，同一队列允许多个 macro 同时 in-flight。
CModel 使用按 next-issue-cycle 排序的最小堆：每 cycle 只检查最早到期项，
发射后更新其内外层坐标并重新入堆。若同一 cycle 有两个到期项、描述符取指
过晚或目标 cycle 已错过，则抛出 `StaticScheduleError`。

相关接口：

```text
InstructionControlUnit::enqueue_mem_macro
InstructionControlUnit::enqueue_mxm_load_macro
InstructionControlUnit::enqueue_mxm_dequant_macro
InstructionControlUnit::enqueue_mxm_compute_macro
```

单测 `icu_macro_schedule_test` 同时覆盖二维地址递增和两个描述符交错发射。
完整软件侧 Vector FFN 测试继续检查最终 BF16 非零数值与 golden，而不仅是
队列能够走完。

