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

## Macro packed-v1 物理前端

`DistributedIcuQueue` 还提供独立于语义 `Entry` i-MEM 的 raw Macro 路径。
MEM 使用 96-bit word，MXM load/compute/dequant 使用 128-bit word。每个 word
由低到高排列 32-bit lane；lane 0 保存 `word[31:0]`，字段和 bitstream 均从
bit 0 向高位消费，字段跨 word 时从下一地址 word 的 bit 0 继续。

runtime 后续应依次调用：

```text
write_*_raw_macro_imem(queue, word_address, raw_word)
configure_*_raw_macro_imem(queue, word_count)
prime_raw_macro_frontends()
START / normal dispatch
```

raw i-MEM decoder位于 CModel 内部，不调用 SOFTWARE reference decoder。它按
Macro v1 grammar解析 control word、Delta Dictionary、首条绝对状态、Run、
Compact/Extended Template和Compact/Wide Escape。decoder使用四个固定 word
的有限 reservoir，每个 frontend cycle最多发起一次 i-MEM读取、最多创建一个
Macro context。context RAM满时停止消费下一条 record，并记录backpressure cycle。

启动预取不会推进 program cycle。`prime_raw_macro_frontends()` 让所有物理 queue
并行预解码，直到本 queue已经全部解码或其有限 context RAM已满。正常执行时，
calendar先发射本周期已经可见的context，decoder随后写入新context，因此新写入
context在下一周期才可见；若此时已经到达其`start_cycle`，CModel报告静态调度错误。

单测 `icu_raw_macro_decoder_test` 使用硬编码的 96/128-bit golden image覆盖：

- MEM dictionary run和跨96-bit word字段；
- MEM Compact/Extended Template；
- Compact/Wide Delta Escape；
- MXM load、compute和dequant解码；
- 每周期一word的有限取指；
- 单context配置下的context-full反压；
- raw decoder到现有next-issue calendar的发射周期与operand恢复；
- control word保留位检查。

