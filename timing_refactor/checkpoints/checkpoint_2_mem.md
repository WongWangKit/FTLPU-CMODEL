# Checkpoint 2：MEM 统一时序迁移

## 结论

Checkpoint 2 已完成源码迁移和回归。MEM 已进入与 Stream Fabric 相同的全局
validate/commit 屏障；原先在 `evaluate()` 中立即发生的指令 pop、流水移位、SRAM
写入、R1 响应替换和 cycle 增加，已经改成 LOW 准备、POSEDGE 提交。

本结论只证明定义的 CModel 行为契约和回归一致性。当前没有可靠 RTL/SRAM Macro 规格
可确认真实 port 数、冲突规则和物理延迟，因此不宣称 MEM 已成为 RTL 周期 Golden。

## 迁移前状态映射

| 对象 | 迁移前行为 | 判断 |
|---|---|---|
| `instruction_rows_` | `evaluate()` 内 dispatch、执行、立即数组移位 | 真实流水状态，需 Current/Next |
| `instruction_queues_` | 未限定深度 `std::deque`，立即 pop | 投递容器；物理 FIFO 身份未证实 |
| `read_responses_` | R1 模式中向量 swap | 读输出寄存器 |
| SRAM pages | 稀疏页式存储，Write 在 `evaluate()` 中立即更新 | 大型 Memory，需局部 write staging |
| `cycle_` | `evaluate()` 末立即增加 | 已提交周期状态 |
| execution/consumption Trace | 每周期清空并记录事件 | 软件观察状态 |
| transport commit scoreboard | 写函数内立即增加 | 必须与物理 SRAM write 同边沿提交 |

## 实现

- `MemArrayModel` 实现 `ClockedModule`。
- `instruction_rows_state_`、`read_response_state_`、`cycle_state_` 使用
  `ClockedState`。
- `pending_sram_writes_` 只保存本 LOW 接受的地址和 8-byte 数据；POSEDGE 才调用
  `Sram::write_segment()`，没有复制整个 Memory。
- `Sram::validate_segment()` 允许 LOW 完成地址/tile 校验而不修改 Memory。
- `dispatched_queue_slots_` 记录本 LOW 接受的入口，POSEDGE 才 pop。
- TspSliceSystem 的两个 MEM 与两个 Fabric 使用同一个全局 Clock；
  `SxmMatrixMemReference` 的本地 Clock 同样注册 MEM。

主要实现位置：

- `include/ftlpu/mem/mem_array.hpp:89`：ClockedModule 接口。
- `include/ftlpu/mem/mem_array.hpp:447`：LOW 准备和执行。
- `include/ftlpu/mem/mem_array.hpp:481`：提交前验证。
- `include/ftlpu/mem/mem_array.hpp:495`：POSEDGE 提交。
- `include/ftlpu/mem/mem_array.hpp:597`：Queue dispatch。
- `include/ftlpu/mem/mem_array.hpp:650`：Read 路径。
- `include/ftlpu/mem/mem_array.hpp:694`：Write 采样和局部暂存。
- `include/ftlpu/mem/mem_array.hpp:869`：流水 Next 移位。
- `include/ftlpu/system/tsp_slice_system.hpp:99`：系统 Clock 注册 MEM。

## 周期契约

1. LOW 只读取 SRAM 旧 Memory 和 Fabric 旧 Current。
2. Main Read 在 LOW 写 Fabric Next，随本周期 POSEDGE 可见。
3. RegisteredR1 Read 在 LOW 写 R1 Next；下一个 LOW 将旧 R1 Current 写 Fabric Next。
4. Write 在 LOW 捕获输入，POSEDGE 才更新 SRAM。
5. 同周期同地址 Read/Write 读取旧数据，随后写入，即 read-before-write。
6. 每个正沿指令流水只前进一个 tile；没有 stall/hold 机制。
7. Queue pop、SRAM write、transport counter、pipeline/R1/cycle 提交共享同一边沿。
8. HIGH 与 NEGEDGE 不修改这些正沿状态。

## 验证

| 验证项 | 结果 |
|---|---|
| Release 干净配置和全量编译 | PASS |
| MEM 单元测试 | 8/8 PASS |
| TSP/SXM 受影响子系统 | 4/4 PASS |
| 新增 `unit.mem.clocked` | PASS |
| fast（排除既有 RMSNorm） | 60/60 PASS |
| 原始目录 861 文件复核 | 0 missing / 0 changed |

`unit.mem.clocked` 覆盖：

- LOW 后 SRAM 不变、POSEDGE 后写入可见。
- 指令在 8 个 tile 间逐边沿推进，并与独立 reference 数组逐周期比较。
- R1 响应先提交到 R1 Current，再在下一边沿进入 Fabric Current。
- 正序/逆序 Clock 注册得到相同写入和周期结果。
- 第三个未准备模块令 validate barrier 失败时，MEM/Fabric 均保持未提交。

完整 fast 仍包含一个冻结基线失败：

`kernel.rmsnorm: VXM input Buffer cannot be reconfigured while a Bundle is active`

错误文本未变化，本批没有修改 VXM 来掩盖它。

## 限制和下一步入口

- `Sram` 注释称 single-port，但现有 MEM 行为允许每 bank 的 Read/Write 端口同周期活动；
  缺少硬件证据，不能据此确定真实 Macro。
- 未确认真实 read latency、write latency、byte enable、端口冲突和 initiation interval。
- `instruction_queues_` 仍是无界投递容器，不应当作最终物理 Queue 深度模型。
- 没有可靠 RTL，本批没有执行 L3 逐周期 RTL 对照。

Checkpoint 3 开始前应先完成 MXM Control、Datapath、Accumulator 和 Activation/Weight
状态映射；不要从 MEM 的时序契约机械推导 MXM。
