# B 阶段 ICU FIFO 深度实验规范

日期：2026-11-15  
状态：设计冻结候选；B0 只完成规范，未实现 Unbounded，未执行深度扫描  
基线：`FTLPU-CMODEL-DEV/main@ff1aa41321e07667ebd37f3ebefdf8053f0ee0e5`

## 1. 研究问题与边界

目标是在真实周期模型和代表性 LPU 工作负载下，分别回答：

1. **Unbounded**：若普通数据 FIFO 不因容量而 Full，各级自然积压是多少？
2. **Finite**：给定有限深度后，第一处容量瓶颈在哪里，反压如何传播，完成周期/吞吐率损失多少？
3. **最终选择**：在吞吐率和延迟约束下，如何减少 `Σ(depth × width)`；Peak Occupancy 只是候选上界，不自动等于硬件深度。

严格保留：Clock/SystemPhase 四阶段、Current/Next、producer/consumer 每周期能力、DDR/SRAM 延迟、仲裁、顺序、一跳 Relay、FU/Bank/Context 等非容量资源限制。

本阶段规范不允许用 `SIZE_MAX`、巨大常数、百万 entry 或无限制 `std::deque` 冒充 Unbounded；不允许为了得到较小 occupancy 改变服务率、延迟或调度顺序。

## 2. 显式容量策略

建议所有纳入实验的数据 FIFO 使用同一策略接口，而保留各自的时序实现：

```cpp
enum class FifoCapacityMode {
    Finite,
    Unbounded,
};

struct FifoCapacityPolicy {
    FifoCapacityMode mode = FifoCapacityMode::Finite;
    std::size_t finite_depth = 0;  // Finite 时必须 > 0

    bool capacity_full(std::size_t occupancy) const {
        return mode == FifoCapacityMode::Finite &&
               occupancy >= finite_depth;
    }
};
```

这只是接口契约，不规定具体容器。实现必须满足：

| 契约 | Finite | Unbounded |
|---|---|---|
| 容量性 Full | 按配置深度与该 FIFO 的 RTL replacement 规则 | 永不由存储容量触发 |
| Ready/Valid | 完整保留 | 只去掉容量导致的 ready 拉低；其他 ready 条件保留 |
| 每周期读/写端口数 | 保留 | 保留 |
| Producer/Consumer 速率 | 保留 | 保留 |
| Pipeline/Relay latency | 保留 | 保留 |
| 数据顺序 | 保留 | 保留 |
| 协议/资源限制 | 保留 | 保留 |
| Occupancy/Peak | 统计 | 统计 |

Unbounded 只取消“没有存储 entry”这一原因造成的阻塞，不能让一个 FIFO 单周期接受更多元素，也不能让数据跨过原有 pipeline stage。

### 2.1 同拍 Enqueue/Dequeue

每个 FIFO 继续遵循自己的 RTL 契约：

- DMA RX FIFO：Finite 模式保留 full+dequeue replacement；Unbounded 模式不再 capacity full，但仍最多接受/弹出原定数量。
- 当前 Relay：Finite 模式保留 registered-ready、full 时不做 replacement；Unbounded 只消除容量阻塞，仍保持一拍一跳、原先写/读带宽与顺序。
- 其他 FIFO：必须先从现有实现提取 replacement/端口契约，不能用统一容器行为覆盖模块差异。

## 3. 实验对象与非实验资源

此分类已提交“CMODEL 建模”聊天审阅并获批准。

### 3.1 普通数据 FIFO：允许独立 Unbounded

| FIFO | 当前默认深度 | Unbounded 处理 | 标识粒度 |
|---|---:|---|---|
| Instruction DMA RX FIFO | 16 | 去掉容量性 Full | 每个 Instruction DMA |
| Program Word FIFO | 8 | 去掉容量性 Full | 每个 ProgramLoader |
| Distribution Ingress | 2 | 去掉容量性 Full | 每个 ingress/目标域 |
| East/West Relay | 2 | 每一级独立去掉容量性 Full | hemisphere × relay index |
| local iMEM Response FIFO | 4 | 去掉容量性 Full | FU kind × side × logical queue |
| Macro Word FIFO | 4 | 去掉容量性 Full | FU kind × side × logical queue |
| Macro Context FIFO | 4 | 去掉容量性 Full | FU kind × side × logical queue |
| Timed Issue FIFO | 16 | 去掉容量性 Full | FU kind × side × logical queue |

当前默认值来自 `config/ftlpu-lpu32.json:4-39,257-271`。不同 FU、side、queue 不得错误共用一份 occupancy 或深度参数。

Native IQ（当前 16 entries）属于 CModel local frontend 中的容量性缓冲，可作为**单独敏感性实验**；它不是 Macro frontend 三 FIFO 的替代项，结果必须单独报告。

### 3.2 协议/执行资源：默认保持有限

| 资源 | B 阶段默认 | 原因 |
|---|---|---|
| DMA Outstanding Credits / ID FIFO | **固定 16** | 同时规定 DDR 最大未完成请求并发度；无限化会改变请求接受率、DDR 响应分布和 RX 积压，不是单纯取消缓冲容量 |
| Reassembly buffer | 固定 64 | Packet/事务组装状态与最大包约束；不是普通弹性 FIFO |
| Command/config/commit/prime 状态 | 保持现状 | 控制协议、提交延迟和启动语义 |
| local iMEM SRAM | 保持当前容量/地址契约 | 本实验研究传输 FIFO；不得同时改变 SRAM 容量 |
| Macro reservoir | 固定 3 | Parser/位流状态，不等于普通 entry FIFO |
| DDB | 固定 8 | 解码/依赖资源 |
| Active Context | 固定当前配置（4） | 执行上下文上限，不是普通缓冲容量 |
| DDR bandwidth/latency/arbiter | 保持现状 | 外部存储服务能力 |
| SRAM bank conflict/ports | 保持现状 | 真实结构冲突 |
| FU issue/execute/busy | 保持现状 | Consumer 服务率 |

Commit queue若只表达提交延迟/准入控制而不承载可积压 payload，不纳入普通 FIFO Unbounded；实际实现前须逐项核实。

## 4. 配置组织

配置必须与功能硬件配置分层，默认 Finite 值继续对应 Stage A/RTL Golden；没有显式实验覆盖时行为完全不变。建议的逻辑结构：

```json
{
  "experiment": {
    "mode": "finite",
    "workload": "program_loader_delivery",
    "trace": {
      "per_cycle_fifo": true,
      "event_trace": true
    },
    "fifo": {
      "instruction_dma": { "rx": { "mode": "finite", "depth": 16 } },
      "program_loader": {
        "word": { "mode": "finite", "depth": 8 },
        "ingress": { "mode": "finite", "depth": 2 },
        "relay": {
          "east": [{ "mode": "finite", "depth": 2 }],
          "west": [{ "mode": "finite", "depth": 2 }]
        }
      },
      "local_frontend": {
        "mem": {
          "east": {
            "0": {
              "response": { "mode": "finite", "depth": 4 },
              "macro_word": { "mode": "finite", "depth": 4 },
              "context": { "mode": "finite", "depth": 4 },
              "timed_issue": { "mode": "finite", "depth": 16 }
            }
          }
        },
        "mxm_load": {},
        "mxm_dequant": {},
        "mxm_compute": {},
        "vxm": {},
        "sxm": {},
        "c2c": {}
      }
    },
    "protocol_limits": {
      "dma_outstanding_credits": 16,
      "reassembly_entries": 64
    }
  }
}
```

要求：

1. `mode` 可在全局给默认值，也可按 FIFO override；最终解析结果必须把每个物理/逻辑实例展开为明确配置。
2. East/West Relay 逐级独立；MEM/MXM Load/MXM Dequant/MXM Compute/VXM/SXM/C2C 独立。
3. 配置快照写入每次结果 manifest；未识别 key 必须报错，不能静默共用或忽略。
4. Unbounded 时仍保存 `finite_depth_reference`，便于同一 Trace 判断“若按默认深度会在何处 Full”，但不得用它影响 ready。

## 5. 统一逐周期 Trace

每个 FIFO 每周期至少输出一行，采样契约统一为“本周期边沿前事件 + 提交后 occupancy”：

| 字段组 | 必需字段 |
|---|---|
| 身份 | `run_id, workload, cycle, fifo_id, fifo_class, side, fu_kind, queue_index, relay_index` |
| 配置 | `capacity_mode, configured_depth, width_bits` |
| 状态 | `occupancy_begin, occupancy_end, empty_begin, full_begin, peak_so_far` |
| 写端 | `enq_valid, enq_ready, enq_fire, enq_count` |
| 读端 | `deq_valid, deq_ready, deq_fire, deq_count` |
| 阻塞原因 | `capacity_stall, protocol_stall, downstream_stall, upstream_idle, resource_stall, stall_cause` |
| 关联 | `transaction_id/program_destination/queue_pc`（可用时） |

硬性守恒检查：

```text
occupancy_end = occupancy_begin + accepted_writes - accepted_reads
0 <= occupancy_end
Finite: occupancy_end <= configured_depth
Final drain: initial + total_writes - total_reads = final occupancy
```

事件字段必须说明是 pre-edge 接受事件，occupancy_end 必须是统一 commit 后状态；HIGH 只观察，不得重复 enqueue/dequeue 或准备 Next。

## 6. 聚合统计定义

### 6.1 Unbounded

每个 FIFO 独立报告：

- 每周期 Occupancy、Peak Occupancy、Peak Cycle；
- Average Occupancy = `Σ occupancy_end / observed_cycles`；
- Occupancy Histogram（0..peak，每个 bucket 的 cycle count）；
- Total Accepted Writes/Reads；
- Producer/Consumer Rate（accepted count / 明确的 active window 与 total window，两者分列）；
- Maximum Consecutive Growth：occupancy 连续净增加的最长区间及净增量；
- Maximum Consecutive Drain：occupancy 连续净减少的最长区间及净减量；
- 由非容量资源造成的 stall，继续分类统计。

Unbounded 结果若持续增长到 workload 结束，不得直接将终点 occupancy 当作稳定容量需求；必须判断工作负载是否包含 drain phase，并报告净生产率。

### 6.2 Finite

每个 FIFO 独立报告：

- Configured Depth、Peak/Average Occupancy、Full/Empty Cycles；
- Accepted/Rejected Writes、Accepted Reads；
- Producer Stall Cycles、Consumer Idle Cycles；
- Capacity Backpressure Duration、Maximum Consecutive Stall；
- 首个 Full cycle；反压首次传播到每个上游 stage 的 cycle；
- Completion Cycles、首/末有效事务、Latency、Throughput；
- 与默认 Finite 和 Unbounded 的差值。

`Rejected Write` 仅表示 producer valid 且因容量未接受，不能丢弃事务；事务必须保持/重试直到 handshake，除非原 RTL 契约另有明确行为。

## 7. 工作负载层次

### Level 1：基础且可复现

| 子基线 | 输入/程序 | 涉及资源 | Golden | 用途 |
|---|---|---|---|---|
| L1-A ProgramLoader Delivery | `subsystem.program.loader` 的确定性 ProgramImage | DMA RX、Word、Ingress、East/West Relay、local iMEM | 170 load cycles + 结构断言 | 验证上游 occupancy、反压、路由 |
| L1-B MEM/MXM | `kernel.prefill_weight_streams`，T/K/N=32/128/64 | direct local iMEM/IQ、MEM、MXM Load/Dequant/Compute | 6 cases，数值误差 0 | 验证真实 FU 消费率与 local queue 观测 |
| L1-C Macro/SXM | `subsystem.sxm.mem_transpose` | packed Macro、MEM、SXM、双半区 | 96 matrices | 验证 Macro frontend 与 SXM 消费 |

这些是三个独立运行，不能合成端到端结果。

### Level 2：混合并发

正式进入条件：同一 ProgramImage 通过 ProgramLoader 产生可被真实 MEM/MXM（必要时 SXM/VXM）消费的指令，覆盖多个 destination、East/West、连续装载与执行，并具有数值或结构 Golden。当前尚无满足全部条件的现成入口；建立它属于 B0 后续任务。

### Level 3：完整 Attention

目标规格：Sequence=128、Hidden=576、Q Heads=9、KV Heads=3、Head Dimension=64，覆盖 Q/K/V Projection、RoPE、QK、Softmax、P×V、Output Projection。必须同时具备：

- 当前 CModel 支持的 ProgramImage/物理编码；
- ProgramLoader 到 local iMEM 的完整供给链；
- 真实 FU 周期执行（不能用 C++ reference 替代关键阶段）；
- 独立输入/权重/scale/binding；
- 数值 Golden；
- 完整 FIFO/Issue Trace。

历史 81,273-cycle schedule 只作恢复参考，未复现前不进入正式数据集。

## 8. 实验顺序

1. **B0（本次）**：冻结 workload 审计、资源分类、配置与 Trace 规范。
2. **B1 Observability**：只增加观测层；以全部默认 Finite 参数运行 Stage A Golden 和定向回归，要求周期、输出、Trace 事件完全不变。
3. **B2 Unbounded micro/delivery**：分别运行 L1-A/L1-B/L1-C；普通数据 FIFO逐项和组合 Unbounded，协议资源保持固定；先验证守恒与终止，再分析 occupancy。
4. **B3 End-to-end workload**：建立 Level 2；随后恢复/移植 Level 3 完整 Attention。没有数值 Golden 和完整链路不得升级层级。
5. **B4 Finite sweep**：在通过的 workload 上先单 FIFO扫描，再做少量组合点；典型候选深度 2/4/8/16/32，但必须包含当前 RTL 默认点。
6. **B5 Selection**：建立 Pareto 对比，按吞吐/延迟门槛筛选，再比较总 `depth × width`；不未经综合宣称面积/PPA收益。

## 9. 正式验收标准

### B1 观测层

- 默认 Finite 参数下，Stage A DMA 108-cycle 比较仍 `mismatches=0, semantic_failures=0`；ProgramLoader 仍为 170 cycles。
- 所有既有 passing workload 的 issue cycle、最终结果、完成周期不变。
- 每个 FIFO 的逐周期守恒成立，aggregate 可由 Trace 独立重算。
- 关闭 Trace 时不改变行为；HIGH 不产生重复推进。

### Unbounded

- 只有 capacity full/ready 分支失效；端口数、latency、ordering、服务率和协议限制不变。
- Outstanding Credits 恒为 16；DDR、Reassembly、Active Context、SRAM、FU 限制保持当前值。
- 无丢失、重复、乱序；最终 drain 和守恒检查通过。
- 若 completion cycle 改善，必须能用被消除的容量 stall 解释；若无改善，也应如实报告。

### Finite

- 默认深度点必须复现原基线；每个扫描点均通过数据/指令守恒和相同功能 Golden。
- 报告首个 Full、反压传播、stall cause、完成周期与吞吐变化；不能只报告 Peak。
- 深度建议必须来自至少一个完整供给链 workload 和一个真实 FU 消费 workload；Level 3 缺失时明确标为 provisional。

## 10. 分析和报告原则

1. 分开报告物理 iMEM fetch、frontend FIFO、IQ pop、Macro/Context 展开与 FU issue；不得互相替代。
2. 分开报告 Issue Utilization 与 FU Busy Utilization；没有可靠 busy 信号时只报告 issue。
3. 区分容量性 backpressure、协议 credit、资源 busy、bank conflict、producer idle。
4. ProgramLoader Delivery 与 Direct ICU FU/Macro 是互补实验，禁止拼接 Trace 或把它们称为完整 Attention。
5. 任一配置、源码、输入、ProgramImage、权重和 Golden 均写入 manifest；结果必须可复现。
6. B0 到此停止；不得据此自动开始实现 Unbounded、Finite sweep 或完整 Attention 开发。
