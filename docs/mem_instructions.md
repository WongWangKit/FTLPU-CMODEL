# 参数化 MEM 指令与内部编码

> 同步自 FTLPU-VMODEL。文中的 RTL 路径和验证命令以 VMODEL 仓库根目录为准。

修订：0.3。日期：2026-11-11。适用于 `rtl/mem_native` 的 32 位入口。
指令时序、模式交接和编译器空拍规则见 [mem_scheduling_contract.md](mem_scheduling_contract.md)。

`native` 表示 MEM 核心内部的标准化指令格式，不是版本号。
当前参数化集成测试的发令链为“参数化原始 32 位指令 → mem_c7_primitive_adapter → 内部 native 32 位”。
`delay3` 已通过现有参数化原始入口的合法性检查与透传，并在内部核心实现。
适配器保持组合转换，不增加发射拍数。旧冻结 C7 编码使用显式兼容配置保留原语义。

## 参数化配置

仓库的 `config/native_profiles.json`、`rtl/core/lpu_native_config_pkg.sv` 定义以下档位。
各档位的指令容器仍为 32 位，内部 row 字段仍为 `[29:15]`，但有效地址范围随深度变化：

| 档位 | Superlane 数 | 每个 Bank 深度 | 有效地址位数 | 内部 row 高位要求 |
|---|---:|---:|---:|---|
| LPU16 | 2 | 1024 | 10 | `[29:25]=0` |
| LPU32 | 4 | 2048 | 11 | `[29:26]=0` |
| LPU64 | 8 | 2048 | 11 | `[29:26]=0` |

若有效深度另有缩小，地址还须小于 `P_MEM_ACTIVE_ROWS`。当前参数化实现约束地址范围，
没有随档位压缩整个指令字宽度。

## 内部标准位域

| 位 | 名称 | 含义 |
|---|---|---|
| `[31]` | preserve | Write 时 0 为消费输入，1 为 WriteTap；Read 必须为 0 |
| `[30]` | delay3 | 0：每个 Superlane 交界处传播 1 拍；1：传播 3 拍 |
| `[29:15]` | row | 15 位 SRAM 行地址，须小于物理深度和有效深度 |
| `[14:9]` | reserved | 必须为 0 |
| `[8]` | direction | 0 为 East，1 为 West |
| `[7:3]` | stream_index | Stream Register 编号 0～31 |
| `[2:0]` | opcode | 0 为 Read，1 为 Write；其余编码非法 |

指令字仍为 32 位；只将原保留位 `[30]` 用作 `delay3`。
Bank 由独立入口选择，Superlane 由纵向传播到达的位置确定，不另占指令位。
valid 是入口与流水上的独立信号，不在指令字中；没有 COUNT 或重复执行字段。
参数化原始 32 位指令使用相同的操作、地址、方向、Stream、preserve 和 delay3 位域；
区别在于原始 `[14:9]` 接受输入映射位，转换为内部标准格式时将其归零。

## 操作语义

- Read：读取本地 SRAM 行，按 direction/stream_index 产生带有效位的读响应。
- Write：从所选 Stream Register 接收完整行，写入 SRAM 并消费输入；缺少有效数据时报告故障。
- WriteTap：与 Write 使用同一 opcode，设置 preserve=1；写入但不消费输入。
  输入无效时是本地空操作，地址等合法性检查仍生效。

`delay3` 只改变指令到达下一 Superlane 的时间；不改变本地操作、SRAM 响应延迟或数据宽度。
每条有效指令在每个 Superlane 执行一次，两种模式均支持每拍一条。
连续相同字仍是连续的多个操作。模式随指令传播，不修改其他在途指令的模式。

示例：`0x00000000` 为短模式 Read 行 0、East Stream 0；
`0x40000000` 为相同本地操作，采用 3 拍层间传播。

## 发令入口与适配边界

- 旧 47 位 VMODEL 适配器保持原来的编码语义，生成 native preserve=0、delay3=0。
  旧 `[30]` 属于地址，仍检查是否超出 native 15 位地址范围，不能重解释为模式位。
- 参数化原始 32 位入口使用 c7 编码及按 SRAM 深度生成的合法位掩码。
  其 `[14:9]` 为输入映射位，经适配器归零；其 `[30]` 为本次扩展的 delay3，保持透传。
  这是参数化验证链实际使用的入口，不能仅因名称含 c7 就视为已淘汰的旧入口。
  `mem_c7_primitive_adapter.P_ALLOW_DELAY3` 默认 1，启用当前扩展格式。
- 设置 `P_ALLOW_DELAY3=0` 时，适配器严格校验冻结的原始 C7 格式，将 `[30]` 仍视为保留位。
  该配置用于旧 CModel 编码参考回归，不是最新参数化发令链的默认配置。
  `sim/mem_p1/adapter_tb.sv` 和 `sim/mem_profiles/codec_tb.sv` 显式选择此兼容配置，
  原始 CModel 参考生成器及历史 golden 保持不变。

验证完整原始指令入口：

```powershell
python scripts/test_mem_parameterized_ingress.py
```

该测试覆盖参数化地址与模式编码、严格旧 C7 兼容、原始指令→适配器→2/4/8 层 MEM 的读写和时序，
以及四层原始入口→适配器→实际 MEM/SRF 的逐拍数据前沿。
结果保存在 `build/mem_parameterized_ingress/results.json`，与此前内部入口的验证结果分开记录。
