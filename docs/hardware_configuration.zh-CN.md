# JSON 硬件配置指南

FTLPU-CMODEL 与 FTLPU-SOFTWARE 使用同一份 JSON 描述物理硬件 target。两个工程在
CMake 配置阶段读取它并生成编译期常量，Software 编译器也可以通过命令行读取同一文件。

## 默认配置和 JSON 格式

默认配置为 [`config/ftlpu-lpu32.json`](../config/ftlpu-lpu32.json)。未指定
`FTLPU_HARDWARE_CONFIG` 时，CModel 使用该文件；Software 使用
`${FTLPU_CMODEL_DIR}/config/ftlpu-lpu32.json`。默认内容如下：

```json
{
  "schema_version": 2,
  "target": {
    "name": "ftlpu-lpu32"
  },
  "topology": {
    "hemispheres": 2,
    "tiles_per_slice": 4,
    "lanes_per_tile": 8
  },
  "mem": {
    "slices_per_hemisphere": 52,
    "banks_per_slice": 2,
    "rows_per_bank": 8192,
    "bytes_per_lane": 1
  },
  "sr": {
    "registers_per_lane": 64,
    "bytes_per_stream_per_lane": 1
  },
  "mxm": {
    "accum_contexts": 4,
    "supported_modes": [
      "native4x4",
      "linear1x16"
    ]
  },
  "vxm": {
    "alus": 8
  }
}
```

### `stream_topology`（schema v2）

`stream_topology.fabric_template` 描述每个 hemisphere 共用的物理 SR 网络模板：

- `replicas` 给出模板实例名，数量必须等于 `topology.hemispheres`。
- `columns` 按稳定 ID 顺序声明物理 SR column；descriptor 中的端口和 route 都使用该 ID。
- `paths` 是连续链路的简写。生成器按 `columns` 顺序展开 `forward` 逐跳 route；当
  `bidirectional` 为 `true` 时同时生成 `reverse` route。path 中的 column 必须遵循
  `fabric_template.columns` 的顺序，因此不需要额外声明方向。
- `routes` 声明无法用连续 path 表达的显式 route，例如 bypass。字段为 `name`、
  `source`、`destination`、`flow`、`kind`、`enabled_by_default`、
  `multicast_allowed` 和 `latency_cycles`。

`stream_topology.bindings` 把 MEM boundary、SXM、MXM、VXM 和 C2C 端口绑定到 column；
端口由 `{"column": "...", "flow": "forward|reverse"}` 表示。`forward` 表示沿
column ID 递增方向传播，`reverse` 表示沿 column ID 递减方向传播；两者都是 fabric
局部方向。
`system_transfers` 描述 fabric 实例之间的传输，source/destination 还包含 `fabric`。
数组中存在的 transfer 即被生成并执行；不需要传输时应从数组中删除对应项。当前支持的
跨 fabric 类型为 `passive_bridge`。完整的 LPU32 实例见
[`config/ftlpu-lpu32.json`](../config/ftlpu-lpu32.json)。

`stream_topology.bindings.mem.slices_per_group` 独立于
`topology.tiles_per_slice`：前者定义 MEM slice 共用 SR 边界的分组，后者定义
数据通路的 tile 数。分组必须为正整数，且整除 `mem.slices_per_hemisphere`；
MEM 边界数量为 `slices_per_hemisphere / slices_per_group + 1`。
例如 52 slices、每组 4 slices 始终对应 13 组和 14 个 MEM 边界，增加 tile 数
不会改变这些 SR 边界。compiler、CModel 和 target ABI 都读取独立的分组配置。

SXM 编码按物理 lane 数派生 Permute map 位宽和包长度：4 tiles × 8 lanes
保留原有 5-bit map、4 个 96-bit ICU word；8 tiles × 8 lanes 使用 6-bit map、
6 个 ICU word。后者使用版本 1 的三位 continuation word index；Transpose
仍为 2 个 ICU word。旧 FU 编码也随 lane/tile 选择器位宽扩展。
MXM 地址、column 和 stride 位宽也从物理几何派生：8-tile 使用 14-bit
accumulator 地址、3-bit column 和 15-bit signed stride，Compute ICU 包为
3 个 128-bit word；4-tile 保留原来的 2 个 word 和字段布局。ICU 到 FU 的
编码使用同一几何定义。ABI 的编码版本字段在 4-tile 为 1、宽格式为 2，
runtime 会拒绝不匹配的版本。模型是否完成整层验收，以实际运行报告为准。

`c2c.beat_bytes` 独立指定每条 C2C lane 每拍传输字节数。未设置时默认
32 B（小于 32 B 的物理 vector 使用其自身宽度）；值必须为正且不大于
物理 vector 宽度。默认配置及 8-tile 实验都保持 32 B，因此 32 B vector
一拍完成，64 B vector 两拍完成；完整 vector 收到后才交给片内 SR。
DDR 使用同一 beat 宽度，字节带宽与延时仍由 `external_memory` 独立配置。
DMA 请求携带物理 C2C hemisphere/lane/direction 身份，DDR 调度每周期最多
服务该方向的同一 lane 一次；增加 outstanding 请求不会增加链路带宽。
不同 lane 仍可并行，且共享 DDR 总字节预算和 request queue。
默认使用 4 accumulator contexts，每个 MXM 的 FP32 accumulator 容量
从 32×32×4×4 = 16 KiB 派生为 64×64×4×4 = 64 KiB。

`fabric` 名称 `east`/`west` 表示全局物理 hemisphere；`flow` 不使用 east/west，避免
和全局位置混淆。对于默认的对称模板，East fabric 的 `forward` 是全局向东，West
fabric 的 `forward` 是全局向西。跨 VXM 的全局方向由
`source.fabric -> destination.fabric` 唯一确定。

CMake 在 configure 阶段校验名称、引用、数量、flow 和 latency，并将 path 展开为
`ftlpu::hw::config::kStreamTopology` constexpr descriptor。

## 字段含义

### 基本信息和拓扑

| 字段 | 含义 | 默认值 |
| --- | --- | ---: |
| `schema_version` | JSON schema 版本，目前必须为 2 | 2 |
| `target.name` | target 名称，进入 target ABI、MLIR 和诊断信息 | `ftlpu-lpu32` |
| `topology.hemispheres` | 芯片的 hemisphere 数量 | 2 |
| `topology.tiles_per_slice` | 一个数据 slice 覆盖的 tile 数 | 4 |
| `topology.lanes_per_tile` | 每个 tile 的 lane 数 | 8 |

`target.name` 只能包含字母、数字、点、下划线和连字符。创建不同物理配置时应使用不同名称。

一个 row 的 lane 数为：

```text
row_lanes = tiles_per_slice × lanes_per_tile
```

默认配置中为 `4 × 8 = 32 lanes`。tile 和 lane 数还会影响 MEM row 宽度、MXM
矩阵宽度、stream 数据宽度及每周期吞吐量。

### MEM

| 字段 | 含义 | 默认值 |
| --- | --- | ---: |
| `mem.slices_per_hemisphere` | 每个 hemisphere 的 MEM slice 数 | 52 |
| `mem.banks_per_slice` | 每个 MEM slice 的 bank 数 | 2 |
| `mem.rows_per_bank` | 单个 bank 的 row depth | 8192 |
| `mem.bytes_per_lane` | 一个 row 中每条 lane 的字节数 | 1 |

容量按照以下关系派生：

```text
row_bytes           = tiles_per_slice × lanes_per_tile × bytes_per_lane
slice_depth_rows     = banks_per_slice × rows_per_bank
slice_capacity_bytes = row_bytes × slice_depth_rows
chip_capacity_bytes  = hemispheres × slices_per_hemisphere × slice_capacity_bytes
```

默认配置中：

```text
row_bytes        = 4 × 8 × 1 = 32 bytes
slice_depth      = 2 × 8192 = 16384 rows
单 slice 容量   = 32 × 16384 = 512 KiB
全芯片 MEM 容量 = 2 × 52 × 512 KiB = 52 MiB
```

`rows_per_bank` 是单个 bank 的地址深度，不是整个 slice 在两个 bank 之间平分后的
总深度。修改该值后，CModel 和 Software 都需要重新配置并构建。

### SR

| 字段 | 含义 | 默认值 |
| --- | --- | ---: |
| `sr.registers_per_lane` | 每条 lane 可编码的 stream/register 数 | 64 |
| `sr.bytes_per_stream_per_lane` | 每条 stream 在每条 lane 上的字节宽度 | 1 |

当前 6-bit stream ISA 要求 `registers_per_lane = 64`，派生为 eastward 和 westward
各 32 条 stream。当前 byte-stream 数据通路要求两个 byte-width 字段都为 1。

### MXM

| 字段 | 含义 | 默认值 |
| --- | --- | ---: |
| `mxm.accum_contexts` | 每个 MXM 的完整 MXM tile FP32 accumulator context 数 | 4 |
| `mxm.supported_modes` | target 支持的 MXM decode layout | `native4x4`、`linear1x16` |

`Block8` 已删除，不能再写入 `supported_modes`。模式列表不能为空，不能重复，也不能包含
未知模式。Software 直接读取 JSON 时还会检查模式集合是否与 Software 构建配置一致。

### VXM

`vxm.alus` 表示每个 lane 的物理 ALU 总数。当前 develop 链式结构固定为 16，
对应两条镜像八级数据通路和 8 条共享 ICU queue；Ci 与 C(i+8) 共用配置。
此值不等于 ICU queue 数量。CMake 配置与 C++ 编译均拒绝非 16 配置。
2/4/8 是硬件 chain depth，不能通过更改此字段改变。原可变物理列数的显式 SR
VALU 结构已备份至 `valu` 分支。

## 创建和修改配置

建议复制默认文件，不要用临时实验覆盖默认配置：

```powershell
Copy-Item `
  .\config\ftlpu-lpu32.json `
  .\config\ftlpu-experiment.json
```

推荐流程：

1. 修改 `target.name`，为新配置提供唯一名称。
2. 修改需要实验的物理字段。
3. 使用新 JSON 重新配置并构建 CModel。
4. 使用同一 JSON 重新配置并构建 FTLPU-SOFTWARE。
5. 编译模型时继续显式传入同一 JSON，避免误用默认 target。

JSON 已加入 CMake configure dependency，修改后执行 `cmake --build` 通常会自动重新配置。
为了确保两个工程的 CMake cache 指向同一个文件，修改物理参数后仍建议显式重新执行两边的
configure 命令。不需要删除整个 build 目录。

## CModel 读取指定 JSON

在 FTLPU-CMODEL 目录执行：

```powershell
cmake -S . -B build-custom `
  -DFTLPU_HARDWARE_CONFIG="E:/configs/ftlpu-experiment.json"

cmake --build build-custom --config Debug
```

CMake 会校验 JSON，并在 build 目录的 `generated/include` 下生成
`ftlpu/core/hardware_config.hpp`。这是构建产物，不应手工编辑。

## FTLPU-SOFTWARE 读取同一 JSON

在 FTLPU-SOFTWARE 目录执行：

```powershell
cmake -S . -B build-custom `
  -DFTLPU_CMODEL_DIR="E:/path/to/FTLPU-CMODEL" `
  -DFTLPU_HARDWARE_CONFIG="E:/configs/ftlpu-experiment.json"

cmake --build build-custom --config Release
```

`FTLPU_CMODEL_DIR` 指向配套的 CModel 源码目录。两个工程的
`FTLPU_HARDWARE_CONFIG` 必须指向同一文件，建议使用绝对路径。

## `ftlpu_opt` 读取指定 JSON

```powershell
.\build-custom\compiler\ftlpu_opt.exe `
  --input .\model.stablehlo.mlir `
  --output .\model.kernel.mlir `
  --pipeline ftlpu-stablehlo-to-kernel `
  --target-config "E:/configs/ftlpu-experiment.json"
```

省略 `--target-config` 时，工具使用构建 Software 时生成的默认 target。自动化脚本中建议
始终显式传入该参数。

需要区分两种读取方式：

- CMake 读取用于生成固定数组尺寸、指令相关常量和 runtime ABI，修改后必须重新构建。
- `ftlpu_opt --target-config` 用于选择和验证编译 target，不能把已构建的 Software 动态变成
  任意硬件；若固定结构或支持模式与构建配置不一致，工具会拒绝该 JSON。

## 校验规则和注意事项

- `schema_version` 必须为 2。
- 所有数量和尺寸字段必须是正整数。
- `target.name` 不能为空，只能包含字母、数字、`.`、`_` 和 `-`。
- `sr.registers_per_lane` 必须为 64，以匹配当前 6-bit stream ISA。
- `mem.bytes_per_lane` 和 `sr.bytes_per_stream_per_lane` 必须为 1。
- `mem.rows_per_bank` 必须是 64 到 32768 之间的 2 的幂。
- `mxm.accum_contexts` 不能超过 256。
- `mxm.supported_modes` 不能为空、不能重复，只能包含已实现模式。
- 修改 tile、lane、bank、row 或 accumulator 数量后必须重新构建两个工程。
- CModel 和 Software 使用不同 JSON 时，可能造成地址、容量或 target ABI 不匹配。
- 不要修改 generated header；应修改 JSON 后重新运行 CMake。

配置不满足约束时，CMake 会在 configure 阶段报告具体字段错误；Software 命令行读取 JSON
时也会执行相同的核心检查。

## 检查配置是否生效

CMake configure 会输出 target 名称和 JSON 路径。也可以运行一次 Kernel lowering，在输出
MLIR 顶部检查 `ftlpu.target`，例如：

```text
name = "ftlpu-lpu32"
hemispheres = 2
slices_per_hemisphere = 52
banks_per_slice = 2
sram_depth_rows = 8192
mxm_accumulator_blocks = 4
vxm_alus = 16
```

其中 `sram_depth_rows` 是单 bank depth；单 slice 总 depth 还需要乘以
`banks_per_slice`。

默认 4-tile 配置和 8-tile 实验配置均使用每个 MXM 4 个 ACC context。
一个 context 保存一个完整的 FP32 MXM tile：4-tile 时为 32×32，
每个 MXM 共 16 KiB；8-tile 时为 64×64，共 64 KiB。
编译器按配置分配 ACC 行地址，已生成的旧程序不能直接复用于缩小容量后的硬件。

## DMA 程序装载模型（首版）

共享 JSON 另外接受以下仿真参数。旧配置缺省时使用表中默认值；CMake 生成
`hw::config` 常量，`C2cDmaSystem`、程序 DMA 和 `ProgramLoadConfig` 使用这些值初始化。
修改 JSON 后重新 configure/build 即可改变模型时序；这些参数不进入指令 ABI。

| 字段 | 默认值 |
| --- | ---: |
| `external_memory.dma_fifo_depth_vectors` | 16 |
| `program_load.bus_bytes_per_cycle` | 32 |
| `program_load.route_latency_cycles` | 1 |
| `program_load.imem_write_ii_cycles` | 1 |
| `program_load.imem_write_commit_latency_cycles` | 1 |

五项均为 `[1, 2147483647]` 范围内的整数。带宽以 bytes/cycle 计，其余三项时序以物理周期计。
也可以在构造 C2cDmaSystem 时传入 FIFO 深度及第四个参数 ProgramLoadConfig，或在没有
活动装载时调用 `program_loader().configure(config)`。DDR 仍是系统原有的同一个实例，
程序读取与权重/数据读取共同受其请求容量和带宽限制。

32 bytes/cycle 和两个 1-cycle 延迟是首版建模假设，不表示已经确认的 RTL 位宽和延迟。
装载器固定单活动程序、单活动目标，并按 runtime 提交清单的顺序处理；上一个目标最后
一次写入提交后才允许下一个目标开始 DDR 读取。

硬件接口位于 `include/ftlpu/system/program_loader.hpp`：

```cpp
auto& loader = system.program_loader();
loader.submit(images, system.cycle()); // images 是已经解析的 ProgramImage 清单
loader.start();                       // 可以提前请求；请求会保留
loader.set_dependencies_ready(inputs_and_initial_weights_ready);
system.tick();                       // 每调用一次，物理时钟和共享 DDR 前进一拍
```

首版物理格式仅接受 MEM 96-bit、MXM load/compute/dequant 128-bit Macro 映像。
长度包含物理控制 word、物理 word 尾部填充，不包含 DMA 32-byte vector 的尾部填充。
每个 word 内 lane 0 对应最低 32 bit，每个 lane 按 little-endian 字节顺序传输。
不使用语义指令数量估算传输长度；未实现的 Native/VXM/SXM 格式明确报错。

接收 FIFO 与 outstanding 共用同一 DMA credit 上限；64-byte 重组缓存、每拍最多接收
一个 packet 的弹性传输流水线、提交延迟决定容量的写流水线都是有限资源。完整 word
才会请求写入，每拍最多接受一个；内容在接受周期加 commit latency 时才可见。
传输流水线末端受阻时逐级反压，最终停止发出新的 DDR 请求。

`all_images_committed()` 后，各 Macro 前端每个物理周期并行预取一拍。
`ready_to_start()` 表示程序映像和本地前端已经就绪；统一启动还要求外部依赖门控为真。
`statistics().program_start` 记录统一启动且本次 kernel 逻辑周期 0 发射的物理周期。
程序切换仅重置计算队列，不复位 DDR、权重 DMA、传输队列或物理时钟。

`statistics()` 提供逐队列 source_done、最后提交、首笔请求/返回，以及全程序提交、
预取完成、启动时间戳、有效映像/DMA 字节数、word 数、DMA credit/总线/写端等待周期。
`statistics().print(os)` 输出这些数据，装载和预取时间直接由时间戳相减得到。
总线等待统计部分 vector 由于带宽限制仍需后续拍传输的周期；写端等待统计已有完整
word 但写端尚不能接受的周期；DMA credit 等待统计仍有请求且 FIFO 加 outstanding
已满的周期。这些等待可以重叠，不能相加作为总装载时间。

验收测试：`subsystem.program.loader` 和 `subsystem.program.configuration`。
