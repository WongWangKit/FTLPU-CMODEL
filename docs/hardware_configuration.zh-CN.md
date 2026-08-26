# JSON 硬件配置指南

FTLPU-CMODEL 与 FTLPU-SOFTWARE 使用同一份 JSON 描述物理硬件 target。两个工程在
CMake 配置阶段读取它并生成编译期常量，Software 编译器也可以通过命令行读取同一文件。

## 默认配置和 JSON 格式

默认配置为 [`config/ftlpu-lpu32.json`](../config/ftlpu-lpu32.json)。未指定
`FTLPU_HARDWARE_CONFIG` 时，CModel 使用该文件；Software 使用
`${FTLPU_CMODEL_DIR}/config/ftlpu-lpu32.json`。默认内容如下：

```json
{
  "schema_version": 1,
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
    "accum_contexts": 32,
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

## 字段含义

### 基本信息和拓扑

| 字段 | 含义 | 默认值 |
| --- | --- | ---: |
| `schema_version` | JSON schema 版本，目前必须为 1 | 1 |
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
| `mxm.accum_contexts` | 每个 MXM 的完整 32 x 32 FP32 accumulator context 数 | 32 |
| `mxm.supported_modes` | target 支持的 MXM decode layout | `native4x4`、`linear1x16` |

`Block8` 已删除，不能再写入 `supported_modes`。模式列表不能为空，不能重复，也不能包含
未知模式。Software 直接读取 JSON 时还会检查模式集合是否与 Software 构建配置一致。

### VXM

`vxm.alus` 表示每个镜像方向上的 VXM ALU 数量。默认每个方向为 8，两个方向共
`2 × 8 = 16` 个物理 ALU，因此 Software 的 `ftlpu.target` 中显示 `vxm_alus = 16`。

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

- `schema_version` 必须为 1。
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
mxm_accumulator_blocks = 32
vxm_alus = 16
```

其中 `sram_depth_rows` 是单 bank depth；单 slice 总 depth 还需要乘以
`banks_per_slice`。
