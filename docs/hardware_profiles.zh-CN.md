# Excel 三档硬件配置

来源：工作区 `LPU参数配置.xlsx`，`三档参数对照` 工作表 C、D、E 列的三档参数。
这些是实验配置，不代表 RTL、FPGA 或整层模型已经验收。原来的
`config/ftlpu-lpu32.json` 保持原样，历史测试仍可使用它。

| 参数 | LPU-MAX | LPU32 edge | LPU16 FPGA |
|---|---:|---:|---:|
| tiles × lanes/tile | 20 × 16 | 4 × 8 | 2 × 8 |
| 每侧物理 MXM | 2 | 1 | 1 |
| MXM 阵列 | 320×320 | 32×32 | 16×16 |
| 每 MXM ACC contexts / KiB | 4 / 1600 | 4 / 16 | 4 / 4 |
| VXM ALUs | 16 | 16 | 16 |
| 每侧 slices / 每组 slices | 44 / 4 | 24 / 4 | 20 / 4 |
| banks / rows | 2 / 2048 | 2 / 2048 | 2 / 1024 |
| 总 MEM MiB | 110 | 6 | 1.25 |
| 时钟 MHz | 1000 | 500 | 200 |
| SR 编码 stream 数 | 64 | 64 | 64 |
| 每侧 RX 通道 / beat bytes | 8 / 320 | 8 / 32 | 8 / 16 |
| 单 LPU 外存预算 GB/s | 1200 | 68.264 | 25.6 |
| RX 上限 GB/s | 5120 | 256 | 51.2 |
| 效率 | 100% | 80% | 80% |
| 估算有效上限 GB/s | 1200 | 54.6112 | 20.48 |
| 读 / 写基础延时 cycles | 100 / 100 | 50 / 50 | 30 / 30 |
| outstanding 上限 | 512 | 128 | 64 |

## 文件与字段语义

- `config/ftlpu-max.json`
- `config/ftlpu-lpu32-edge.json`
- `config/ftlpu-lpu16-decode.json`：使用表中 FPGA 档参数试验 decode，未另行提高频率或放大 SRAM。

新增 `mxm.instances_per_hemisphere` 控制 CModel 实际 MXM 数组和 ICU 数量；
省略时保留原来的每侧两个物理实例。`throughput.mxms_per_hemisphere`
仍表示本程序启用数量，不得超过物理实例数。

三档由同一份工作簿生成。MEM slice 的产品设计上限为每侧 44，历史基线仍可使用 52。
每侧传输通道由 `c2c.streams_per_direction` 指定，与 tile 宽度独立；省略该字段的旧目标沿用原先的 lane 数。
MAX 的四个 MXM 合计使用 6.25 MiB ACC。320 B 向量 MEM 回读和最后一个 ACC 分段的累加均有独立回归。
集成 SXM 的旧 FU 包最多携带 16 个 stream 选择，部分行选择使用 4 位并保留广播和释放编码。
16 lanes/tile 的整机运行需要另行确定 SXM 指令格式和 matrix adapter。

外存 `external_memory` 分三层记录带宽，避免把器件峰值当成可用带宽：

| 字段 | 含义 | 来源 |
|---|---|---|
| `ddr_interface_peak_bytes_per_second` | 器件接口峰值 | 表中第 53 行 |
| `ddr_budget_bytes_per_second` | 分配给单 LPU 的预算 | 表中第 54 行 |
| `ddr_rx_entrance_bytes_per_second` | 片上 RX 理论入口 | 表中第 47 行 |
| `ddr_effective_peak_bytes_per_second` | `min(预算, RX 入口) × 效率` | 表中第 56 行 |

CMake 生成 `hw::kDdrPeakBytesPerSecond` 时直接取 `ddr_effective_peak_bytes_per_second`，
即模型按"可持续上限"限流，而不是按器件引脚峰值。`ddr_read/write_latency_cycles`
与 `ddr_request_queue_depth` 一并进入 `Ddr4Config`，所以实际可达带宽还受 outstanding
和延时约束，不再由效率系数单独决定。省略 `external_memory` 的旧 target 回落到
500 MHz 双通道 DDR4-3200 平台（51.2 GB/s、35/25 cycles、队列 256、抖动 15/10）。

存储器容量与类型仍只保留在 `planning.workbook_rows`，不参与容量校验或控制器协议。
表中没有给延时抖动，三档设为零；基础延时保留 ns 到周期的换算。

布局种子使用 `attention_mask_base_row = rows_per_bank - 256`、
`matmul_result_base_row = min(1600, rows_per_bank/2)`，避免直接沿用越界地址。
这不是完整的张量生命周期分配，也不保证整个模型能放下。

ICU 仍为每个 slice bank 一个队列、每队列一个活动指令、IQ 深度 16；
这些值目前与表一致但仍是实现常量。FPGA 的 iMEM 分段装载尚未实现。

## 生成与构建

生成脚本只读取 Excel，依赖 `openpyxl`，不修改工作簿。公式必须具有缓存值。

```powershell
python tools/generate_workbook_profiles.py ../LPU参数配置.xlsx
cmake -S . -B build-max -DFTLPU_HARDWARE_CONFIG="$PWD/config/ftlpu-max.json"
cmake --build build-max --config Release --target vector_roundtrip_test mxm_accumulator_test
ctest --test-dir build-max -C Release -R '^(unit.mem.vector_roundtrip|unit.mxm.accumulator)$' --output-on-failure
cmake -S . -B build-lpu16-decode -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DFTLPU_HARDWARE_CONFIG="$PWD/config/ftlpu-lpu16-decode.json"
cmake --build build-lpu16-decode --target hardware_profile_test geometry_encoding_test wide_transport_codec_test vector_roundtrip_test mxm_decode_streamed_weight_test --parallel 2
ctest --test-dir build-lpu16-decode -R '^(integration.hardware_profile|unit.core.geometry_encoding|unit.core.wide_transport_codec|unit.mem.vector_roundtrip|subsystem.mxm.decode_streamed_weight)$' --output-on-failure
```

Windows 使用 VS 开发者命令环境，或软件仓库的 `tools/build_vs2026.cmd` 包装 CMake。
软件仓库也必须独立构建并传入同一个 JSON，不能用 LPU32 runtime 执行 LPU16 二进制。

```powershell
# 在 FTLPU-SOFTWARE 中；MLIR 路径按本机已有构建填写。
tools/build_vs2026.cmd -S . -B build-lpu16-decode -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DFTLPU_MLIR_DIR=E:/timesintelli/workspace2/FTLPU-SOFTWARE/build-mlir-vs2026/lib/cmake/mlir -DFTLPU_HARDWARE_CONFIG=E:/timesintelli/workspace2/FTLPU-CMODEL/config/ftlpu-lpu16-decode.json
tools/build_vs2026.cmd --build build-lpu16-decode --target ftlpu_opt ftlpu_compile compiled_qwen_real_decoder_layer_runtime_test ftlpu_icu_program_export --parallel 2
python tools/run_model_tests.py run --model qwen2_5_1_5b --suite decode --build-dir build-lpu16-decode --target-config ../FTLPU-CMODEL/config/ftlpu-lpu16-decode.json --output-root results/lpu16_decode_validation
```

## 首轮发现

- 修复 SXM Permute context 最小位数检查写死为 32-lane 的问题；现在随 lane map 和 selector 宽度派生。
- 修复传输测试写死 32-byte beat 的问题，按目标 beat 检查字节计数和完成时间。
- LPU16 上述五项底层测试通过；实际物理 MXM 总数 2，MEM 3,407,872 bytes，每 MXM ACC 4,096 bytes。decode 小测试包含 K=N=16 的 INT8 解量化、BF16 MAC 与 MEM 写回，输出逐元素与参考 BF16 位模式一致。
- 专用 LPU16 编译器、runtime、ICU exporter 构建成功，Qwen2.5 decode 的 StableHLO→Kernel IR 通过；统一入口执行到 Kernel→Tensor 时在 RMSNorm gamma 分配失败：当前每个 gamma 元素占一个广播行，1536 行无法置入 1024 行 bank。需要 gamma 分段/紧凑布局以及相应地址生成，不能仅移动起始地址。尚无整层 ICU 二进制和整层数值结果。
- 代码审查还发现 native decode FFN 的权重布局与循环仍使用 `/32`、`/64` 和 32 个权重 slices，且没有启用其普通路径的细粒度权重分页。这些不是通过底层测试即可认定已适配的内容。
- 统一测试入口读取目标 JSON 改为显式 UTF-8，支持配置里的中文 Excel 来源记录。验证日志见软件仓库 `results/lpu16_decode_validation/`。
