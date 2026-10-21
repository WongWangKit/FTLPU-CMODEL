# Excel 三档硬件配置

来源：工作区 `LPU参数配置.xlsx`，`三档参数对照` 工作表 C、D、E 列的三档参数。
这些是实验配置，不代表 RTL、FPGA 或整层模型已经验收。原来的
`config/ftlpu-lpu32.json` 保持原样，历史测试仍可使用它。

| 参数 | LPU64 cloud | LPU32 edge | LPU16 decode |
|---|---:|---:|---:|
| tiles / lanes | 8 / 64 | 4 / 32 | 2 / 16 |
| 每侧物理 MXM | 2 | 1 | 1 |
| MXM 阵列 | 64×64 | 32×32 | 16×16 |
| 每 MXM ACC contexts / KiB | 4 / 64 | 4 / 16 | 4 / 4 |
| VXM ALUs | 32 | 16 | 16 |
| 每侧 slices / 每组 slices | 52 / 4 | 52 / 4 | 52 / 4 |
| banks / rows | 2 / 2048 | 2 / 2048 | 2 / 1024 |
| 总 MEM MiB | 26 | 13 | 3.25 |
| 时钟 MHz | 1000 | 500 | 200 |
| SR 编码 stream 数 | 64 | 64 | 64 |
| 每侧 RX 通道 / beat bytes | 8 / 32 | 8 / 32 | 8 / 16 |
| 单 LPU 外存预算 GB/s | 1200 | 68.264 | 25.6 |
| RX 上限 GB/s | 512 | 256 | 51.2 |
| 效率 | 80% | 80% | 80% |
| 估算有效上限 GB/s | 409.6 | 54.6112 | 20.48 |
| 读 / 写基础延时 cycles | 100 / 100 | 50 / 50 | 30 / 30 |
| outstanding 上限 | 512 | 128 | 64 |

## 文件与字段语义

- `config/ftlpu-lpu64-cloud.json`
- `config/ftlpu-lpu32-edge.json`
- `config/ftlpu-lpu16-decode.json`：使用表中 FPGA 档参数试验 decode，未另行提高频率或放大 SRAM。

新增 `mxm.instances_per_hemisphere` 控制 CModel 实际 MXM 数组和 ICU 数量；
省略时保留原来的每侧两个物理实例。`throughput.mxms_per_hemisphere`
仍表示本程序启用数量，不得超过物理实例数。

外存 `ddr_peak_bandwidth_mbytes_per_second` 使用
`min(单 LPU 带宽预算, RX 入口)`，再通过效率字段施加表中的 80%，
从而与表中第 56 行一致。存储器原始峰值、容量、类型均保留在 `planning.workbook_rows`。
这些规划字段不自动实现内存容量限制或控制器协议。
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
