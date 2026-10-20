# LPU32 prefill 单权重 stream 实验

`prefill_weight_streams_test` 对比一个物理 MXM 的 8-stream supercell IW
与 1-stream column IW。计算为 `[M,128] × [128,64]`，M=16/32/64，
INT8 权重、BF16 激活和反量化权重、FP32 累加。每种形状处理 8 个权重块，
在两块 weight buffer 间交替，覆盖同一 buffer 被多次覆写的情况。

输入由测试初始化到物理 MEM；此后读写均通过 MEM ICU、SR 和 MXM ICU，
结果通过 AccumulatorRead 写回 MEM 再做数值比较。测试不模拟 DDR/C2C，
也不等同于完整 decoder 层的编译器验收。单 stream 使用 LOAD_COLUMN_3D 原始 ICU 指令，8-stream 使用 FU 指令
兼容队列。没有新增并行 ICU context，没有将 stream 总数从 64 裁减。

权重按列分布到 8 个 MEM slices。单 stream 保留反向列组存储布局，
在 8 个 slices 间逐周期轮换读出，同一个 slice 每 8 周期读一次，
送入同一条 stream。MXM 按正向列 0..31 装载。每列包含 32 字节，随 4 个
tile 的指令传播分段搬运；不能把逻辑 32 B/vector 理解为某一拍全向量原子到达。
两个方案均实际搬运 8192 字节权重。

## 装载与计算

单块权重为 1024 B，8-stream 装载发射持续 4 cycles，1-stream 持续 32 cycles。
一个权重块供 M 个 token 使用，计算每周期发射一行。
实验直接构造 `interval=max(M, load_cycles)` 的连续调度，无事后压缩。
装载顺序为列块、块内列；逐列覆写跟随旧权重的最后一次使用，而不等整个阵列排空。

默认配置实测：

| M | streams | 首次计算 | 计算发射 cycles | 内部空泡 cycles | 最大绝对误差 |
|---:|---:|---:|---:|---:|---:|
|16|8|24|128|0|0|
|16|1|52|128|112|0|
|32|8|24|256|0|0|
|32|1|52|256|0|0|
|64|8|24|512|0|0|
|64|1|52|512|0|0|

周期来自实际 ICU/FU 执行状态和 MEM 传输记录。内部空泡统计第一次到最后一次
tile0 Compute 发射之间的非计算周期，不包括启动、排空及最后的结果读回。
单 stream 的首次计算比基线晚 28 cycles；M=32/64 时稳态吞吐相同。
测试逐输出对比 CPU 参考并要求 1-stream 与 8-stream 结果完全一致。

## 复现

```powershell
cmake --build build-vs2026 --config Release --target prefill_weight_streams_test
ctest --test-dir build-vs2026 -C Release -R '^kernel.projection.prefill_weight_streams$' --output-on-failure
```

从 FTLPU-SOFTWARE 运行 `python tools/run_prefill_weight_streams.py` 可重复生成
`results/lpu32_prefill_weight_streams/` 的报告、6 份实际流水 CSV 及 viewer。
也可以给可执行文件传入输出目录；不传目录时只验证，不生成 trace。

## Qwen2.5 整层路径

编译器新增独立的调度策略 `throughput.mxm_prefill_weight_streams=1`。
使用 `config/ftlpu-lpu32-prefill-1stream.json`，从 FTLPU-SOFTWARE 运行：

```powershell
python tools/run_model_tests.py run --model qwen2_5_1_5b --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32-prefill-1stream.json
```

它运行完整尺寸合成权重的 seq32 单 decoder 层数值测试，覆盖 Q/K/V/O、
Gate/Up/Down 的 W8 权重单 stream 装载；attention QK/PV 的动态 BF16 数据装载
不属于这一开关。结果更新 `results/qwen2_5_decoder_layer_prefill/`，
通过与否以该目录本次 `runner.json` 和 `cmodel.runner.log` 为准。

编译器直接生成 MEM READ_3D 和 MXM LOAD_COLUMN_3D 域，不先展开 FU 再压缩。
LOAD_COLUMN_3D 使用现有双 weight buffer，内层循环是实际列号；ICU 每周期
向 MXM 发一条 IWColumn。MEM 维持 8-slice 存储/分页，改变读取顺序与 stream
选择。DDR 完成仍由运行时 page-ready SYNC 决定，不使用预测 DDR 延迟的 NOP。

配置目前限定 LPU32、本地反量化、每半球一个 MXM 的已适配 projection 路径。
`mxm_int8_load_streams_per_cycle=8` 仍表示 supercell 指令的硬件能力，不应改成 1。
此实验未裁剪物理 SR 数量，也未证明整层只需要一条 stream；激活、结果、
C2C、VXM/SXM 仍使用各自的 stream。

Gate→Up 的 epoch 边界另保留一个 SR 链长度的片上排空间隔（当前 16 cycles）。
单 stream 将权重装载提前并拉长为 32 cycles，若直接沿用原来的边界，
Up 的首次权重读取可能早于 Gate 最后几次激活读取，导致各 MEM ICU 在不同的
指令边界遇到 SYNC。这个固定间隔保证旧 MEM 域结束后再进入下一个 epoch；
它只发生在两个 projection 的边界，不插入每个权重块之间，也不估算 DDR 的完成时间。

2026-10-15 的 seq32 单层验收通过（500 MHz、51.2 GB/s，包含配置中的 DDR
延迟与抖动）：49152 个 BF16 输出，最大绝对误差 0.0625；与同条件 8-stream
数值对照的最大误差一致。动态分页使用 18 个 page uses、19 次预取，搬运
47194112 字节。实际运行流水覆盖 973327 cycles（约 1.947 ms，包含启动数据
读入与最终状态传出）；这个数不是整模型 TTFT。

输出在 FTLPU-SOFTWARE 的 `results/qwen2_5_decoder_layer_prefill/`：
`pipeline.html` / `runtime.pipeline.csv` 为实际执行流水，`icu_programs/index.csv`
索引运行时链接的各 ICU 程序，含 pre-execution 指令。数值验收使用完整尺寸
合成权重，不是官方 checkpoint。
