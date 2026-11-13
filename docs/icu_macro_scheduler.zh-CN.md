# ICU Macro ISA（当前实现）

更新：2026-11-13。本文描述当前编译器与 CModel 的正式执行路径。

## 指令与队列

| ICU 队列 | 当前编码 | 物理 iMEM word | LPU32 Macro context |
| --- | --- | --- | --- |
| MEM，每 slice / bank 独立 | packed Macro + 同步/传输控制 | 96 bit | 256 bit |
| MXM Load / Compute / Dequant，彼此独立 | packed Macro | 128 bit | 256 bit |
| SXM Transpose | packed Macro | 96 bit | 256 bit |
| SXM Permute，与 Transpose 独立 | packed Macro | 96 bit | 384 bit |
| VXM C0…C7 | 顺序 CONFIG，一字一条 | 96 bit | 不使用 Macro context |
| C2C 与 common barrier | 原有传输/同步控制 | 96 bit | 不使用计算 Macro |

已移除独立的 Repeat2D、MEM/MXM 2D/3D 短包及长包、MEM WRITE_READ_2D、
STREAM_ND、MEM slice program、SXM tile program 和 SXM/VXM RUN_2D/RUN_3D。
旧产物需要重新编译；没有运行时将这些旧包转换为 Macro 的兼容执行器。
底层测试使用的单次 FU / Repeat1D helper 不代表编译器计算队列的另一种部署格式。
MEM_READ_SYNC、MEM_WRITE_SYNC、SYNC 等传输/同步语义保留。

## 编译与压缩

C++ Schedule 中的多维仿射域是编译器的数学表示，不是硬件 2D/3D ISA。
`MacroCommandMaterializer` 直接从这些域产生 `ftlpu.command.macro`：合并可展平轴，
按时间顺序划分剩余轴，解析 blocked 地址、weight-buffer 切换和末次 accumulator 行为。
无需先展开全部 FU 指令再反向压缩。MEM 读写对成为显式读 Macro 和写 Macro。

Macro 包含 FU 模板、起始周期、inner/outer 次数与周期步长，以及适用 FU 的操作数归纳。
这些是 Macro 自身的压缩字段，不是重新引入的 RUN_2D 或 READ_3D 指令。
物理编码使用 compact/extended 描述符、delta 字典及 run；最终重定位后再打包为 iMEM bitstream。
容器中的 semantic descriptor 数、物理 word 数和实际 FU issue 数分别统计，不能混为指令大小。

默认 `--icu-compression raw-macro`，`macro` 为同义参数。
`none`、`control` 和旧 `--mem-slice-program` 模式不再支持。
VXM 保持顺序 CONFIG，不再套用通用循环包。

## SXM Macro

两个独立 decoder 分别只产生 Transpose 或 Permute FU 指令。
LPU32 Transpose FU 模板为 16 bit（源/目标 stream base 与输入 row selector）；
Permute 模板为 179 bit（另含输出 selector 与完整 32-lane map）。
保留既有 wide transpose / local transpose / broadcast 的 FU 语义和合法 stream 范围。
Macro 不隐式旋转 permute map：map 变化由编译器生成不同模板，必须计入取指与解析开销。

SXM 使用与 MEM/MXM 相同的有界前端：固定 3-word reservoir、64-bit decode window、
8-entry DDB，按 target 中既定 fetch latency / request II 取指。
LPU32 SXM iMEM 每队列 2048 × 96 bit，context slot 数仍为 4；没有扩大容量或提高取指带宽。
更宽 target 的 Permute context 按既有几何上限检查。

## 排程与同步约束

同一 ICU 队列只执行一条 Macro。后一条的首次 FU issue 必须晚于前一条末次 FU issue；
不能利用前一条内部空闲周期交错执行另一条 Macro。编译、打包和 CModel 都检查此约束。
前端可预取/预译码后续描述符；多个已缓存 context 不等于多个交错执行的 Macro。
超过物理窗口、带宽、iMEM 或 context 位宽时明确报错，不自动放宽硬件。

DDR/C2C 仍经 common barrier 与 MEM 同步传输控制协同；Macro 计算队列在 barrier 处同步。
执行导出的 ICU CSV 包括同步控制以及真正装载的 packed image；runtime pipeline 由实际 FU issue 产生。

## 验证与复现

关键回归覆盖 packed bitstream 往返与真实译码、FU 逐周期等价、SXM stream/map 字段、
重定位、同步分页、iMEM 容量、队列禁止交错、wide transpose 数值和原生 RoPE 数值。
`direct_raw_macro_cli_test` 覆盖默认/显式 Macro、Command/StableHLO 入口与小 FFN 数值。

```powershell
ctest --test-dir build-macro-only --output-on-failure
```

CModel 单独构建后执行 `ctest --test-dir build-macro-only --output-on-failure`。
模型回归仍需同时检查完整输出、时延、runtime pipeline 和实际 ICU program。
本次 DeepSeek SWA 回归仅覆盖 seq32 的 attention 前端与 SWA core，不包含 mHC、输出投影或 FFN。
