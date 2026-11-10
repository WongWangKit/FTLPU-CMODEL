# SXM/MEM transpose：cycle 23 buffer full 修复说明

更新：2026-11-10。CModel 基线 `76a0b2e5`，保留工作区已有的 MEM Main/R1 修改；
对照本地 VModel `9bd20a29` 的 `rtl/sxm_native/sxm_compat_engine.sv`。

## 结论

这次报错来自 **测试发令排程超出了 ICU 取指带宽**。SXM 的单源槽满检查正确，
不需要增加 buffer，也不需要修改 MEM、MXM、VXM 或 ICU 实现。

原 `tests/subsystem/sxm/mem_transpose_test.cpp` 每拍分别调用 enqueue T/P。
当前 API 将每次调用编码成一个 count=1 的 RUN_2D 描述符：LPU32 的 T 占 2 个
指令字、P 占 4 个指令字，队列每拍补 1 个字。初始预取掩盖了这个缺口。

原用例的 P 队列实测如下（cycle 按日志从 0 开始）：

| cycle | P 发出 | cycle 结束 IQ 字数 | 结果 |
|---|---|---|---|
| 18 | 是 | 12 | P 释放旧槽，T 可同拍补入 |
| 19 | 是 | 9 | 同上 |
| 20 | 是 | 6 | 同上 |
| 21 | 是 | 3 | 同上 |
| 22 | 是 | 0 | source tile 0 装入 ready_cycle=22 的块 |
| 23 | 否 | 1 | P 为 ThreeDDecodeWait；T 继续捕获，正确报 buffer full |

仅换成更深的数据池不能修复这个持续的指令带宽缺口。仅把数据通路换成
triangle/WriteTap 也不能让缺失的 P 指令出现。

## 本次改动

1. 测试的 SXM T/P 各使用一个现有 RUN_2D，保持每拍一次操作：
   - T：从 `kCaptureStart=17` 起，共 `matrix_count*4` 次；
   - P：从第 18 拍起，共 `matrix_count*4+3` 次，包含北向尾部；
   - P map 每次加 8 lane，等价于原逐拍 `wavefront_map(wave)`；
   - 绝对起拍由原有 NOP 表达，RUN_2D 的 start_cycle 保持 0。
   没有改 ICU 译码器、取指带宽、IQ 深度或指令语义。
2. SXM `idle()` 检查所有已捕获行，而非只检查完整块的 ready。
   这与 VModel compat 的 occupied-row busy 一致，避免半块被误报完成。
   修改前已独立复现该误报；它与 cycle 23 的根因分开处理。
3. 添加局部时序回归：部分行占用、禁止 T→P 同拍旁路、P 读旧值且 T 同拍复用、
   缺 P 仍报满、逐行输出直到 row 7 才释放。
4. 系统测试支持参数 `[matrix-count: 1..96]`，无参数保留原 96 矩阵规模。
   新增双矩阵 smoke，并检查每拍 T/P 发令、MEM Main 模式、所有结果和
   ICU/SXM/MEM/SR 的完整 drain。没有靠延长等待来掩盖丢失操作。

## 与当前 VModel 的对应范围

这条故障用例走 **16 byte-stream 的 BLOCK8→BLOCK8 兼容路径**，每个源 tile
只有一个有效槽；对应 VModel compat 引擎使用共享池的 slot 0。
T 按 root+tile 传播，P 读取拍前完整块，允许释放与重填同拍发生。该有效工作负载的
规则已对齐；本次只做 CModel 回归，未重新运行 RTL 对拍或 Vivado。

**没有将这个旧用例改成 triangle ROW→BLOCK 路径，也没有声称整个 SXM CModel
已与当前 VModel 全功能等价。** 以下范围仍然不同：

- VModel typed triangle 的 ROW 输入、每源 `s+1` 槽、II=8、BF16/FP32 及私有 job
  控制，是另一条模块级接口；本次旧兼容指令测试不覆盖它。
- VModel triangle→MEM 的多 bank 条带布局和原生 WriteTap 错拍排程，是稀疏 P
  输出的适配；本次旧测试仍使用原来的稠密普通 Write。因此无需改 MEM WriteTap。
- CModel 原有 fanout、ShiftSelect/Distribute 软件能力及部分非法输入处理仍保留，
  不据此宣称 VModel 支持这些功能。CModel 对缺失 T 数据的跳过行为也未改成
  VModel 的 capture fault。

## 验证与复现

LPU32、小规模 native CModel Debug 回归：14/14 PASS，另跑双向 1、3 矩阵均 PASS；
2 矩阵包含在 smoke 中。每种规模都校验完整矩阵、发令日历和 drain。
此前 A/B 已证明：SXM/MEM 实现完全不动，仅替换 SXM 发令排程，原双向
96 矩阵用例也通过。

```text
cmake --build <build> --target sxm_local_blocks_timing_test sxm_mem_transpose_test
ctest --test-dir <build> -R "unit.sxm.local_blocks_timing|subsystem.sxm.mem_transpose_smoke" --output-on-failure
<build>/sxm_mem_transpose_test 3
```

工作区完整证据：`learn/sxm_cmodel_alignment_20261110/README.md`（相对 LPU 根目录）。
本次未做 LPU16/64 系统回归、FP32 新路径验证、RTL 仿真、综合或 STA；未提交、推送。

后续 typed 矩阵路径和三档小回归见 [SXM 指令现状](sxm_instruction_status.md)。
独立克隆本仓库时，使用 [仓库内回归入口](sxm_mem_regression.md)，不依赖上述本地实验目录。
