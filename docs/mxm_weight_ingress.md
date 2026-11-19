# MXM 权重入口反量化

适用于新版 `lpu_mxm_core`。压缩权重在每行入口恢复成 BF16，之后进入原有的权重移位链和 Supercell 两套权重 Buffer。

| 权重编码 | scale 模式 | 恢复公式 |
|---|---|---|
| INT8 | BF16、FP32 | `(q − zero) × scale` |
| INT8 | UE8M0 | `(q − zero) × 2^(scale − 127)` |
| FP4 E2M1、FP8 E4M3FN | BF16、FP32 | `decode(q) × scale` |
| FP4 E2M1、FP8 E4M3FN | UE8M0 | `decode(q) × 2^(scale − 127)` |
| FP4 E2M1 | NVFP4 | `decode(q) × block_scale × global_scale` |

INT8 可选择隐式零或逐 lane 有符号 INT8 zero。FP4 和 FP8 使用隐式零。

UE8M0 的 0xff、E4M3FN 的 NaN 以及无效乘积输出规范 BF16 NaN 0x7fc0。有限输入按完整尾数乘积最终舍入一次，使用 RNE，保留 BF16 次正规数和带符号零，溢出产生带符号 Inf。INT8 的差值用 9 位有符号数表示。

## 指令和字节布局

参数加载为 5 位，反量化执行为 3 位，各有独立队列。两者 bit 0 均选择参数 Buffer，bits 2:1 分别选择 scale 模式和权重编码。参数加载的 bits 4:3 选择 zero 模式，`00` 为隐式零，`01` 为显式 INT8 zero，`10/11` 保留。完整位域见 [MXM 指令](mxm_instructions.md)。

| scale 模式 | 同拍读取的 scale 字节 | `zero_mode=01` 同拍额外读取 |
|---|---|---|
| 00 BF16 | SR1/2，低字节在前 | SR3 |
| 01 FP32 | SR1～4，低字节在前 | SR5 |
| 10 UE8M0 | SR1 | SR2 |
| 11 NVFP4 | SR1～4 为 FP32 全局 scale，SR5 为 E4M3FN 分组 scale | 编码非法 |

上述为 `lpu_mxm_config_pkg.sv` 中的默认 SR 绑定。SR0 为权重输入。INT8、FP8 每 lane 一个字节。FP4 使用 SR0 的前 `ceil(CELL_K/2)` 个物理 lane 字节，低半字节对应偶数权重 lane，高半字节对应紧邻的奇数权重 lane。奇数宽度的最后一个高半字节忽略。隐式零的加载不读取额外字节。`WEIGHT_ZERO_POINT_ENABLE=0` 时显式零编码报 fault。

每个 lane 的参数槽默认保存 32 位 scale、8 位分组 scale 和 8 位 zero。默认每行每套 Buffer 的参数数据为 384 bit，两套为 768 bit。关闭显式零支持后，每 lane 的参数数据为 40 bit。每个 Buffer 另存 2 位 scale 模式、1 位显式零标志和 1 位有效位。完整输入在同一沿被消费，并更新所选 Buffer。

编译器把模型分组 scale 排布到对应的 lane。默认 `k = row × CELL_K + lane`，沿 K 维 16 个权重共享的 NVFP4 分组 scale 覆盖同一输出列的两个 Superlane。参数可以提前装入另一套 Buffer，或在多次读取时复用。参数和权重的字节掩码独立，可同拍加载下一套参数并执行当前权重。

## 五级流水

| 阶段 | 处理 |
|---|---|
| D1 | 拆包、解码权重、锁存所选参数的符号/指数/尾数。INT8 计算差值，识别零和特殊值。 |
| D2 | NVFP4 做 2×4 位有效尾数乘法并保留 6 位结果。其余模式寄存原有效尾数。 |
| D3 | 共享 8×24 位主乘法器，保留完整 32 位乘积。UE8M0 旁路乘法，携带修正后的指数。 |
| D4 | 检测最高有效位，确定 BF16 指数与移位量。 |
| D5 | 规格化、RNE 和特殊值选择，接收端 BF16 权重寄存器直接采样。 |

在边沿 N 接收权重，N+4 写入。若底层权重与反量化指令在 W 发射，行 r 的第 j 拍权重于 W+3r+1+j 进入 D1，于 W+3r+5+j 写入左端。所有格式同延迟，输入就绪时可逐拍切换，II=1。目标频率需要综合和时序分析确认。

D1 后下一沿即可重载原参数组。若同沿读写同一组，拒绝双方并置 sticky fault。缺失必要字节、参数未初始化、权重与参数模式组合非法，或使用保留编码也报 fault。run=0 清在途有效位和 fault，保留参数值及初始化状态。复位使两套参数失效。

## 快速回归

```text
python scripts/test_mxm_supercell.py --top lpu_mxm_float_weight_tb,lpu_mxm_weight_dequant_tb,lpu_mxm_dequant_control_tb,lpu_mxm_weight_shift_tb
```

独立有理数模型覆盖 13 种合法组合、UE8M0 全部编码、FP4/FP8 解码、FP32 scale 边界值、NVFP4 两级缩放及 INT8 回归。混合格式用例核对消费掩码、D5 固定时序、逐拍切换和 BF16 权重经阵列后的点积。
