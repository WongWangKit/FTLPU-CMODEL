# MEM、MXM、VXM 的 RTL 对齐模型

当前入口为 `ftlpu/cmodel.hpp` 中的 `ftlpu::CModel`，实现放在 `include/ftlpu/rtl/`。
对齐对象是 VModel 的 `mem_logical_bank_column`、`lpu_mxm_core`、
`lpu_vxm_tile_execution` 及其算术模块。数值路径使用整数位运算实现浮点舍入，
不使用宿主机 `float`、`double` 或 `std::log` 代替硬件计算。

## 配置与接口

RTL 对齐测试使用 `config/ftlpu-rtl-lpu32.json`，另提供 LPU16、LPU64 配置。
`Config::profile(16/32/64)` 可在原生模块接口上显式选择几何参数。
编译期配置通过 `-DFTLPU_HARDWARE_CONFIG=<绝对路径>` 指定；已有 CMake cache
不会自动更换配置，应使用新的 build 目录或显式设置该变量。

| 属性 | LPU16 | LPU32 | LPU64 |
| --- | --- | --- | --- |
| Superlane / Supercell 行列数 | 2 | 4 | 8 |
| 每个 Supercell 的输入数 / 输出数 | 8 / 8 | 8 / 8 | 8 / 8 |
| MEM 每 Bank 的行数 | 1024 | 2048 | 2048 |
| MXM 累加结果长度 | 16 | 32 | 64 |
| MXM 累加深度 | 1024 | 1024 | 1024 |

默认构造一个 MEM Bank column、一个 MXM Core，以及每个 Superlane 一个 VXM Tile。
可在构造函数中指定 MEM column 数和 MXM 单元数；这些实例独立，不隐含一套整芯片布线。
`model.mem(i)`、`model.mxm(i)`、`model.vxm(superlane)` 返回对应模块。
测试平台显式连接相邻模块的端口，并在每拍开始前保存所有模块的旧输出，避免 C++ 调用次序
让数据在同一拍穿过多个寄存器。示例见 `examples/rtl/schedule_demo.cpp`。

每次 `tick(input)` 表示一个上升沿：输入是沿前信号，返回该沿后的寄存器输出。
组合型 accept/consume 则描述本沿实际接收的输入。下文 E0 指首次采样的上升沿；
例如 9 级流水的首拍输出在 E8，不能把它误记为额外的 E9。

## 指令与 macro

基础指令不携带重复数量。`rtl::Macro<T>` 在规定的 `start/count/interval` 上展开原始指令；
地址步长由 macro 层生成，每条后端指令仍携带本次操作的地址。
模型不自动插入 NOP、等待数据或重排指令。

| 指令 | 位宽与内容 |
| --- | --- |
| MEM 当前原始指令 | 32 bit：preserve[31]、delay3[30]、row[29:15]、map[14:9]、west[8]、stream[7:3]、opcode[2:0] |
| MXM Compute | 7 bit：activation_stream[6:1]、weight_buffer[0] |
| MXM Weight Load | 2 bit：weight_buffer[1]、column_mode[0] |
| MXM Parameter Load | 5 bit：zero_mode[4:3]、scale_mode[2:1]、parameter_buffer[0] |
| MXM Dequant | 3 bit：weight_format[2:1]、parameter_buffer[0] |
| MXM Backend | 3 + clog2(depth) bit；默认 13 bit：BF16 输出[12]、address[11:2]、operation[1:0] |
| VXM | Global 15 bit；Q0..Q7 分别为 6/5/7/6/7/5/7/6 bit，镜像到另一半物理级 |

MEM raw map 位可输入，但按当前适配器约定忽略；生成的 canonical 指令将其清零。
旧 47 bit MEM packet、旧 MXM 计算/累加混合指令和旧 VXM packet 属于 legacy ABI。
新模型不把这些旧编码当作当前指令解析。

MXM 的 enqueue 与 launch 分离：空队列同拍 enqueue + launch 不能执行新入队指令。
`Output.enqueue_ready/launch_ready` 的顺序为 Compute、Weight Load、Parameter Load、Dequant、Backend。
满队列未接收的新指令需由调用方在后续继续提供，不产生隐藏缓存。
计算、权重、参数和反量化指令向上传播间隔为 3 拍，后端为 1 拍。
矩阵权重加载本地固定执行 MATRIX_N 拍，单列模式执行 1 拍；该固定结构计数不是基础指令 COUNT。
编译器按约定安排至少 32 批连续计算，并为换权重留足 MATRIX_N 拍。
连续原位累加的每轮安排偶数批，并遵守两个单口 Bank 的读写相位。需要填充时发射完整数据批。
详细规则见 [MXM 编译器时序约定](mxm_scheduling_contract.md)。

## MEM

对应硬件的编译器调度与输入编码见 [MEM 时序约定](mem_scheduling_contract.md)
和 [MEM 指令](mem_instructions.md)。可用 [MEM 数据波前](visualizations/mem-stream-edge.html)
与 [MEM/MXM 时序图](visualizations/mem-mxm-phase-3d.html)查看层间相位。

`MemBankColumn` 是一条逻辑单口 Bank 的完整纵向列：每个 Superlane 独立存储 8 byte/行，
每拍至多一个读或写请求。读取在本次采样沿后可见，写入接收本沿 Stream 数据。
Write 消耗数据；WriteTap 保存数据的继续传播资格，缺失数据时不写入。
valid 与命令一起向上传播，每条输入只执行一次；连续相同的 valid 指令仍是不同操作。

delay3=0 每层 1 拍，delay3=1 每层 3 拍，中间保留两个仅供长路径使用的寄存器。
从 3 拍切回 1 拍需满足编译器留空约定。若最后一条长指令在 T 发射，下一条短指令
最早在 `T + 1 + 2*(rows-1)` 发射，可让最上层两条指令连续执行。
非法挤压抛出 `logic_error`，不悄悄丢掉旧指令。复位清控制流水，保留 SRAM 数据。

## MXM

Activation 在一个 Supercell 内同时广播到各输出组，每经过一个 Supercell 横向寄存一拍。
两套 BF16 权重逐组移位，满尾部才继续传到右侧；计算期间不能写同一套权重。
同组各输出同时有效，纵向部分和通过独立的 3 拍 FP32 加法流水。

点积先对 BF16 乘积统一对阶，按当前 RTL 的有限窗口截去右移部分，再以有符号 64 bit
求和，最后 RNE 到 FP32；CSA 的算术效果与精确的同宽整数求和一致。
不会在每一对 BF16 乘加后舍入，也不会用 FP64 保留硬件本应舍去的远端尾数。
9 级点积、3 级部分和相加，连同 tag、特殊值和 `alignment_lost` 一起建模。
上方 Supercell 的本地点积必须与下方部分和同拍、同 tag 到达。

顶部结果通过独立的向下/向左寄存网络回传：以底行为 0，列 c 在行 c 转向，
经过相同的 rows 拍后从左侧输出，后端控制随返回数据以 1 拍间隔传播。
它与原有向上求和通路相互独立。

入口支持 INT8、FP4 E2M1、FP8 E4M3FN，参数可选 BF16、FP32、UE8M0 或 NVFP4 两级缩放。
显式有符号 INT8 zero 只与 INT8 权重配合，NVFP4 使用 FP4 和隐式零。
`ParamLoadInstruction` 和 `DequantInstruction` 提供原始位域的编解码，合法组合见 [MXM 指令](mxm_instructions.md)。
BF16 scale 搭配显式 zero 加载到参数 Buffer 0/1 时使用 `01000`/`01001`。

`MxmCore::Input::weight[row][sr][lane]` 为逐 lane 可选的 SR 字节。
`Config::weight_q_stream` 和 `Config::weight_parameter_streams` 的默认值为 SR0 和 SR1～5，
`Config::weight_zero_point_enable` 控制是否接受显式 zero。
参数按指令同拍消费所需字节，`Output::weight_consume[row][sr]` 的各位表示本沿消费的 lane。
准确字节排布和 FP4 半字节顺序见 [入口反量化约定](mxm_weight_ingress.md)。

D1 快照所选参数，使用整数尾数保留完整乘积并一次 RNE 到 BF16。
CModel 在 D1 计算位模式，经固定流水延迟到 D4 输出，Supercell 在 D5 采样。
硬件在 D1 接收后第四个上升沿写入，所有合法模式均可逐拍切换。
D1 后下一拍即可重载旧参数 Buffer，同拍读写同一组时双方拒绝并置 fault。

每个返回输出组绑定一个累加器，包含 8 个 FP32 加法器和两块 256 bit 单口 Bank。
逻辑地址 a 映射为 Bank=`a%2`、Bank 内 row=`a/2`；默认每块 512 行。
后端四种操作为 DirectOut、Store、AccStore、AccOut，无累加结果反馈。
新部分和与预读 B 在 A1 直接相遇，前面没有额外的宽 A 缓冲。
原语 launch 到读服务为 1 拍，到 A1 为 3 拍，到写回/输出为 6 拍；读到写相差 5 拍。
可选 FP32→BF16 与 FP32 直通输出具有相同延迟。
MXM 量化对 NaN 保留符号和可截取的 payload，并强制置 quiet bit；有限数包含 subnormal 的 RNE 舍入。
同 Bank 读写冲突时两侧请求均抑制；未初始化读取、未提交地址依赖、缺失/多余部分和都会 fault。
复位保留数据阵列但清除初始化有效位；run=0 仅清流水与故障，保留队列、SRAM 有效数据和参数 Buffer。

## VXM

RTL 数据通路和指令见 [VXM 架构](vmodel_architecture.md#vxm-module-boundaries)。
独立验证入口见 [VXM RTL 验证](vxm_rtl_verification.md)。

`VxmTile` 包含每 Lane 16 个物理算术级，以及寄存的 token、original/auxiliary、
反馈、累加状态和输出。配置装载与 execute 分离，支持 2/4/8 级链路及 first/last 元数据。
`VxmController` 对应 RTL resident configuration 的装载、重复执行和 final marker 退休过程；
其 repeat 是配置控制层的字段，不会重新加入 MXM 基础指令。

FP16/BF16 输入使用固定的两条 byte Stream，一拍组装；FP32 使用低半字、高半字两拍，
随后一拍才参与执行。LHS/RHS 分别维护组装状态。输出也保留 FP32 两拍及背压状态。
格式转换中的 DAZ、FTZ、NaN 规范化和无需转换时的原始位模式保留均按 RTL 区分。
VXM 跨类型转换对 NaN 保留符号并规范化 payload；它与 MXM 输出量化的规则不同。

基础操作的外部延迟为 1 拍，Multiply 为 2 拍，特殊函数为 8 拍且单事务在途。
C1 类级支持 Exp，C3 类级支持 Recip/Rsqrt/Log；使用 64 项 LUT、硬件相同的系数量化、
插值/多项式、指数恢复及 RNE，不调用数学库直接生成函数结果。
每个 Tile、每个 Lane、每个函数各有独立 LUT；多个物理级请求同一 Lane 的同一函数
仍共享该 LUT 的单读端口。模型对这种非法编译时调度直接抛出异常，不能用它仿真非法冲突后的任意数据。
LUT 编程接口要求 Tile 空闲，向各 Lane 副本广播写入。

## 构建与黑盒对拍

```powershell
cmake -S . -B build-rtl -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DFTLPU_HARDWARE_CONFIG=$PWD/config/ftlpu-rtl-lpu32.json
cmake --build build-rtl
ctest --test-dir build-rtl --output-on-failure
python scripts/test_rtl_alignment.py --vmodel <VMODEL目录> --probe build-rtl/rtl_probe.exe --output build-rtl/compare --samples 1024
python scripts/test_rtl_edges.py --vmodel <VMODEL目录> --probe build-rtl/rtl_probe.exe --output build-rtl/edges
```

需要 Python 3 和 Icarus Verilog；工具可位于 PATH 或 VModel 的 `build/iverilog/bin`。
测试只驱动模块外部端口并比较结果位模式、周期、tag、valid、consume、Bank 请求及 fault，
不调用 RTL 内部算术函数。输入、两侧逐拍 trace、日志及 SHA256 写到指定输出目录。
原生 C++ 单测还独立检查 50,000 组 FP32 加乘除，并包含数值相消测试。
MXM 入口对拍覆盖全部 13 种权重/scale/zero 组合、双缓冲交替、
非法指令、字节缺失及同组冲突。每次运行在输出目录记录对应源码摘要和比较结果。

Icarus 对 VXM 原代码的 package 常量、packed struct cast 和组合逻辑调度存在兼容问题。
runner 只在测试输出目录生成副本：补全 package 名、将等宽 packed struct 的恒等 cast
改为直接赋值，并把无环加法组合逻辑按 front/shift/pack 分成三个 process。
运算语句和寄存器不变，VModel 原文件不改；结果文件记录原文件和副本的 hash。
这不是商业仿真器或形式等价验证的替代品。

## 兼容边界

`TspSliceSystem`、旧 ISA、旧 workload 和 SXM/C2C/ICU 实现继续保留。
使用仓库默认目标即可构建现有整芯片目标。切换至 RTL 专用配置时，默认构建独立 RTL 模型与测试。
新 CModel 已包含完整 MXM 阵列→回传→后端通路，MEM 为完整 Bank column，VXM 为完整 Tile；
尚未把新的三个模块替换进旧整芯片网络。MEM→SRF→MXM/VXM 的整机程序仍需后续在该集成边界迁移，
不能把旧整芯片 kernel 回归通过解释成新模块整机对拍完成。
