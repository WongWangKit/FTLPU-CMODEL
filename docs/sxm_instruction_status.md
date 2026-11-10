# SXM 指令现状与矩阵参考路径改造

更新：2026-11-10。范围：**一个矩阵功能＋一个启动命令＋MEM 正常行读出适配**。
本文件说明本轮实际实现，不替换旧 ISA 说明，也不宣称全芯片已经切换到新路径。

## 1. 结论与基线

- 保留旧 SXM 指令及现有工作区修改；新增独立启用的 `SxmMatrixMemReference`。
- 新路径只做 **ROW → BLOCK8 的矩阵转置，以及转置内的合法 Tile/lane 映射**，支持 BF16、FP32。
- 没有新增独立 VECTOR_PERMUTE、广播、BLOCK 输入、通用 ROW 输出或 P-pair/RUN 前端。
- MEM 源码、默认 `ReadTiming::Main`、MXM/VXM、ICU/System 和 RTL 均未修改。
- CModel 代码基线：`76a0b2e5c27a21e0185e22798fbe5e0abb766ef5` 加本轮开始前的本地修改。
- RTL 合同来自本地 VModel `9bd20a29a614e22f8d1b4bd2f3436359e744f55e`：
  `sxm_triangle_core.sv` 的 mode=0、`lpu_sxm_triangle_mem_adapter.sv`、
  `lpu_sxm_writetap_completion.sv` 和 `lpu_sxm_triangle_mem_integration.sv`。
  本轮没有更新/检出远程分支，也没有把历史 RTL 全链成绩算成本轮验证。

代码入口：

| 文件 | 职责 |
|---|---|
| `include/ftlpu/sxm/matrix.hpp` | 启动参数、私有配置、三角源槽、逐行 CAP 和定时宽 P |
| `include/ftlpu/sxm/matrix_mem.hpp` | bank 条带、原生 WriteTap 错拍、实际 commit/drain、正常 Read 日历 |
| `include/ftlpu/sxm/matrix_vxm.hpp` | 两拍 FP32 → 现有 CModel VXM 同拍 API 的显式边界适配 |

## 2. 到底改不改 SXM 指令编码？

**本轮不改任何已有二进制编码。** `SXM_MATRIX_START` 是新路径唯一的逻辑启动命令，
代码表达为 `SxmMatrixStart` 加 `reference.start(job, resources_reserved)`。
它对应 RTL typed batch 接口，不是已分配 opcode 的 ICU 指令，也不是一条已经能被原 ICU 解码的机器码。

| 入口 | 当前状态 | 本轮变化 |
|---|---|---|
| 旧 `Transpose` / `Permute` 与 T/P、RUN_2D | 继续走原 `SxmSlice`、原 codec 和 ICU | 不替换、不重编码；保留之前 C23 排程修复 |
| 旧 `ShiftSelect` / `Distribute`、软件 fanout | 原软件参考仍存在 | 不加入新矩阵路径，不作为 RTL 支持声明 |
| `SXM_MATRIX_START` | 新的模块级 typed API | 复制一次私有配置，自动推进整批 CAP/P/MEM 写入 |
| MEM `WriteTap` | 原 `Write` 加 `preserve_stream=true` | 只新增 SXM 专用发令日历，MEM opcode/执行语义不变 |
| MEM `Read`、VXM 指令 | 原有接口 | Read 选择 bank、地址、stream 和发令拍；VXM 实现不变 |

`SXM_MATRIX_START` 参数：

| 参数 | 含义与限制 |
|---|---|
| `matrices` | 矩阵数，不是 Tile 数；1…65535，还必须满足地址范围 |
| `dtype` | BF16 或 FP32；不接受 FP16/INT8/FP8 标签 |
| `tile_map[a]` | 输入块行 a 发往哪个物理 Tile；必须是 0…T−1 的双射，不支持跨 Tile 复制 |
| `lane_map[d][l]` | 目的 Tile d 的 lane l 选择块内哪一原始行；0…7，可在 Tile 内重复选择 |
| `mem_base` | 每个条带 bank 的起始 SRAM 行地址 |
| `resources_reserved` | 调用者已独占预留 bank/stream/地址/写服务窗口的声明；不是新指令位 |

构造函数另收 `reserved_depth`；要求 `base + matrices*T <= reserved_depth <= min(配置深度,32768)`。
调用方若缩小了 MEM active rows，预留深度也必须相应缩小。此处没有增加全局资源分配器。
非法启动参数在 API 准入时拒绝；运行中的坏 CAP/P/commit 会进入 sticky fault，必须显式 reset。

**发令变化说人话：**旧路径要安排每次 T/P 的发令，且受 ICU 取指带宽约束；新路径先启动一批，
然后按日历送入行数据，计数器负责块完成和输出。不能把新 START 伪装成旧 RUN_2D，
也不能把模块级启动可用写成 ICU macro/编译器 lowering 已完成。

## 3. 周期、Tile、数据的关系

T=2/4/8 对应 LPU16/32/64，矩阵尺寸 N=8T，每个源 s 每拍最多送一个 8 元素行片段。
输入容器每元素 32 bit：BF16 低 16 bit 有效，FP32 全 32 bit 有效。
**主 SXM 入口和宽 P 均是完整元素；FP32 高低两拍位于后面的 MEM → VXM 行接口。**

令 job 接受为 C0，m 为矩阵编号，a 为输入块行，r 为块内行，s 为源 Tile：

```text
CAP(m,a,r,s) = 1 + 8T*m + 8a + r + s
P(m,a,s)     = 9 + 8T*m + 8(a+s)
目的 Tile d  = tile_map[a]
输出[q][l]   = 输入块[lane_map[d][l]][q]
```

- 连续矩阵的源间隔为 8T 拍；同源完成块的 P 间隔为 8 拍。
- 每个源 s 有 s+1 个槽，私有读写指针、接收计数和 16-bit due timer。
- row 7 的 CAP 后至少下一拍才能 P；同拍 P 先读旧值、释放，随后允许下一块 row 0 复用。
- 配置在 START 时复制，修改调用者的 job 对象不会改变已经在途的矩阵。
- 对固定 WriteTap 日历，迟到/缺失 P 直接报错，不通过增加队列、悄悄推迟发令来补救。
- `core.busy()==false` 只表示源块已输出；写任务要等实际 MEM 提交及指令/数据 drain。

LPU32 连续 A/B 两矩阵，identity 映射：

| 矩阵 | 首 CAP | 末 CAP | 首 P | 末 P | BF16 末提交 | FP32 末提交 |
|---|---:|---:|---:|---:|---:|---:|
| A | C1 | C35 | C9 | C57 | C60 | C62 |
| B | C33 | C67 | C41 | C89 | C92 | C94 |

C33 起两张矩阵的输入重叠；C41 的宽波也会同时包含 A/B 的不同块，必须按各 Tile 私有身份解释地址。
此日历在 C23 不需要额外的 P 指令取指，源槽仍有界。此前旧用例 C23 的问题和修复继续见
[旧路径 C23 排程说明](sxm_mem_transpose_timing.md)，两者不是同一条执行路径。

## 4. MEM 条带与错拍发令

设 B=2(BF16)/4(FP32)，q 为输出块内行，b 为元素字节号：

```text
j = B*q+b                          # West stream / 逻辑 byte-plane bank
g = floor(j/8)
物理 MEM group = 12-g
组内逻辑 bank = j%8
SRAM 行 = base + m*T+s
```

CModel 原有每 group 为 4 slices × 2 banks；逻辑 bank 映射为 `slice_offset=(j%8)/2`、
`bank=(j%8)%2`，不改变 MEM 几何。BF16 使用 16 banks/2 groups，FP32 使用 32 banks/4 groups。
宽 P 直接 stage 到现有 West SR column 14；目标采样距离 H=2+g。

某波 `P_k=9+8k`，`n=k-inverse_tile_map[d]` 有效时：

```text
根 WriteTap 发令 = P_k + H - d
目标 Tile d 采样 = P_k + H
WriteTap 地址    = base+n
```

原 MEM 根命令每拍向下一 Tile 传播。非目标 Tile 的空 Tap 合法；目标 Tile 缺数据必须 fault。
计数依据是 `MemArrayModel::executed_transfers()` 的**实际写入**，不是命令数。
MEM 读到旧 SR 值后，适配器在相同周期标记最后使用 consume，阻止残留向后传播。

每批应提交 `matrices * 8 * T*T * B` 个 8B 段。最后 P、最后提交、writer 退场和 done 分开：
writer 尾部到 `last_P+max(H)+T`，completion 下一拍检查所有状态清空及总数相等。
`done()` 是单拍事件，`statistics().done_cycle` 保留完成边沿编号；下一次 tick 清除事件。
LPU32 双矩阵参考结果：BF16 在 C97、FP32 在 C99 的 tick 完成后可观察 done。
这些是本 CModel 入口的周期定义，未用本轮 RTL 仿真重新签核外部系统 done 的观察拍。

## 5. MEM 行读出和 VXM 配对

写任务 drain 后，读转置矩阵第 y 行，使用普通 Read：
`s=y/8, q=y%8, address=base+m*T+s`，选择 q 对应的 B 个 bank。
默认消费点为四个 MEM group 之后的 West column 9，与 RTL 模块级读边界一致；
可显式配置更西侧消费点，发令日历随 SR hop 调整。

对默认消费点，Read 从根采样到 Tile 0 可见的距离为 `4-g`；
要求 Tile 0 在 E 收低半，则 `root_read=E+phase-(4-g)`，Tile t 在 `E+phase+t` 收到。
BF16 一拍两个 byte streams；FP32 **先低两字节、下一拍高两字节，复用同一 stream 对**。
MEM 保持 Main：本拍采样/stage，commit 后下一拍 SR 可见；没有启用 RegisteredR1。

物理 VXM 头 h 的 LHS stream 对从 2h 开始、RHS 从 2h+2 开始，镜像头是 h+8。
边界适配器为四个独立操作数分别保存相位，不复制镜像、不令 RHS 偷用 LHS；缺字节、
重复相位、高半先到或低高不相邻均拒绝。随后按现有 `VxmLane` 固定绑定重组 wide API 输入。

**必须保留的版本差异：**CModel 的 FP32 是同拍 4 字节，绑定为
`(stage/4)*8 + (rhs?4:0)`。chain2 的 h0/h2、h4/h6 分别共享绑定，
无法在这个 API 上同时表示各自独立 FP32 输入。本轮逐个“头＋镜像”验证；
这证明链2/4/8各头的数值和输入可达，不证明所有 chain2 头的 RTL 并发吞吐。
为保持 VXM 不变，没有把这个差异藏在 VXM 内部，也没有声称两套接口逐拍等价。

## 6. 接入方法与容量

新模型不会自动替换 `TspSliceSystem` 的原 SXM。使用者显式构造 MEM、SR fabric 和矩阵 reference，
配置同一实际 SRAM 深度，然后执行：

```cpp
SxmMatrixStart job;
job.matrices = 2;
job.dtype = SxmMatrixDtype::Float32;
job.mem_base = 17;
reference.start(job, /*resources_reserved=*/true);
reference.tick();                 // C0：接受启动，不送 CAP
// C1 起，每周期调用 tick(rows)，按第3节给各 source 送 8 个完整元素。
// 不在外面再 tick 同一 MEM/fabric；reference.tick() 拥有整个写周期。
// busy 退出后用 SxmMatrixRowRead::plan 发普通 Read；明确使用 Main 时序。
```

读取阶段由集成调度器执行 `begin_cycle → enqueue到期Read → mem.evaluate → 消费 → links → commit`。
下一次 start 会在 idle 时对齐已推进的 MEM/fabric 周期；允许不同 dtype/map/base 的新批次。
`reset_execution_state()` 协同清除 reference/MEM 指令/fabric 的执行状态，保留 SRAM 内容。
不要对正在被别的任务使用的共享底座调用此模块级 reset。

| 配置 | 源槽数 | 源 payload |
|---|---:|---:|
| LPU16/T2 | 3 | 768 B |
| LPU32/T4 | 10 | 2560 B |
| LPU64/T8 | 36 | 9216 B |

每槽 8×8×32 bit；**额外反向完整块队列为 0**。P 返回对象是当拍组合观察值，不是持久 FIFO。
WriteTap 适配层仅持有配置、计数等元数据，运输和落盘使用既有 SR/MEM。
`SxmMatrixVxmPairAdapter` 另外保存一个“四操作数×T×8×32bit”的软件边界 bundle，
payload 为 128T B（256/512/1024 B）。它服务于旧 CModel wide API，不计作新增 SXM RTL 硬件，
也不能拿其 C++ `sizeof` 推算综合 FF。源码里的主机 size_t/容器开销同样不等于 RTL 寄存器位数。

## 7. 验证和仍有的边界

测试入口：`unit.sxm.matrix`、`subsystem.sxm.matrix_mem_vxm`；LPU32 同时小规模回归旧 SXM/ICU/MEM/VXM。
仓库内可复现命令和发布范围见 [SXM/MEM 回归说明](sxm_mem_regression.md)，无需仓库外脚本。
结果与逐配置日志由项目根目录实验 `learn/sxm_matrix_cmodel_20261110/` 留存；见其 README 和 `results/final02/`。

最终 native Debug 回归（断言保持开启）：**LPU16 2/2、LPU32 18/18、LPU64 2/2，全部通过**。
每档新集成测试包含两种格式 × 两种映射的连续双矩阵，不是一个空壳启动用例。
LPU32 的 18 项包含旧路径双矩阵 smoke、旧 SXM/ICU、MEM Main/R1 及原 VXM 输入检查。

| 连续双矩阵 | BF16 实际提交段 / done 边沿 | FP32 实际提交段 / done 边沿 | 每种格式/映射的 VXM 数值输出检查 |
|---|---|---|---:|
| LPU16 | 128 / C47 | 256 / C49 | 224 |
| LPU32 | 512 / C97 | 1024 / C99 | 448 |
| LPU64 | 2048 / C197 | 4096 / C199 | 896 |

保护审计：324 个 MEM/MXM/VXM/ICU/System/RTL 文件 hash 未变；本轮开始时 12 个非 CMake 修改文件逐字节保留，
CMake 仅追加新目标。本轮无提交、推送或 L5 操作。

| 已验证层次 | 内容 |
|---|---|
| CModel 三档模块 | 连续双矩阵，BF16/FP32，identity/反向 Tile 双射、非恒等 lane map；逐 CAP/P、实际根命令传播/空 Tap/commit、全部正常 Read 值 |
| 控制与边界 | 私有配置、满占用 1…T、同拍释放复用、16-bit timer 回绕、busy/reset/restart、缺 P、非法配置、读写冲突、FP32 相位错误 |
| 真实 CModel MEM/SR/VXM 数值 | 不改 MEM/SR/VXM；正常 Read 穿过 SR；链2/4/8全部合法头及独立镜像/LHS/RHS；最后 MEM/SR 和 VxmLane datapath drain |

未闭合部分只列真实边界：

1. **ICU raw/macro/编译器 lowering 未接入。** 新 START 不会由旧 ICU 自动产生，原系统入口也没有静默切换。
2. **MXM 真实端到端控制链未新增验证。** 本轮按 typed raw-row 合同驱动源；没有强写 ACC，但也没有运行真实 MXM Compute/ACC_OUT 后直接联入新模型。
   本地 CModel 仍是原 `Compute/AccumulatorRead/IS` 控制参考，不能把它改名当作最新 VModel 的新 backend。
3. **VXM 两拍和旧 wide API 的差异保留。** 本轮是显式相位适配＋真实 lane 算术验证，不是完整 VxmSlice/ICU 时序和全头并发等价验证。
4. **只支持独占预留、有限批次、先写后读。** 没有并发地址分配、任意动态路由或无限生产；读日历当前保守串行验证，不宣称最大吞吐。
5. **未新增 RTL 仿真、差分、综合或 STA。** 三档通过只指本轮 CModel 模块测试；默认 Main 与 native RTL 其它读流水差异没有被修改掉。
6. **收缩掉的功能不新建模。** 旧软件模式继续保留；新路径不承诺全模式互斥/共享池迁移，也不新增向量 permute。
