# SXM/MEM 小规模回归与发布范围

更新：2026-11-10。本页随 CModel 源码发布，不依赖个人电脑路径、仓库外实验脚本或 VModel 工作区。

## 本次发布包含什么

- 保留 MEM 默认 Main 时序，提供已有本地改动中的显式 RegisteredR1 选项及其单测。
- 旧 SXM C23 buffer-full 测试改用合法 RUN_2D 发令；部分输入行占用也计入 busy。
- 新的独立 typed 矩阵参考：BF16/FP32、ROW → BLOCK8、私有配置和三角源池。
- MEM 多 bank 条带、原生 WriteTap 错拍、实际提交计数、正常 Read 行读回。
- 显式 FP32 低/高拍 → 现有 CModel VXM wide API 适配；不新增独立向量 permute。
- 对应源码、三档配置、CMake 测试注册、回归脚本和说明文档。

不包含 VModel/RTL、综合/布局布线/STA 结果、原始实验日志、构建目录、个人配置或凭据。
MXM/VXM/ICU 的实现未修改；新入口没有自动取代 `TspSliceSystem` 原 SXM。

## 独立克隆后的运行方法

需要 Python 3、CMake/CTest 3.21 或更新版本（JUnit 导出）和支持 C++20 的编译器。
工具需要在 PATH 中，也可以通过 `--cmake`、`--ctest`、`--compiler`、`--make-program` 指定。
以下命令均从 CModel 仓库根目录执行。

Windows/MinGW：

```powershell
python scripts/run_sxm_mem_regression.py --generator "MinGW Makefiles" `
  --compiler g++ --make-program mingw32-make `
  "--cmake-arg=-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g0 -Wa,-mbig-obj"
```

MinGW 的 `-Wa,-mbig-obj` 用于旧完整系统测试的较大翻译单元，不传给 MSVC/Linux 编译器。
MSVC 可在相应 Developer PowerShell 中选本机已安装的 Visual Studio 生成器；
Linux 或已配置编译器环境下可直接使用：

```text
python scripts/run_sxm_mem_regression.py
```

- 默认运行三档：LPU16 2 项、LPU32 18 项、LPU64 2 项。
- `--profiles 32` 只跑 LPU32；`--matrix-only` 只保留各档 2 个新矩阵测试。
- `--dry-run` 仅打印命令，不创建文件或执行工具。
- 默认产物位于 `build-sxm-mem-regression/<运行编号>/`，已被 `.gitignore` 排除。
- 每轮保存命令、构建/CTest 日志、JUnit、`summary.json`；不覆盖已有运行编号。
- 脚本只做 CModel 构建和测试，不运行 Vivado，不提交或推送任何仓库。

也可以直接使用产品 CMake 目标：

```text
cmake -S . -B build-matrix -DFTLPU_HARDWARE_CONFIG=config/ftlpu-lpu32.json
cmake --build build-matrix --config Debug --target sxm_matrix_test sxm_matrix_mem_vxm_test
ctest --test-dir build-matrix -C Debug -R "^(unit.sxm.matrix|subsystem.sxm.matrix_mem_vxm)$" --output-on-failure
```

## 已完成的验证

代码基线 `76a0b2e` 加本次本地修改；2026-11-10 的 native MinGW Debug 回归，断言开启：

| 配置 | CTest | 连续双矩阵覆盖 | 源槽峰值 | 源 payload |
|---|---:|---|---:|---:|
| LPU16/T2 | 2/2 PASS | BF16/FP32 × identity/反向 Tile 映射 | 3 | 768 B |
| LPU32/T4 | 18/18 PASS | 同上，另含旧 SXM/ICU/MEM/VXM 小回归 | 10 | 2560 B |
| LPU64/T8 | 2/2 PASS | BF16/FP32 × identity/反向 Tile 映射 | 36 | 9216 B |

额外反向完整块 payload 队列为 0。逐条检查根 Tap 到各 Tile 的传播、实际提交、全部正常 Read 回读、
FP32 相位、chain2/4/8 的各合法头和独立镜像/LHS/RHS 数值、最后 drain。
另测私有配置、同沿释放复用、timer 回绕、配置/缺 P/读写冲突/相位错误以及 reset/restart。

上述是此前 `final02` 源码回归结果的仓库内摘要，原始日志留在本地实验目录，不随此次提交上传。
包装回归脚本不改变硬件或测试语义。各档不是全芯片或完整神经网络回归。

## 必须保留的差异

1. `SXM_MATRIX_START` 是模块级 typed API，没有新增或冻结 ICU 机器码，没有接通 macro/编译器 lowering。
2. 输入按 MXM raw-row 合同驱动，本轮没有运行真实 MXM Compute/ACC_OUT 后端端到端联调。
3. CModel VXM 仍使用 FP32 同拍 4-byte API；相邻 chain2 头共享部分输入绑定。
   最新 VModel main `468ad6b` 使用固定 stream 对收发两次有效拍；这项历史差异尚未对齐。
   当前逐“头＋镜像”验证不能作为 RTL 全头并发或完整 Slice 时序通过的证据。
4. 仅验证独占预留、有限批次、先写后读，没有最大吞吐或无限输入保证；未新增 RTL/综合/STA 验证。

详细说明：[SXM 指令现状](sxm_instruction_status.md)、[旧 C23 修复](sxm_mem_transpose_timing.md)、
[MEM Main/R1](mem_read_timing.md)。

## 提交前检查

- 使用明确的文件清单 `git add -- ...`，不要从 LPU 外层仓库执行 `git add .`。
- `git diff --cached --check`、`git diff --cached --stat`、`git diff --cached --name-only` 检查暂存内容。
- `.gitignore` 不会追溯排除已跟踪文件；已跟踪的 VSCode 设置已移除本机编译器绝对路径，改用 CMake Tools。
- 凭据忽略规则只是兜底，提交前仍检查新增文件；不要强制添加被忽略的构建或私有配置。
- 本地提交后先 fetch/rebase 远程 main；出现冲突或推送被拒绝即停止，不使用 force push。
