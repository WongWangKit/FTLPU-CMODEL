# Checkpoint 1：StreamRegisterFabric 迁移与复验

## 结论

Checkpoint 1 已从当前 `FTLPU-CMODEL-main` 完整迁移到独立工作区，并在新路径完成
干净编译和回归。结论为 PASS，已知 RMSNorm 基线失败除外。没有开始 MEM 迁移。

## 复制规则

完整复制源码、测试、文档、配置、工具以及 `clock_experiment`。只排除源根目录
`D:\LPU\FTLPU-CMODEL-main\out`，因为它是可重新生成的旧构建缓存。实验目录内部的
CSV、Comparator、编译日志、VCS 日志和历史实验输出全部保留。

复制前后清单均为 861 个文件，逐文件大小和 SHA-256 完全一致。23 个 Checkpoint 1
关键文件另有独立哈希表。

## 实现核对

- `include/ftlpu/core/timing.hpp`：ClockPhase、ClockParticipant、ClockedModule、Clock、
  SystemPhase、ClockedState、Register 和 RegisterBank。
- `include/ftlpu/core/stream_fabric.hpp`：专用稀疏 Current/Next，LOW staging，统一
  validate/commit，多生产者冲突检查和提交周期计数。
- `include/ftlpu/system/tsp_slice_system.hpp`：Clock 注册 East/West，SystemPhase 显式
  LOW/HIGH 节点，完整四阶段 tick。
- `include/ftlpu/sxm/matrix_mem.hpp`：参考模型驱动适配新阶段接口。

任务说明中要求保留的 `stage_write()` 和冲突检查仍存在。早期
`begin_cycle()`/`commit_cycle()` 已在正式 Checkpoint 1 中被
`low()`/`validate_posedge()`/`posedge()` 取代；复验以实际冻结实现为准，没有恢复
已删除的兼容 API。

## 专项覆盖

1. Source Register -> Combinational A -> Combinational B -> Destination Register。
2. LOW 使用旧 Current 准备 Next，HIGH 读取新 Current 且不修改 Next。
3. Clock 注册正序/逆序不改变 SystemPhase 拓扑或结果。
4. RegisterBank enable、hold、同步复位和一次性 LOW 副作用。
5. East/West Fabric 统一验证后提交；预期验证失败前不发生部分提交。
6. Tsp 每个 tick 经过四个阶段，两个 Fabric 提交周期与全局周期锁步。

## 复验结果

| 项目 | 结果 |
|---|---|
| CMake configure | PASS |
| Release clean build | PASS |
| Checkpoint 1 focused tests | 5/5 PASS |
| fast tests excluding known RMSNorm | 59/59 PASS |
| fast tests including RMSNorm | 59 PASS，1 known baseline failure |

已知失败原文：

`VXM input Buffer cannot be reconfigured while a Bundle is active`

本批不把该既有失败算作 Checkpoint 1 回归，也不提前修改 VXM。

## 证据路径

- `timing_refactor/checkpoints/checkpoint_1_evidence/cmake_configure.log`
- `timing_refactor/checkpoints/checkpoint_1_evidence/build_release.log`
- `timing_refactor/checkpoints/checkpoint_1_evidence/ctest_checkpoint1_focused.log`
- `timing_refactor/checkpoints/checkpoint_1_evidence/ctest_fast_all.log`
- `timing_refactor/checkpoints/checkpoint_1_evidence/ctest_fast_excluding_known_rmsnorm.log`
- `timing_refactor/manifests/checkpoint_1_key_hashes.csv`

