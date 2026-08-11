# 测试分层与命名

测试按照行为范围组织，不再按照实现历史堆放。

| 层级 | 目录 | CTest 前缀 | 范围 |
| --- | --- | --- | --- |
| Unit | `tests/unit` | `unit.` | 单个类或功能块，不经过完整芯片路由 |
| Subsystem | `tests/subsystem` | `subsystem.` | 多个硬件单元通过物理 SR 路径协作 |
| Kernel | `tests/kernel` | `kernel.` | 单个神经网络算子或融合算子流水 |
| Model | `tests/model` | `model.` | 模型形状的阶段和完整层调度 |

测试专用的共享辅助代码放在 `tests/support`，不单独注册为测试。

CTest 名称从宽到窄使用点号分隔。构建 target 继续保留原来的下划线名称，
因此已有的 `cmake --build --target ...` 命令不受影响。例如：

```text
unit.mxm.array
subsystem.c2c.dual_chip
kernel.swiglu.w8a16
model.smollm2.prefill_attention
```

可以按层级或硬件域运行：

```powershell
ctest --test-dir build-vs2026 -C Release -L unit
ctest --test-dir build-vs2026 -C Release -L subsystem
ctest --test-dir build-vs2026 -C Release -L kernel
ctest --test-dir build-vs2026 -C Release -L model
ctest --test-dir build-vs2026 -C Release -L mxm
ctest --test-dir build-vs2026 -C Release -L fast
```

新增行为应放在能够证明它的最低层级。Unit 或 Subsystem 足以表达的契约，
不要再增加 Model 测试；已有完整模型回归后，也应删除被覆盖的旧局部 workload，
避免长期维护重复的离线排程。
