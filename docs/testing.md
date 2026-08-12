# Test Organization

Tests are organized by behavioral scope rather than by implementation history.

| Layer | Directory | CTest prefix | Scope |
| --- | --- | --- | --- |
| Unit | `tests/unit` | `unit.` | One class or functional block without a whole-chip route |
| Subsystem | `tests/subsystem` | `subsystem.` | Multiple hardware blocks connected through physical SR routes |
| Kernel | `tests/kernel` | `kernel.` | One neural-network operator or fused operator pipeline |
| Model | `tests/model` | `model.` | Model-shaped phases and complete layer schedules |

Shared test-only helpers live in `tests/support` and are not registered as
tests.

CTest names use dot-separated scope from broad to specific. Build target names
retain their existing underscore form so established build commands continue to
work. Examples:

```text
unit.mxm.array
subsystem.c2c.dual_chip
subsystem.c2c.dma_ddr4
kernel.swiglu.w8a16
model.smollm2.prefill_attention
```

Run one layer or one hardware domain with labels:

```powershell
ctest --test-dir build-vs2026 -C Release -L unit
ctest --test-dir build-vs2026 -C Release -L subsystem
ctest --test-dir build-vs2026 -C Release -L kernel
ctest --test-dir build-vs2026 -C Release -L model
ctest --test-dir build-vs2026 -C Release -L mxm
ctest --test-dir build-vs2026 -C Release -L fast
```

Use the lowest layer that proves the behavior. A model test should not be added
when a unit or subsystem test can express the same contract, and model-shaped
regressions should replace older partial workload tests instead of duplicating
them indefinitely.
