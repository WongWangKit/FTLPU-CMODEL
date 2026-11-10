# MEM read timing: main default and opt-in R1

The local MEM changes were integrated on top of main `76a0b2e` on 2026-11-10.
Horizontal MEM grouping already uses `stream_topology.bindings.mem.slices_per_group`
upstream; it remains independent of the vertical tile count.

## Default is unchanged

Existing `MemArrayModel()` and two-argument construction keep the upstream timing:
Read samples SRAM and stages its SR output in the same `evaluate()` call, followed
by the owning fabric's commit. Existing replay/ICU schedules need no extra cycle.
Upstream transport provenance checks and SRAM commit accounting remain enabled.

## Explicit registered-read mode

```cpp
ftlpu::MemArrayModel mem(
    ftlpu::MemStreamPortMap::BetweenBoundaries(),
    ftlpu::MemArrayModel::MissingStreamPolicy::Error,
    ftlpu::MemArrayModel::ReadTiming::RegisteredR1);
```

- At A, SRAM data/address/tile/stream/tag are captured in `read_responses()`.
- At the next evaluate, the captured response is staged for fabric commit B;
  SRAM is not reread. `staged_read_responses()` is an observation, not another stage.
- Write/WriteTap sampling and consumption do not change.
- `idle()`, `compute_idle()` and `transport_idle()` include the corresponding
  pending R1 read responses. Staged responses belong to the fabric's lifecycle.
- Reset clears pending responses and observation/commit counts, retaining the
  selected mode; execution reset preserves SRAM.
- In Main mode, both response views are empty. Execution trace remains at the
  SRAM sampling cycle in either mode and does not control correctness.

Mode is chosen at construction, not switched during an active command wave.
No system-wide R1 switch is introduced: an R1 consumer schedule must explicitly
account for its extra response edge. This merge does not retime ICU/MXM/VXM.

## Local MEM profiles and tests

`config/mem-profiles/ftlpu-lpu{16,32,64}.json` preserve the local 2/4/8-tile,
52-slice, two-bank, 1024/2048/2048-row geometry and four slices per horizontal
group. Only obsolete MXM mode declarations and required C2C beat configuration
were migrated to the current schema. Existing filenames/target identifiers are
retained; the historical `c7-r1` name does **not** select a read timing mode.

- `unit.mem.array` tests Main; `unit.mem.array_r1` runs the same cases explicitly in R1.
- `unit.mem.sync_read` retains the local R1 phase/snapshot/throughput/reset tests.
- `unit.mem.read_mode` compares both timings, tail idle domains, trace-disabled
  transport commit counters, reset and missing/wrong-provenance payload rejection.
- Upstream `unit.mem.replay_copy` keeps its unchanged Main-mode schedule.

These tests are not a claim of full-system R1 schedule compatibility.

## Initial MEM-only validation (2026-11-10, before the SXM follow-up)

Native MinGW Debug, assertions enabled: all seven MEM tests passed under the
upstream LPU32 profile and under each of the three local MEM profiles (28 runs).
The unchanged upstream program-loader and C2C/DMA subsystem tests also passed
using the default Main timing.

`subsystem.sxm.mem_transpose` failed at system cycle 23, hemisphere E, with
`SXM transpose buffer is full at tile 0 (ready_cycle=22)`. The same test on pristine
main `76a0b2e` reproduced the exact failure with identical build configuration.
At that stage this was an upstream issue; the MEM-only change did not alter SXM or its schedule.
Full-system RegisteredR1 operation and the full CModel suite were not validated.

Subsequent SXM work fixed the cycle-23 test scheduling problem without changing
MEM Main timing. The earlier failure above is historical, not an outstanding
failure in the current combined change. See [the fix](sxm_mem_transpose_timing.md)
and [the current regression commands/results](sxm_mem_regression.md).

The subsystem test build used MinGW `-Wa,-mbig-obj` to lift the object-section
limit. That flag was confined to the experiment build, not product CMake settings.
