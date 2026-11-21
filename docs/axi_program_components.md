# Direct program loader / shared AXI: DEV semantic merge (2026-11-17)

Fixed upstream: `ff1aa41321e07667ebd37f3ebefdf8053f0ee0e5`; local input and all dirty changes are fingerprinted in `learn/c2c_dma_dev_merge_20261117/results/audit001/workspace_before.json`.

**Default program path:** DDR -> shared multi-ID AXI Master / Program Read client -> bounded RX bytes -> reassembly -> Program Word FIFO -> 2-entry Distributor ingress -> registered East/West Relays (MEM/MXM), or registered direct route (other targets) -> private iMEM -> ICU -> FU.

* No program bank claim, `ProgramPageStore`, program page cache, data-MEM write, or ordinary SR program transfer remains. Every nonresident reload reads DDR again.
* `program_read_client.hpp` reserves RX capacity before submission, retains owner/generation through completion, and drains already-admitted requests on failure. `InstructionFetchDma` protocol-component tests use the same physical AXI Master, not a second DDR service.
* WRITE and PRIME use the same finite route. PRIME seals each image only after all words commit; frontend-ready/dependencies gate START. Loading is not execution completion.
* Safe single-slot runtime segmentation, RegisteredR1, finite DMA descriptors/notifications and recovery survive the merge. The MEM32/MXM40 Macro-context budgets remain the local product baseline; large iMEM defaults are NOT optimized by this merge.
* DEV global LOW/POSEDGE/HIGH is used. The local MXM row-0 dispatch contract is retained (same-cycle combinational consumer as MEM, then registered inter-tile propagation); all architectural writes still commit at the edge. This is one product contract, not a selectable legacy mode. Blindly copying DEV's extra queue-to-row0 cycle misaligns the existing R1 software schedule.
* PRIME materializes logical cycles 0/1 into the registered Timed FIFO before START. Transport owner/overlap configuration is staged after task reset, so BARR observes the new task's actual MEM commits. Standalone ModelSession transport append waits for a real reset edge before installing packets; it no longer races a pending reset.
* Program RX currently uses `dma_fifo_depth_vectors * AXI beat_bytes` (512 B in the default configuration), distinct storage but a coupled tuning knob. Decoupling that configuration and optimizing all FIFO/iMEM capacities belong to the following measurement phases; no optimal depth has been selected.
* No DEV RTL merge. G-MERGE and G-CMODEL status must be read from the latest spec/report, not inferred from individual test passes.

## Historical pre-v3 records (retired architecture, not current APIs)


> **历史：2026-11-17 bank 路线状态**：bank/R1、统一失败任务恢复与 MEM/MXM 分页已推进到默认 runtime；下文含历史组件阶段记录。当时事实以 [原 bank 闭环报告](C:/Users/huazh/Desktop/LPU/lpu_src/FTLPU-VMODEL/spec_new_c7ae225/exploration/c2c/阶段1_2_程序bank_R1_回收恢复与MEM_MXM分页.md) 和主 spec §21 为准。完整注册回归仍有失败，阶段 1–2 未全部验收。


> 历史状态更新：默认系统联合接入和完整工具链已实施；下文保留此前组件/调查阶段记录。当前事实、失败与未关闭验收以 [默认通路联调报告](C:/Users/huazh/Desktop/LPU/lpu_src/FTLPU-VMODEL/spec_new_c7ae225/exploration/c2c/阶段1_2_默认通路联调与完整回归.md) 及主 spec 第 20 节为准。

# AXI / shared-program components: stage 1–2 implementation status

These components are on the local `product/c2c-dma-stage12` branch based on
`c49f6ac0dcc455b84c8f69d46c7518233310a09a`. They are **not yet a replacement
for the default C2cDmaSystem / runtime path**. No RTL or Peer implementation
is included.

## Components and contracts

| Component | Contract |
|---|---|
| `c2c/axi_dma.hpp` | One physical AXI4 master, multiple IDs, per-ID/global admission bounds; one preplanned INCR burst per submission |
| `system/program_page_store.hpp` | Finite page slots in an existing MEM bank; one shared read/write acceptance port; scheduled write commit and reserved read returns |
| `system/program_segment_loader.hpp` | Explicit safe-boundary segments; shared pages → finite byte FIFO → registered local write → real ICU RAM and frontend |
| `mem/mem_array.hpp` | Whole-bank claim/release and rejection of normal instructions or host data initialization into a claimed program bank |

### AXI edge convention

1. Submit a complete burst (write payload must already be present) and reserve
   payload, transaction and client-completion storage.
2. Call `eval()` to observe the held AR/AW/W outputs.
3. Call `tick(slave_signals)` once at the sampling edge. The subordinate must
   only return responses for transactions accepted before this edge.
4. Retire with `completion_ready(client)` / `pop(client)`. Completions from
   later IDs remain resident until earlier requests for that client retire.

AR and AW/W offers do not wait for READY. AW and W progress independently;
W follows submission order and never interleaves bursts. Full RX reservation
allows RREADY to remain asserted even while all clients stop consuming.
Reservation lasts until client retirement, which is intentionally conservative
and must be included in later capacity optimization.

Read/B errors stop new client admission. Already submitted work finishes, and
completion records carry error/aborted/transfer length. `abort_client` stops
new admission; it does not withdraw held VALID or attempt AXI cancellation.
`resume_client` requires all of that client's records to retire. There is no
automatic DDR retry or rollback.

Supported address subset: aligned, full-width INCR bursts with a byte-masked
write tail. `plan_axi_bursts` splits at 4 KB and maximum burst length; zero
length produces no burst. Misalignment and range overflow are rejected before
publication. Scatter/stride descriptor scheduling and source-return reservation
are not implemented in this class.

Protocol reference: [Arm AMBA AXI/ACE IHI 0022H](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf).
This is a cycle model and directed/random test suite, not AXI VIP certification.

### Shared bank and page lifetime

The page directory contains no shadow page payload: bytes are committed into
the claimed `MemArrayModel` bank. Page validity is published only when every
scheduled byte has actually committed. Read acceptance reserves a return slot
and increments the page reader count; this count is held until response pop.
A page with readers, pins or pending filling cannot be evicted.

Per-byte scheduled bits are model checking metadata, not proposed hardware
area. Bank claim is a task-level action and currently requires MEM idle.
The controller must explicitly release the bank after references and accesses
drain. Reset/clock-gating/CDC are outside this component's verification scope.

### Segment boundary

The caller supplies a retirement predicate covering the old program's FU/SR
and endpoint side effects; ICU queue completion is checked separately. This
implementation resets all ICU program queues at that explicit boundary.
It therefore does **not** yet implement per-target live replacement while
unrelated ICU transport programs continue. Unrelated AXI clients do not need
to reach DDR-global idle.

The image set and generation are validated before clearing the old program.
Code bytes are immutable within one generation; source address plus generation
identifies cached pages. All local images commit before priming/start.
The private capacity limit applies per target; no hidden standby slot exists.
A manifest fingerprint mismatch is rejected, but the current caller supplies
the fingerprint string: automatic JSON/ABI hashing and runtime export remain
to be integrated.

The test loads independent native MEM and MXM-control fixtures of 96/128 bits
across 24-byte transport pages, then executes a MEM Read on the real TSP model.
It covers three distinct segments and reload of the first. MXM uses NOP
control words; it does not prove new MXM numerical execution. Packed Macro,
SXM/VXM product templates, all six control-image classes, runtime epoch paging,
program/data contention, shared-cache hits and whole-system pause still need
dedicated integration tests.

## Tests

```powershell
cmake --build <build> --target axi_dma_test program_segment_loader_test
ctest --test-dir <build> -R "unit.c2c.axi_dma|subsystem.program.segment_loader" --output-on-failure
```

Reproducible local matrix, immutable logs/configs and known failures are in
[C2C stage 1–2 experiment](C:/Users/huazh/Desktop/LPU/learn/c2c_dma_stage12_20261116/README.md).
Do not mark T10–T12 or either complete phase accepted from these component
tests alone.

