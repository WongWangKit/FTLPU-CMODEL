# Shared C2C Receive Path

The external C2C interface has a runtime-selectable 1..8 lanes per direction.
Each lane transfers one complete 32-byte vector per cycle. External lane IDs
and ordinary SR IDs are independent.

With `c2c_dedicated_streams=false` (the default), RX does not write SRAM
directly. A received vector is injected at the MEM/C2C edge into the selected
ordinary west stream. Its four tile segments form a one-tile-per-cycle
wavefront, propagate through the shared fabric, and are consumed by a normal
MEM `Write` instruction at the target slice:

```text
DDR4 -> DMA lane -> C2C RX -> west SR -> MEM Write -> SRAM
```

The first vector may notify the target `(hemisphere, slice, bank)` ICU queue so
a `Sync` releases exactly when data starts entering the fabric. Software must
then account for the target group's SR hop count and issue `Write`/`Repeat` at
the matching cycles. Multiple C2C lanes may target independent slices in the
same cycle. Ordinary producer collisions and MEM port conflicts remain errors.

With `c2c_dedicated_streams=true`, external lanes are independent of the 32
ordinary SRs. An RX command carries its destination hemisphere, slice, bank,
and row, and the target MEM receiver commits SRAM without occupying an ordinary
SR. The current `ModelSession` uses this mode for fine-grained weight prefetch.
In both dedicated and shared modes the source must be DDR DMA/C2C; the host
cannot write LPU MEM directly. `c2c_dma_ddr4_test` covers one-lane and eight-lane
shared ingress plus dedicated ingress and verifies every SRAM byte.

## DDR timing model

DDR is an external backing store, not an LPU-MEM initialization shortcut.
Test/runtime code may call `Ddr4Model::initialize_vector` or `read_vector` to
represent host-side DDR access, but data crosses the LPU boundary only through
DMA and C2C. The modeled ingress and egress paths are:

```text
host <-> DDR4 <-> DMA <-> C2C <-> [dedicated RX or SR/MEM] <-> SRAM
```

The default platform assumes a 500 MHz LPU and dual-channel DDR4-3200. Its
aggregate peak is 51.2 GB/s, represented as a fractional 102.4-byte-per-cycle
token budget shared by reads and writes. The default read latency is 35 cycles
(70 ns) plus 0..15 cycles of jitter; write latency is 25 cycles (50 ns) plus
0..10 cycles. Jitter is a deterministic hash of request ID, address, operation,
and seed: completion timing varies per request while regression tests remain
reproducible. Queue depth, channel count, bandwidth, base latency, jitter, and
seed are all configurable.
