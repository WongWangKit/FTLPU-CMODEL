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

The legacy direct-SRAM path is retained only behind
`c2c_dedicated_streams=true` for comparison tests. Deployment uses the shared
path. `c2c_dma_ddr4_test` checks one-lane and eight-lane shared ingress and
verifies every SRAM byte after MEM consumption.
