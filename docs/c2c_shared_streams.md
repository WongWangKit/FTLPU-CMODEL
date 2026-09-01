# Shared C2C Receive Path

The external C2C interface has a runtime-selectable 1..8 lanes per direction.
Each lane transfers one complete 32-byte vector per cycle. Lane IDs are
transport channels; RX maps them onto ordinary westbound SR IDs at the
MEM/C2C edge. With eight lanes the default mapping is `W24..W31`.

There is no direct-to-SRAM or dedicated-stream receive path:

```text
host -> DDR4 -> C2C DMA -> C2C RX -> ordinary west SR -> MEM Write -> SRAM
```

Every received vector sends a point-to-point notification to the target
`(hemisphere, slice, bank)` MEM ICU C2C command context. A coarse synchronized
write descriptor records the vector count, destination base/stride, SR ID, and
fixed RX-to-MEM transport latency. The MEM ICU retains each notification's
arrival cycle and issues the corresponding `Write` when
`notify_cycle + transport_latency` is reached. Consecutive vectors therefore
write one per cycle, while a DDR latency bubble automatically stalls the MEM
consumer instead of reading an invalid SR segment.

The C2C command context is a synchronization/command context, not a separate
data path. It shares the physical MEM port with the normal compute MEM queue;
issuing both in one cycle is a static schedule conflict. C2C traffic also obeys
ordinary SR producer and route conflicts. Page readiness means the final MEM
write was issued, not merely that DMA delivered the final vector to RX.

## DDR timing model

DDR is an external backing store, not an LPU-MEM initialization shortcut.
`Ddr4Model::initialize_vector` and `read_vector` represent host-side DDR
access; all LPU-bound data still crosses DMA, C2C, SR, and MEM Write.

The default platform assumes a 500 MHz LPU and dual-channel DDR4-3200. Its
51.2 GB/s aggregate peak is modeled as a fractional 102.4-byte-per-cycle token
budget shared by reads and writes. Read latency is 35 cycles plus 0..15 cycles
of deterministic jitter; write latency is 25 cycles plus 0..10 cycles. Queue
depth, channel count, bandwidth, base latency, jitter, and seed are configurable.
