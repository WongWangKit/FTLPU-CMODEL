# Shared C2C Receive Path

The external C2C interface has a runtime-selectable 1..8 lanes per direction.
Each lane transfers one complete 32-byte vector per cycle. Lane IDs are
transport channels; RX maps them onto ordinary westbound SR IDs at the
MEM/C2C edge. With eight lanes the default mapping is `W24..W31`.

There is no direct-to-SRAM or dedicated-stream receive path:

```text
host -> DDR4 -> C2C DMA -> C2C RX -> ordinary west SR -> MEM Write -> SRAM
```

Each physical `(hemisphere, slice, bank)` owns exactly one MEM ICU, one local
i-MEM, one FIFO, and one program counter. Its ordinary `READ_3D`, `WRITE_3D`,
and `WRITE_TAP_3D` commands share that same program stream with
`MEM_WRITE_SYNC`; there is no C2C sidecar MEM queue or second command context.
`MEM_WRITE_SYNC` occupies two consecutive 96-bit i-MEM words: a synchronization
header followed by the native MEM write template.

The runtime loader inserts the two-word command into the MEM program before
launch. It does not append to or modify local i-MEM after launch. When the
single program counter reaches `MEM_WRITE_SYNC`, that command owns the queue
head and waits for the matching point-to-point notifications from C2C RX.
Later ordinary MEM commands cannot bypass it. The descriptor records the
vector count, destination base/stride, SR ID, and fixed RX-to-MEM transport
latency. Each notification token carries its own relative age counter, starting
at zero and advancing once per ICU tick after dispatch. Its corresponding
`Write` becomes eligible when the age reaches `transport_latency + 1`, including
the registered arrival boundary. The ICU stores and compares no global or
absolute cycle. Consecutive vectors therefore write one per cycle, while a DDR
latency bubble stalls the sole MEM command stream instead of reading an invalid
SR segment.

Each `MEM_WRITE_SYNC` also reserves a statically scheduled ownership window for
the MEM ICU. If all vectors arrive and are written early, the command remains
at the queue head until that window expires, preserving the following static
schedule. If data arrives late, the command keeps the queue past the reserved
end until its final write issues. C2C traffic still obeys the ordinary MEM port,
SR producer, and route conflicts. Page readiness means the final MEM write was
issued, not merely that DMA delivered the final vector to RX.

## DDR timing model

DDR is an external backing store, not an LPU-MEM initialization shortcut.
`Ddr4Model::initialize_vector` and `read_vector` represent host-side DDR
access; all LPU-bound data still crosses DMA, C2C, SR, and MEM Write.

The default platform assumes a 500 MHz LPU and dual-channel DDR4-3200. Its
51.2 GB/s aggregate peak is modeled as a fractional 102.4-byte-per-cycle token
budget shared by reads and writes. Read latency is 35 cycles plus 0..15 cycles
of deterministic jitter; write latency is 25 cycles plus 0..10 cycles. Queue
depth, channel count, bandwidth, base latency, jitter, and seed are configurable.
