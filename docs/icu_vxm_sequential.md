# VXM sequential configuration ISA

VXM no longer implements RUN_2D, RUN_3D, STREAM_ND, or ICU Repeat2D.
Recompile old programs. MEM/MXM/SXM loop instructions are unaffected.

Each of the eight shared C0–C7 queues controls its corresponding mirrored
ALU pair. A configuration occupies one 96-bit local iMEM word:

| Physical bits | Meaning |
|---|---|
| 1:0 | Common ICU envelope; `00` selects a VXM configuration |
| 55:2 | Used bits 53:0 of the existing compact FU control |
| 87:56 | FP32 immediate, bit-preserving |
| 95:88 | Reserved, must be zero |

The emitted FU configuration is still 96 bits (64 control + 32 immediate).
Its upper ten control bits are reserved. The physical iMEM packing uses two
of those otherwise unused bits for the common envelope; no precision or
repeat bits are dropped. There is no additional launch descriptor.

The existing NOP/WAIT count, SYNC, WAIT_EVENT, and NOTIFY controls retain their
common encoding. A NOP consumes its specified number of queue cycles. A
configuration consumes one ICU issue cycle; its `repeat_count` controls a
contiguous sequence of identical FU operations. A later configuration can be
issued early into the bounded Superlane FIFO/Next register while Current
executes. Only one configuration executes at a time; prefetch is not interleaved
execution. The physical three-entry FIFO capacity remains enforced.

There are no VXM multi-dimensional launch counters or active loop contexts.
The local iMEM frontend still fetches one word per cycle with the existing
latency and IQ capacity. Alternating accumulator reset/emit configurations
can therefore sustain one configuration per cycle after pipeline fill.

For RoPE, two products are accumulated with two sequential ADD configurations:
`reset=1, write=1, emit=0`, then `reset=0, write=1, emit=1`. A two-cycle output
pitch is supported by VXM without new accumulator behavior, ALUs, stream
ports, or contexts. Achieving that rate also requires continuous MEM supply.
The planar RoPE emitter computes even outputs and then odd outputs per tile:
two ascending X scans, matching prearranged coefficients, and strided result
stores avoid short alternating forward/reverse MEM descriptors. VXM can run
continuously inside a tile; SXM timing and buffer reuse still constrain tile
boundaries.

Tests: `icu_vxm_sequential_instruction_test` covers sustained alternating
configuration fetch/decode, compact bit preservation, and rejected 2-D control.
