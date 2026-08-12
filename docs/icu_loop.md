# ICU Loop Instruction

`Loop` is a queue-local control instruction that replays a contiguous window
of previously issued functional instructions. It complements `Repeat`, which
only replays the immediately preceding instruction.

## Semantics

`Loop(window_size, count, interval, address_stride)` executes `count`
additional rounds of the preceding `window_size` functional instructions.
The Loop issue cycle is also the issue cycle of the first replayed
instruction. `interval` is the distance, in cycles, between round starts and
must be at least `window_size`.

For a MEM queue, `address_stride` is multiplied by the replay-round index and
added to read and read-write destination addresses. Other queue types require
a zero stride. Loop windows cannot contain control instructions and cannot be
nested.

## 32-bit Encoding

| Bits | Field |
| --- | --- |
| 1:0 | opcode = 3 |
| 7:2 | window_size, 1..63 |
| 15:8 | count, 1..255 |
| 23:16 | interval, 1..255 |
| 31:24 | signed address_stride, -128..127 |

The ICU stores the replay window as local i-MEM PCs and fetches each replayed
instruction from i-MEM. Runtime therefore submits one Loop command instead of
expanding the repeated sequence on the host.
