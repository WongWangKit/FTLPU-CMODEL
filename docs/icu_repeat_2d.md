# ICU Repeat2D Instruction

`Repeat2D` is a queue-local, blocking two-dimensional iterator for one
functional instruction. The immediately preceding instruction has already
issued as coordinate `(0, 0)`. Remaining points issue in outer-major,
inner-minor order.

```text
cycle(o, i) = base_cycle + o * outer_interval + i * inner_interval
delta(o, i) = o * outer_stride + i * inner_stride
```

Both counts include the base point. Only the local queue is blocked while the
descriptor is active.

## 96-bit Encoding

| Bits | Field |
| --- | --- |
| 1:0 | extended-control opcode in the Loop class |
| 11:2 | inner_count |
| 21:12 | outer_count |
| 37:22 | inner_interval |
| 53:38 | outer_interval |
| 69:54 | signed inner_stride |
| 85:70 | signed outer_stride |
| 87:86 | induction target |
| 91:88 | extended subtype = 1 |
| 95:92 | reserved |

Counts are `1..1023`, intervals are `1..65535`, and strides are
`-32768..32767`. The outer interval must be greater than the final issue
offset of one inner wave.

Typed targets are `None`, `MemAddress`, and `MxmWeightColumn`. MEM induction
updates the selected bank's local row address; MXM induction is valid only for
an `IW` weight column. Execution retains the previously decoded functional
instruction and does not reread i-MEM.
