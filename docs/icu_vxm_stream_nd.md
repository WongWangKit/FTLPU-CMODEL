# VXM_STREAM_ND

`VXM_STREAM_ND` carries one native 96-bit compact VXM packet and a one-to-three-
dimensional absolute-cycle iteration space in one ICU macro instruction. The
ICU latches the packet once and generates all launch points with N-D counters;
there is no configuration slot or CONFIG/RUN pairing state.

For coordinate `i[d]`, the ICU issues the selected compact packet at:

```text
cycle = start_cycle + sum(i[d] * cycle_stride[d])
```

`VXM_STREAM_ND` has no operand induction. A different opcode, immediate, stream
group, chain depth, output mode, or datapath repeat count requires a different
descriptor. The repeat count inside the compact packet is not expanded by
the ICU: it controls how long the Superlane Current Config Register executes
after each issue.

The CModel represents the macro instruction as `IcuVxmStreamNdSchedule` plus a
`VxmCompactInstruction`. It rejects overlapping issue points, missed absolute
cycles, operand strides, and mixed legacy commands while a descriptor is in
flight.
