> 历史设计记录：本文中的独立 2D/3D、STREAM_ND、slice/tile program 已移除，不是当前 ISA。当前实现见 [Macro ISA](icu_macro_scheduler.zh-CN.md)。旧测试产物需重新编译。

# SXM TILE_PROGRAM

`SXM_TILE_PROGRAM` combines one complete native SXM configuration with a
one-to-three-dimensional absolute-cycle iteration space. The configuration
retains the opcode, source and destination stream lists, row/tile selectors,
and the full 32-lane permutation map.

```text
cycle = start_cycle + sum(i[d] * cycle_stride[d])
```

Transpose and permute remain independent per-hemisphere ICU queues. A tile
program never serializes the two ports or changes their overlap. Typical
lowering uses one transpose program for a long token/block repeat and four
permute programs for the four recurring tile-row maps.

The current form does not induce fields inside the SXM instruction. Irregular
maps are represented by separate tile programs and may interleave through the
queue's next-issue calendar.

