# MEM ICU N-D Stream Descriptor

## Purpose

`MEM_STREAM_ND` lets one MEM ICU entry describe a regular sequence of SRAM
reads or writes. It keeps cycle-exact scheduling while avoiding one queue entry
for every projection block.

The descriptor owns one native `MemInstruction` and this schedule:

```text
start_cycle
rank                                  // 1..3
count[3]
cycle_stride[3]
address_stride[3]
```

Dimension zero is innermost. At coordinate `i[d]`, the expanded transfer is:

```text
cycle   = start_cycle + sum(i[d] * cycle_stride[d])
address = native.address + sum(i[d] * address_stride[d])
```

The native opcode, stream register, direction, and write-tap behavior remain
unchanged. Only the SRAM address is induced.

## MEM ICU Execution

Each descriptor occupies one i-MEM/IQ entry. Once decoded, it moves into the
MEM ICU's active-descriptor calendar. The ICU keeps the three counters, chooses
the descriptor with the earliest next issue cycle, emits one native transfer,
and advances its odometer. Multiple descriptors can therefore interleave on a
single MEM queue without host-generated NOPs.

The CModel reports a static-schedule error when:

- rank is outside 1..3, a count is zero, or a cycle stride is zero;
- a higher dimension overlaps the complete span of lower dimensions;
- two active descriptors target the same MEM queue in the same cycle;
- a descriptor reaches the ICU after its absolute start cycle;
- legacy queue commands are mixed into an active N-D descriptor region.

## Hardware Mapping

A practical RTL implementation needs three small count/current registers,
three cycle strides, three signed address strides, one next-cycle register, and
one current-address register per active descriptor. A bounded next-issue
calendar or comparator selects the due descriptor. The target model should
eventually expose the supported rank, descriptor FIFO depth, and maximum
in-flight descriptor count.

This is a generic MEM access instruction. Projection names, tensor shapes, and
model-specific concepts remain in compiler IR and are not encoded in the ICU.

