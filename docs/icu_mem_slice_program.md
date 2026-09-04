# MEM ICU Slice Program

## Purpose

`MEM_SLICE_PROGRAM` lets one MEM ICU FIFO entry describe a short program for
one physical SRAM slice. It combines MEM operations that traverse the same
one-to-three-dimensional launch domain, while preserving the exact issue cycle
and address of every native read or write.

The launch domain is shared:

```text
start_cycle
rank                                  // 1..3
count[3]
cycle_stride[3]
```

Each of the one through sixteen body entries contains:

```text
cycle_offset
address_stride[3]
native MemInstruction                 // read/write, address, SR, direction
```

For launch coordinate `i[d]` and body entry `b`, the expanded operation is:

```text
cycle(b, i) = start_cycle + cycle_offset[b]
              + sum(i[d] * cycle_stride[d])
address(b, i) = native[b].address
                + sum(i[d] * address_stride[b][d])
```

Body entries may mix reads and writes and may use different stream registers.
The shared launch counts and cycle strides are what make them one program.

## ICU Execution

The complete program occupies one i-MEM/IQ entry and receives one program
counter. The CModel validates it once, then creates queue-local active cursors
for its body entries. This models a MEM ICU microsequencer while retaining
cycle-exact arbitration and diagnostics.

The CModel rejects an empty or oversized body, an invalid N-D launch domain, an
address induction outside the native MEM address space, a missed issue cycle,
or two body/program cursors due on the same MEM queue in one cycle.

## Binary Prototype

The software validation encoding uses the `MSPG` magic, version 1, and the ICU
`Extended` opcode. Its extension payload is:

```text
header: magic, version, start_cycle, rank, body_count,
        count[0], cycle_stride[0], ... count[2], cycle_stride[2]
body:   cycle_offset, address_stride[0..2], native_lo, native_hi
```

The header is 11 32-bit words and each body entry is six words. Relocating one
program relocates the native base address in every body entry. This is a
variable-length software envelope; the final RTL encoding may split the body
store from the fixed FIFO descriptor.

## Hardware Direction

A practical MEM ICU can keep the launch counters once and store up to sixteen
compact body records in a local program RAM. Each body needs a next-issue
offset, three signed address strides, and a native MEM operation. A small
calendar selects the body due in the current cycle. The body limit is an ISA
revision limit, not a model- or projection-specific rule.

