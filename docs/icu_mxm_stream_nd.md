# MXM ICU N-D Stream Descriptor

## Purpose

`MXM_STREAM_ND` lets one MXM ICU queue entry describe a regular set of
weight-load, compute, accumulator-read, or dequant issue points. The descriptor
does not encode a projection or tensor shape. It wraps one native MXM
instruction with a cycle-exact affine schedule:

```text
start_cycle
rank                                  // 1..3
count[3]
cycle_stride[3]
operand_stride[3]
induction_target
```

At coordinate `i[d]`:

```text
cycle         = start_cycle + sum(i[d] * cycle_stride[d])
operand_delta =               sum(i[d] * operand_stride[d])
```

The induction target is typed by queue and native opcode:

| Queue/native opcode | Induced field |
| --- | --- |
| MXM load / `IW` | `weight_column` |
| MXM compute / `Compute` or `AccumulatorRead` | `accumulator_address` |
| MXM dequant | None |

All remaining native fields are invariant. In particular, stream destination,
accumulator clear, and BF16 output conversion survive expansion unchanged.

## MXM ICU Execution

The public enqueue paths are `enqueue_mxm_load_stream_nd`,
`enqueue_mxm_compute_stream_nd`, and `enqueue_mxm_dequant_stream_nd`. Each
descriptor occupies one i-MEM/IQ entry and then moves into the queue's active
descriptor calendar. The calendar selects the earliest next issue cycle,
applies the current operand delta, emits one native instruction, and advances
the nested counters.

Dimension zero is the innermost hardware counter. Dimensions must be ordered so
each higher-dimension stride is greater than the complete time span of all
lower dimensions. Separate descriptors may still interleave; the per-queue
calendar arbitrates their next issue cycles.

The CModel rejects an invalid rank/count/stride, an opcode/induction mismatch,
operand induction on dequant, a missed absolute issue cycle, or two descriptors
issuing on the same queue in one cycle.

## Hardware Mapping

A direct RTL implementation can share the same descriptor front end across the
three MXM ICUs while keeping FU-specific native payload decoders. Each active
descriptor needs three count/current registers, three cycle strides, three
signed operand strides, an induction selector, a next-cycle register, and the
native payload. A bounded comparator or timing wheel selects the due descriptor.

The software extension-word encoding is a semantic prototype, not a frozen wire
format. The eventual target configuration should expose maximum rank,
descriptor FIFO depth, and maximum active descriptors per MXM queue.

