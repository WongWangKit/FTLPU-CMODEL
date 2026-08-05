# VXM compiler first flow

The first implementation deliberately separates four meanings:

```text
operator_kernels.hpp
        |
        v
KernelBuilder -> Kernel IR
        |
        v
lowering.hpp -> Schedule IR (phase, 2/4/8 chain, C0-C3 operation)
        |
        +-> stream_plan.hpp (absolute Stream Register byte indices and cycles)
        |
        v
codegen.hpp -> 96-bit compact instruction packets
        |
        v
VxmSliceCModelAdapter
        +-> Slice south instruction issue
        +-> external Stream data interface
        +-> intermediate/final output writeback
```

## File responsibilities

- `kernel_ir.hpp`: mathematical values, tensor types and dependency graph.
- `kernel_builder.hpp`: checked API for constructing Kernel IR.
- `operator_kernels.hpp`: RMSNorm, Softmax and SwiGLU mathematical graphs.
- `schedule_ir.hpp`: VXM phases, chain depths, operands and ALU placement.
- `lowering.hpp`: operator-specific rules from Kernel IR to Schedule IR.
- `stream_plan.hpp`: converts internal chain ports to fixed, absolute Stream
  Register byte indices for the global static scheduler.
- `codegen.hpp`: checks the decoded lane configuration, then encodes it into
  the compact packet transported by the Slice.
- `external_data.hpp`: interface to the modeled producer/consumer outside VXM
  and a host-side value store for early integration tests.
- `cmodel_adapter.hpp`: issues compact packets at the Slice south edge,
  injects Stream data for the configured lanes, changes each Superlane's phase at its
  systolic offset, and captures intermediate/final outputs.
- `report.hpp`: writes a compact summary, a complete cycle/configuration log,
  and an interactive Superlane/ALU Gantt chart from the actual adapter trace.
- `print.hpp`: prints Kernel IR, Schedule IR and final compiler output for
  inspection.
- `compiler.hpp`: one include for the complete flow.

## Current baseline constraints

- RMSNorm and Softmax lower one row per lane. The configured Superlane lanes can
  execute sixteen rows in lockstep.
- Phases are conservatively serialized. Cross-phase overlap is a later
  scheduling optimization.
- Fixed Stream ports transfer FP16 values. Basic and special ALUs may use
  wider internal arithmetic according to the scheduled precision, then cast
  results back to the FP16 physical interface.
- Different ALU instruction channels may issue together; one channel issues at
  most one compact packet per cycle. A packet reaches Superlane `i` after `i`
  additional Slice cycles and is decoded locally there.
- The external data interface does not model MEM/MXM internals. A configurable
  fixed latency is inserted before each phase and missed statically scheduled
  output/input events are reported as errors instead of causing backpressure.
- A scalar reused by a later phase is represented explicitly by
  `VxmLocalScalarLoad`; it is not treated as an implicit crossbar connection.

The unified compiler test is `tests/vxm/compiler/compiler_test.cpp`. It covers
Kernel IR, dependency lowering, 2/4/8-chain placement, same-depth and
2->4/4->8/2->8 Feedback, compact instruction generation, Slice execution and
report generation. Generated reports are kept in
`tests/vxm/compiler/results/`.
