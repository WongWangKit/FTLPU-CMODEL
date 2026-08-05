# ICU-driven Transformer FFN integration

This is the single maintained `256 x 32 x 64 x 32` edge-Transformer FFN
integration. The distributed ICU is its only replay-time functional-control
source.

The test runs in two passes:

1. The reference scheduler records every cycle-tagged MEM, MXM, VXM, and SXM
   instruction while executing the validated direct datapath.
2. The compiler pass splits that schedule by physical ICU endpoint, inserts
   queue-local NOPs, writes each local i-MEM, prefetches its IQ, and replays the
   same FFN without direct module issue calls.

It checks SwiGLU, static quantization, MEM layout, SXM transpose/permute, Down
MXM results, ICU underflow/completion, aggregate issue counts, and exact
cycle-by-cycle datapath activity against the reference pass.

Generated native test artifacts are written to `results/`:

- `icu_driven_edge_ffn_256x32x64x32_brief_report.md`;
- `icu_driven_edge_ffn_256x32x64x32_detailed_trace.txt`;
- `icu_driven_edge_ffn_256x32x64x32_gantt.html`.

The CMake target is `icu_driven_edge_transformer_ffn_test`.
