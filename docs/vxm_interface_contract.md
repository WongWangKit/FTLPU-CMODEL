# Retained VXM interface contract

Phase one keeps the VXM that already interoperates with the `feat/cjh` ICU,
packet codec, and `TspSliceSystem`. It does not copy the later incompatible
VXM from `main`.

- The VXM has the configured number of lanes (16 in GroqLike, 8 in
  TransformerEval), 16 ALU IQs, and 16 pipeline stages.
- Each ALU has one finite, program-ordered ICU queue. The modeled initiation
  interval is one instruction per cycle when operands and output space are
  available.
- Inputs are selected from the 64 packed directional streams. Legacy VXM
  packets retain their exact encoding and imply east input/output. The
  extended 16-byte packet kind carries input and output hemisphere bits.
- Stream operands support int8, int32, FP16, and FP32. Outputs carry a stream
  base and one, two, or four bytes according to cast type, then return through
  the selected VXM-to-MEM hemisphere bridge.
- Add/subtract/multiply/divide, min/max, clamp, square, sqrt, exp, log, ReLU,
  pass and casts are native scalar-lane operations.
- Reciprocal is a composed `Divide(1,x)` operation; rsqrt is composed from
  `Sqrt` followed by `Divide(1,x)`.
- A single-instruction cross-lane max or sum reduction is not implemented in
  the retained VXM. Attention's three-pass softmax must therefore use an
  explicit reduction schedule/tree or a future special-ALU interface.

The packet exposes opcode, two operand kinds and 6-bit indexes, cast target,
output-valid and a 6-bit output stream. The extended packet additionally
exposes input/output hemisphere. Scale and zero-point fields remain model
metadata unless the selected opcode/cast explicitly consumes them.

The explicit gap for the next VXM integration is:

1. define native cross-lane max/sum reduction instructions, latency, result
   placement, and backpressure. The migrated f621374 schedule can still
   express temporal per-lane reductions explicitly.
