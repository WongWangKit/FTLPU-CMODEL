# Selectable cross-superlane SXM transpose

The default remains `SxmTransposeArchitecture::LocalBlocks`: existing local
transpose plus block permute, with unchanged scheduling and data semantics.
Select the additional implementation when constructing the CModel:

```cpp
ftlpu::TspSliceSystem system(
    {}, ftlpu::SxmTransposeArchitecture::WideStreaming);
auto instruction = ftlpu::SxmInstruction::Transpose(
    {{16}, {17}}, {{18}, {19}});
system.icu().enqueue_sxm_transpose(instruction);
```

An isolated FU can similarly construct `SxmSlice(ports, WideStreaming)`.
Selection is fixed for the lifetime of the system. Reset clears live work but
does not change the architecture. No compiler lowering is changed by this work;
old local-block transpose programs must continue to select the default model.

## Contract

For LPU32, one instruction consumes one 32x32 tile, represented by two byte
streams (BF16/FP16 payload bits). Row r, superlane s arrives at issue+r+s.
The four input delays are 3/2/1/0. At issue+3+r, the aligned row is written to
SRAM, bank=(r+c)%32, address=r. Column c is read at issue+35+c. The four output
delays are 0/1/2/3; output superlane s is staged at issue+35+c+s. Fabric outputs
become visible after that cycle's commit. Output is A[r,c], placed in lane r
of output vector c. No subsequent Permute instruction is needed.

Both East and West stream directions are supported through the configured SXM
ports. Source and destination byte pairs must each be distinct and direction
uniform. The compact ICU encoding additionally requires consecutive pair IDs.
Input and output directions cannot change while jobs remain live. Other SXM
operations must wait for wide transpose drain. Shift/Distribute/Permute retain
the existing implementation when no wide transpose is active.

Minimum tile initiation interval is 32 cycles. The issuing program must schedule
that interval (e.g. RUN_2D with appropriate intervals); the current ICU does not
automatically retry a FU issue that violates it. Missing input on its scheduled
cycle throws, consistent with strict existing stream consumers. There is no
implicit elastic stall or downstream backpressure protocol. Output port
collisions are checked by the stream fabric.

Three or more back-to-back tiles use alternating SRAM buffers. The prior tile
finishes SRAM readout before the next tile reuses that buffer; delayed output
segments are already retained by the output FIFO.

## Storage and encoding

- Two 32x32x2-byte SRAM tiles: 4096 B.
- Input alignment: 48 elements / 96 B.
- Output re-skew: 48 elements / 96 B.
- Total data payload: **4288 B**, excluding tags/valid/last, controls and C++
  simulator object/container overhead. No extra SRAM read latency is modeled.

The implementation carries `last` and `vector_tag` with each original element
through the transpose; it does not invent a new uniform output-vector tag.
The functional model preserves arbitrary 16-bit patterns, without arithmetic.

Transpose RUN_2D uses previously invalid input-row selector **15** for the wide
whole-tile instruction, with two source and two destination streams. Its decoded
instruction uses the existing Transpose opcode and all-rows selectors, with
stream counts distinguishing it from local transpose. Existing packet encodings
remain unchanged; existing CModels predating this extension reject selector 15.
The wide command is rejected by the default LocalBlocks FU. The API is not a
promise that software compiler emission or the HTML viewer is already adapted.

## Validation

`sxm_wide_transpose_test` covers compact ICU encode/decode, system queue acceptance,
three successive tiles, both directions, gaps, every output byte and its exact
cycle, last/tag transport, empty cycles, reset, missing input and default-model
rejection. SRAM overwrite and FIFO bounds are checked inside the model.

Build and run the SXM regressions using the existing CMake build. Assertions for
SXM/ICU-SXM regression targets are enabled even in Release builds.
