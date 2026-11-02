# MXM WS / IS modes

MXM has two dataflows, each with two weight-input granularities. Granularity
is **per supercell column**, not the total across the MXM. For lpu32 there
are four supercell rows and eight lanes per supercell.

| Mode | Resident operand | Weight input per column per cycle |
|---|---|---|
| WS-vector | Weight buffer, followed by streamed activations | 1 vector |
| WS-block | Weight buffer, followed by streamed activations | 8 vectors |
| IS-vector | Activation buffer; weights are consumed in the compute pipeline | 1 vector |
| IS-block | Activation buffer; weights are consumed in the compute pipeline | 8 vectors |

WS uses `IW` followed by `Compute`. `MxmWeightLoadMode::Vector` selects a
single inner column; `Block` fills the supercell weight block. INT8 local
dequant and direct 16-bit input remain independent of this transfer choice.

IS uses `IsLoadActivation` and `IsStreamCompute`, with `MxmIsMode::Vector`
or `Block`. Both broadcast one activation vector across the active columns.
IS currently consumes INT8 weights with BF16 dequantization, retaining the
previous arithmetic. IS-vector serializes a column's eight output vectors
over eight cycles; it does not switch to a 16-cell serial reduction.

For a wave launched at L, row r and column c consume weights at:

- IS-block: `L + r + c`, eight vectors in parallel.
- IS-vector: `L + r + c + j`, vector j for j=0..7.

Input streams are `c*8+j` for IS-block and `c` for IS-vector (INT8 byte
planes). The minimum wave initiation intervals are 1 and 8 respectively.
The complete wave spans R+C-1 or R+C+8-2 cycles. The existing output
superlane skew follows completion. Accumulation, output casting and clearing
operate on the same eight-output segment per active column in both modes.

## Configuration

```json
"mxm": {
  "accum_contexts": 4,
  "is_columns": 4,
  "supported_modes": ["ws-vector", "ws-block", "is-vector", "is-block"]
}
```

`is_columns` is a build-time number of active physical columns. It must be
a positive divisor of the supercell row count (`topology.tiles_per_slice`):
four rows allow only 1, 2 or 4 columns. This is checked by CMake, C++ static
assertions and the compiler target parser. It does not change WS geometry.
Block input must fit `is_columns * lanes_per_tile`
east SR streams. Disabled columns consume no weights. Partial results stay
in ACC; they are not padded and streamed out.
Compiler and CModel must be built with the same target configuration.

## Packing partial IS results in ACC

`IsStreamCompute` already carries an ACC row address, an ACC segment offset
(`accumulator_column`), and the destination selection. With fewer IS columns
than superlanes, use `destination=Sram` and retain the results. Each column
adds its eight results to the selected segment. For C columns and R rows,
use segment offsets 0, C, ..., R-C: exactly R/C waves fill one output vector.
These controls are present in FU and raw macro
instructions; no tensor-level operation or implicit MEM copy is involved.

For four superlanes and two IS columns:

| Instruction | ACC destination | SR output |
|---|---|---|
| IS compute, row 0, segment 0, Sram | row 0 segments 0,1 | none |
| IS compute, row 0, segment 2, Sram | row 0 segments 2,3 | none |
| ACC_READ row 0, BF16, clear | read then clear all four segments | one complete 32-element vector |

The scheduler issues ACC_READ after the last contributing wave completes.
The output preserves the normal one-cycle superlane skew. A retained read
(`clear=false`) is also available. CModel rejects direct stream output from
a reduced-column IS instruction and rejects ACC_READ on an incomplete IS
row. This check does not insert a dynamic stall: instruction timing remains
the compiler's responsibility.

Current high-level IS FFN lowering still requires the full physical column
count. Reduced-column configurations are supported by the primitive path;
the high-level FFN layout rejects them until its output packing is adapted.

Linear1x16 is removed. Its old instruction bit value is now IS-vector;
software ABI generation 22 distinguishes these semantics and includes the
IS column count. Previously generated executables must be recompiled.
The former native4 dataflow is now IS-block by default.

The compiler high-level policy is `auto|ws|is`; IS policy selects the
existing block projection lowering. Primitive schedule/command IR supports
both IS granularities via `is_mode = "is-vector"` or `"is-block"` and
WS input via `weight_load_mode = "vector"` or `"block"`. Selecting a
hardware capability does not automatically change every model's lowering.

The streamed-weight numerical fixture runs both IS modes with multiple waves,
packs complete ACC rows, and checks every output against a bit-exact BF16
reference using nonuniform activations/weights. It also rejects premature
reads of incomplete rows. Existing
BF16 and weight-input tests cover the WS datapath; ICU/FU codec tests cover
the instruction fields. The former Linear1x16 comparison tests are retired.
