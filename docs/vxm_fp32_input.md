# Complete FP32 fixed-port VXM inputs

`VxmLaneOperand::StreamFloat32()` reads all four little-endian IEEE-754
bytes from SR. It does not reinterpret the high two bytes as BF16. This
changes the input path only: Basic ALU precision is still selected separately,
and the special ALU LUT retains its existing FP16 FTZ boundaries.

The existing 32 logical byte streams suffice. The four-byte connections are
fixed, with no arbitrary stream selector or additional crossbar:

| Physical chain heads | FP32 LHS streams | FP32 RHS streams |
| --- | --- | --- |
| C0 / C2 | 0-3 | 4-7 |
| C4 / C6 | 8-11 | 12-15 |
| C8 / C10 | 16-19 | 20-23 |
| C12 / C14 | 24-27 | 28-31 |

These are logical VXM inputs; the existing hemisphere mux determines which
SR fabric supplies each two-byte group. FP16/BF16 keep their old bindings.
Depth-4/8 heads have disjoint buses. Adjacent depth-2 heads share an FP32
bus and cannot receive different values on it in the same cycle. A mixed
format instruction can also alias bytes (for example C0 FP32 LHS and BF16
RHS); software must not treat these as independent operands. The packed
generic lowering rejects that overlapping mixed-format combination.

Slice readiness/consumption includes all four bytes for each FP32 operand.
Both mirrored datapaths decode the same input-format selector. Native
FP32 outputs, chain registers and scalar accumulators are unchanged.

## Encoding

The ALU packet remains 96 bits. Existing operand values 0-7 are unchanged;
`StreamFloat32` is value 8. Control bits 52 and 53 extend the LHS and RHS
three-bit operand selectors respectively. Selectors 9-15 are rejected.
Bits 0-51 and the 32-bit immediate retain their old meanings, including
local scalar write at bit 51. Existing packets remain bit-identical.
New packets require the updated decoder; an old decoder is not compatible
with the extension. The ICU RUN_2D packet transports the complete control
word without a new instruction type or an increased instruction length.

## Software and validation

The compiler accepts `stream_f32` only at the corresponding fixed port.
The optional module policy `ftlpu.vector_fp32_ingress=true` enables full
FP32 reads for the packed vector path, inclusive row scans and causal
pairwise coefficients. BF16 tensors and legacy policies are unchanged.
The transport planner reserves four SR streams and concurrent binary
inputs are placed on separate MEM ICU queues.

`vxm_fp32_input_test` tests compact encoding, low-mantissa preservation
through subtraction, mirrored heads, illegal selectors, legacy packet bits,
and Slice input requirements. Software's `vxm_fp32_input_runtime_test.py`
exercises two FP32 operands through MEM, raw ICU packets and CModel, using
differences that would all vanish under BF16 truncation. Causal-coefficient
regressions cover both legacy and FP32 inputs. The real-weight Qwen3.5 L
test remains on the ordinary compiler/runtime path.
