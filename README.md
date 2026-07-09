# FTLPU-CMODEL

Cycle-level C++ model experiments for FTLPU-style stream and memory behavior.

## Project Layout

- `include/ftlpu/core/`: common topology, stream, and instruction-pipeline primitives.
- `include/ftlpu/mem/`: MEM slice and 20-row tile-array models.
- `include/ftlpu/mxm/`: MXM supercell, array, control, and GEMM execution models.
- `include/ftlpu/vxm/`: VXM ALU and lane-level issue-queue models.
- `tests/<module>/`: unit tests grouped by subsystem, with `tests/integration/` for cross-subsystem tests.
- `examples/<module>/`: trace/demo executables grouped by subsystem.

## Current Models

- `ftlpu::StreamRegister<T>`: one-cycle stream register with `valid` represented by `std::optional`.
- `ftlpu::MemSlice<T>`: memory slice that accepts MEM-style instructions such as `Read a,s`, then emits one stream word per cycle onto stream `s`.
- `ftlpu::NorthboundInstructionPipeline`: MEM instruction pipeline that moves one row north every cycle and prints per-cycle activity.
- `ftlpu::TileArrayModel`: 44 independent slice-column instruction queues plus 20 tiles/superlanes. Each tile has 16 lanes, and each lane has 32 eastbound streams and 32 westbound streams, with per-cycle trace logging.
- `ftlpu::MxmSupercell`: one 16x16 MXM supercell with `IW`/`LW` weight loading. Each `LW` cycle consumes 16 streams across 16 lanes, loading the full 16x16 weight matrix in one cycle.
- `ftlpu::MxmArray`: full 20x20 MXM supercell array. The current array-level model implements per-supercell weight loading.
- `ftlpu::MxmControlSlice`: south-to-north MXM control pipeline. Each tile controls one row of 20 supercells; each row has its own local weight input. `IW(col)` loads that row's current weight input into the selected supercell column, and `Compute(cycles)` opens a compute window.
- `ftlpu::MxmGemmEngine`: 320x320 int8 GEMM execution model. Activation streams enter tile rows with one-cycle south-to-north skew; each valid supercell consumes one 16-byte activation vector, performs 16 int8 dot products against its 16 weight columns, accumulates int32 partial sums, and forwards the activation vector east.
- `ftlpu::VxmAlu`: first VXM building block. It models 16-lane pointwise ALU behavior for arithmetic, compare/select, nonlinear functions, accumulation, and int8 quantize/dequantize operations.
- `ftlpu::VxmLane`: one scalar VXM lane with 16 ALUs and one instruction queue per ALU. Each ALU instruction chooses operands from lane input streams or registered outputs from other ALUs. The first lane program is SwiGLU for an FFN-style path: streams 0-3 form the raw int32 `gate`, streams 4-7 form the raw int32 `up`, cast instructions convert raw stream operands to fp32, multiply instructions apply dequant scales, sigmoid is built from primitive ALU instructions (`neg`, `exp`, `add 1`, `1/x`), then the lane computes `up * gate * sigmoid(gate)`, multiplies by the reciprocal output scale, adds the output zero point, and casts to int8.
- `ftlpu::VxmSuperlane`: 16 `VxmLane` instances running in lockstep. Each cycle accepts one 16-lane gate vector and one 16-lane up vector, then emits a 16-byte int8 vector when all lanes produce aligned outputs.
- `ftlpu::VxmSlice`: 20 south-to-north superlanes/tiles. VXM instructions enter at the south edge and advance one tile per cycle; data for tile rows is supplied with the same south-to-north one-cycle skew. The slice test runs a 320x320 SwiGLU matrix path over 20 tiles x 16 lanes.
- `ftlpu::TspSliceSystem`: fixed local topology with VXM attached to the west side of MEM and MXM attached to the east side. VXM can consume any of the 64 lane streams selected by its instruction operands, and ALU instructions can optionally emit an output stream selected by the instruction. The MEM/VXM integration test uses west streams 32-39 for gate/up input and east stream 0 for the result, but those stream IDs are instruction choices rather than hardwired system behavior.
- `ftlpu::topology`: TSP-style grid constants and address helpers for 20 tile rows, 44 slice columns, 4-slice groups, 12 stream register columns, and the modeled half of MEM.

## Hardware Parameters

The topology model uses these parameters:

- 20 tile rows.
- 44 slice columns.
- 4 slices per group, for 11 groups.
- 12 stream register columns: one boundary column for each 4-slice quad plus the far boundary.
- 16 lanes per tile/superlane.
- 32 one-byte streams flowing east and 32 one-byte streams flowing west in each lane.
- MEM can store at most 16 bytes per tile per cycle from the same stream across 16 lanes into SRAM.
- MEM can load at most 16 bytes per tile per cycle from SRAM into the same stream across 16 lanes.
- MEM activity is driven by instructions as they arrive at each tile; tests do not directly request a MEM action on an arbitrary tile.
- MXM is modeled as a 320x320 MAC array made from a 20x20 grid of 16x16 supercells.
- VXM starts with a single-lane experiment: 16 ALUs, 16 issue queues, 8 byte input streams, and registered ALU-to-ALU dependencies. Public TSP descriptions identify VXM as 16 ALUs for pointwise arithmetic such as add, multiply, and tanh, with 1-8 inputs and 1-4 outputs; this cmodel implements the public arithmetic/nonlinear/quantization subset first.
- The current `TileArrayModel` SRAM is byte-addressable for simplicity: each modeled tile/slice SRAM has 8192 one-byte entries.
- 88 public SRAM blocks across both hemispheres.
- 44 modeled SRAM blocks for the current single-hemisphere model.
- 320-byte physical vector width.
- 8192 vector words per SRAM block.
- 2.5 MiB per SRAM block, 110 MiB modeled SRAM, 220 MiB public total SRAM.

The 45 SRF, 64 stream, 320-byte vector, 88 SRAM block, 55 TiB/s MEM bandwidth, and 220 MiB SRAM figures are public TSP parameters described in Groq TSP literature, including "TSP: Accelerating BERT Using the Tensor Streaming Processor". The 88 SRAM blocks are described as distributed evenly between hemispheres, so this model starts with one 44-block hemisphere.

## Build and Test

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

To generate a per-cycle stream-register trace:

```powershell
.\build-vs2019\Debug\tile_array_trace_demo.exe tile_array_trace.log
```

To run the west-in/east-out vector roundtrip scenario:

```powershell
.\build-vs2019\Debug\vector_roundtrip_demo.exe vector_roundtrip.log
```

To generate the per-cycle 20x20 MXM load trace:

```powershell
.\build-vs2019\Debug\mxm_control_trace_demo.exe mxm_control_trace.log
```

To generate the full GEMM trace with separate MEM and MXM logs:

```powershell
.\build-vs2019\Debug\gemm_320_trace_demo.exe gemm_320_mem.log gemm_320_mxm.log
```

The GEMM trace runs `A[320,320] x B[320,320] -> C[320,320]` with int8 inputs and int32 accumulation. The demo first stores the full activation matrix and the full weight matrix in MEM, then loads weights from MEM into MXM, then streams activations from MEM into MXM.

To generate a VXM lane trace that prints every ALU state each cycle. The demo feeds one SwiGLU token per cycle, uses cast plus multiply instructions for quantization/dequantization, drains the pipeline, and logs operand sources and values:

```powershell
.\build-vs2019\Debug\vxm_lane_trace_demo.exe vxm_lane_trace.log
```

## GEMM Data Layout

The current GEMM demo uses one byte-addressable SRAM space per tile and MEM slice.

Activation matrix `A[m,k]`:

- Stored in MEM slice `16`, stream `E0`, so activation reads can overlap with weight reads from slices `0..15`.
- Address is `k * 16 + lane`.
- Tile row selects `m / 16`, lane selects `m % 16`.
- Mapping: `A[tile*16 + lane, k] -> slice 16, tile, addr k*16 + lane`.

Weight matrix `B[k,n]`:

- Stored completely in MEM before MXM loading starts.
- Uses MEM slices `0..15`; each slice holds one stream/column within a 16-column block.
- Column block `cb = n / 16` is stored at `weight_base + cb*16`, where `weight_base = 6144` in the demo.
- Mapping: `B[tile*16 + lane, cb*16 + stream] -> slice stream, tile, addr weight_base + cb*16 + lane`.
- During loading, all column blocks are read from resident MEM through a continuous pipeline: `IW(0)`, `IW(1)`, ..., `IW(19)` are issued one per cycle, while the matching 16-stream weight data for each column block arrives at the MXM east-side handoff point in the same cycle.

Result stream:

- During compute, each MXM supercell column block emits one 16-lane int32 result vector per valid cycle.
- Column block `0..19` maps to result MEM tile `0..19`.
- Each int32 vector is split little-endian over westward streams `W0..W3` at `sreg11`, then written into contiguous MEM slices `40..43`.

Compute:

- `Compute(320)` is issued, not `Compute(6400)`.
- Activation data is read from MEM and enters tile rows with one-cycle south-to-north skew. The demo starts activation/compute one cycle after tile 0 has loaded all 20 weight column blocks, while later tiles are still receiving their final weight blocks.
- Each supercell consumes one 16-byte activation vector per valid cycle, computes 16 int32 partial sums for its 16 weight columns, and forwards the activation vector to the next supercell to the east.
- MXM logs print a 20x20 state matrix: `L` for loaded/idle, `C` for valid compute, and `.` after that cell has no more valid work.
