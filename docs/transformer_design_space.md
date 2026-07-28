# Transformer datapath phase-one design space

The reproducible source of the numbers in this note is
`transformer_design_space`.  It evaluates:

- `seq_len=128`, `hidden=576`, `ffn_dim=1536`;
- `head_dim=64`, `kv_hidden=192`, `decode_m=1`;
- square `16`, `32`, `64`, and `320` MXM design points;
- FP16 physical weight traffic at 128 bytes/cycle/MXM;
- two concurrently loading MXMs per hemisphere sharing 32 streams.

Run:

```text
cmake --build build-codex --target transformer_design_space --config Debug
build-codex\Debug\transformer_design_space.exe
```

## Main observations

| GEMM | Array | M/N/K blocks | Tail M/N/K | Approx. PE fill |
|---|---:|---:|---:|---:|
| FFN gate/up prefill, 128x1536x576 | 16 | 8/96/36 | 0/0/0 | 100% |
| same | 32 | 4/48/18 | 0/0/0 | 100% |
| same | 64 | 2/24/9 | 0/0/0 | 100% |
| same | 320 | 1/5/2 | 192/64/64 | 34.56% |
| QK, 128x128x64 | 16 | 8/8/4 | 0/0/0 | 100% |
| same | 32 | 4/4/2 | 0/0/0 | 100% |
| same | 64 | 2/2/1 | 0/0/0 | 100% |
| same | 320 | 1/1/1 | 192/192/256 | 3.20% |
| P×V, 128x64x128 | 32 | 4/2/4 | 0/0/0 | 100% |
| FFN decode, 1x1536x576 | 16 | 1/96/36 | 15/0/0 | 6.25% |
| same | 32 | 1/48/18 | 31/0/0 | 3.12% |
| same | 64 | 1/24/9 | 63/0/0 | 1.56% |
| same | 320 | 1/5/2 | 319/64/64 | 0.27% |

The dimensions happen to divide 16/32/64 well, so this table does **not**
establish 32x32 as a final optimum.  Decode is dominated by the `M=1` tail
for every unpartitioned square array. Smaller independently schedulable arrays,
row partitioning, or batching are therefore separate design points worth
evaluating.

For one FP16 square weight tile, the modeled bytes/load cycles are:

| Array | FP16 bytes | Cycles at 128 B/cycle |
|---:|---:|---:|
| 16 | 512 | 4 |
| 32 | 2,048 | 16 |
| 64 | 8,192 | 64 |
| 320 | 204,800 | 1,600 |

Two MXMs request `2 × 16 = 32` streams, exactly one hemisphere's configured
capacity. Four simultaneous loaders on the same hemisphere would request 64
streams and have a 32-stream deficit. Four chip-wide MXMs avoid that conflict
only when placed two west/two east as configured.

## SRAM and tail contract

SRAM capacity is an input, not a result of array width:

| Config | Vector row | Banks | Rows/bank | Bytes/MEM slice |
|---|---:|---:|---:|---:|
| GroqLike | 320 B | 2 | 4,096 | 2,621,440 (2.5 MiB) |
| TransformerEval | 32 B | 2 | 40,960 | 2,621,440 (2.5 MiB) |

Phase one uses planner-inserted zero padding. Every emitted tile carries
`valid_rows`, `valid_cols`, and `valid_k`; input producers zero-fill invalid
lanes, and result stores suppress rows/columns outside the valid extents.
This is simple and deterministic but spends the padded MAC, stream, and SRAM
traffic reported by the tool.

All cycle figures here are analytical CModel scheduling quantities. They are
not claims about final RTL timing or product performance.
