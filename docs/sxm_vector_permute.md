# SXM single-vector Permute

On the existing Permute ICU queue, one source and one destination byte stream
select vector mode. All SLs take part. For each output lane `d`,
`out[d] = in[map[d]]`. The map must contain each physical input lane once.
The payload byte, last flag and vector tag follow the selected source. Each
byte plane of a wider element requires a separate issue with the same map.
Legacy block Permute still selects and releases captured transpose blocks.

For S superlanes of L byte lanes each, an issue at C uses this schedule:

| Stage | Cycles | Operation |
|---|---|---|
| Capture | C through C+S−1 | SL s captures its L-byte segment at C+s. |
| Route | C+S through C+2S−1 | Tap local lanes, then tap successively more distant SLs. Two registered paths forward L bytes per link per direction per cycle. Each output lane latches its requested source at the matching hop. |
| Output | C+2S through C+3S−1 | Stage the result segment for SL s at C+2S+s. Consumers see it after the SR commit. |

Capture, routing and output each hold one vector job. Independent jobs can
issue every S cycles, with independent maps and stream IDs. For LPU32 the
initiation interval is 4 and output staging is at offsets 8 through 11.
Changing SXM operation mode requires the vector pipe to drain. There is no
MEM scratch allocation or Transpose prerequisite.

The raw-macro FU template and active context keep their original widths.
A six-bit source base of 63 identifies vector mode (a block range cannot
start at 63). The legacy row field then stores source bits [3:0], the tile
field stores [5:4], and remaining tile bits must be zero. The destination
and lane-map fields retain their meanings. The layout/timing above is an
FTLPU CModel design choice.

Run `unit.sxm.vector_permute` in the CModel test build. The software
repository's `compiler/tests/sxm_vector_permute_runtime_test.py` exercises
the LPU32 Schedule → raw-macro → MEM/SXM/MEM path and saves the executed
ICU programs and runtime pipeline.
