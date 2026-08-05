# Pipelined Edge Transformer FFN Datapath

The compiled distributed ICU program drives the same two resident Front MXMs, lane-banked VXM, MEM/SXM transpose path and two Down MXMs as the direct reference schedule; all functional controls originate in local i-MEM/IQ endpoints.

- Status: PASS
- Shape: tokens=`256`, hidden=`32`, SwiGLU intermediate=`64`, output=`32`
- VXM chain depth/chains per lane: 8/2
- Weight-buffer reuse rounds before block switch: 2
- Flow: `Gate/Up MXM -> ACC/cast -> SR -> lane-banked VXM Input Buffer -> VXM SwiGLU -> FP16-to-INT8 -> SR/MEM -> SXM byte transpose/permute -> Down MXM -> FP16 cast`
- Total cycles: 825
- Checked values (SwiGLU/layout/Down): 16384/32768/8192
- Activation MEM vector writes/reads: 512/4096
- SXM Transpose/Permute instructions: 64/112
- Down FP32 pair-merge (two 32-wide K halves), FP16 cast element merges: 8192
- Down weight-buffer compute uses: 512
- Front MXM Compute issues/bubbles: 512/0
- Down MXM Compute issues/bubbles: 256/0
- Down inter-burst input-wait cycles: 193
- VXM peak ALU slots (active/all configured Superlanes): 512/512
- VXM quantized outputs/cycle (peak/configured/physical): 64/64/256
- MXM compute/background weight-load overlap cycles: 11
- Gate/Up MXM with VXM overlap: 252 cycles (75.00% of VXM-active cycles)
- SXM with Down MXM overlap: 44 cycles (11.11% of Down-MXM-active cycles)

- ICU control: distributed local i-MEM replay
- ICU prefetch cycles before system cycle 0: 16
- ICU active queues: 54
- ICU functional events/program words/fetched words/issues: 4840/6194/6194/4840
- ICU underflowed queues: 0

| Resource             |  Active |    Time |   Whole |
| :------------------- | ------: | ------: | ------: |
| **MXM total**        |  54.23% |  90.18% |  48.91% |
| Front MXMs (2)       |  95.05% |  66.18% |  62.91% |
| Down MXMs (2)        |  72.73% |  48.00% |  34.91% |
| VXM (all 4 SL)       |  77.68% |  40.73% |  31.64% |
| VXM configured total |  77.68% |  40.73% |  31.64% |
| VXM q configured     |  95.52% |  32.48% |  31.03% |
| VXM q physical       |  23.88% |  32.48% |   7.76% |
| MEM ports            |  10.01% |  70.42% |   7.05% |
| SXM INT8 plane       |  68.75% |  15.52% |  10.67% |

`Utilization over whole test = Utilization while active x Time utilization`. Time utilization counts cycles with at least one operation on that resource.

`SXM INT8 single-byte plane` means each INT8 element occupies one byte plane in the transpose storage; FP16 uses two byte planes (low/high byte). It is the SXM data-width mode, not a separate hardware block. The total and phase MXM rows use all four configured MXMs as capacity. The VXM total row measures every configured Superlane; the chain count and depth above come from the active VXM program, so `VXM q configured` measures only configured chain outputs; `VXM q physical` measures all 8 fixed quantizers per lane.
