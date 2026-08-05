# Distributed ICU Fetch Timing

- Status: PASS
- Total modeled cycles: 12
- Compiler-selected common start cycle: 4
- Local fetch bandwidth/latency: 1 instruction per cycle / 1 cycle
- Data MEM/SR bandwidth used by instruction fetch: 0
- Queue underflows: 0

| Queue       | Width | i-MEM depth | IQ depth | Program | Fetched | Functional issues |
| :---------- | ----: | -----------: | -------: | ------: | ------: | ----------------: |
| MXM.Load    |   32b |          16 |        4 |       4 |       4 |                 2 |
| MXM.Compute |   32b |          16 |        4 |       3 |       3 |                 5 |
| MEM         |   96b |          16 |        4 |       4 |       4 |                 4 |
| VXM         |   96b |          16 |        4 |       3 |       3 |                 2 |
| SXM         |   96b |          16 |        4 |       3 |       3 |                 2 |

The first four cycles prefetch each local program. At cycle 4, all five queues can issue independently as one logical VLIW. NOP, Repeat and Sync remain queue-local and do not consume data-stream bandwidth.
