# VXM physical-ALU verification

The primary ALU regression is organized by physical position rather than by
opcode.  Each `lpu_vxm_aluN_tb` runs the shared execution-stage checker with
`PHYSICAL_STAGE=N`, so local instruction decoding and operand routing are part
of the DUT behavior under test.

| Physical ALUs | Local queues | Legal operations |
| --- | --- | --- |
| 0, 2, 4, 6, 8, 10, 12, 14 | Q0/Q2/Q4/Q6 | six Basic operations |
| 1, 5, 9, 13 | Q1/Q5 | six Basic operations plus EXP |
| 3, 7, 11, 15 | Q3/Q7 | six Basic operations plus RECIP and RSQRT |

For every physical position, the checker covers:

- all six Basic operations in FP16, BF16, and FP32;
- every legal position-specific source encoding;
- chain lengths 2, 4, and 8, including head/internal behavior;
- legal Special operations for that position and representative edge cases;
- reserved formats, chain lengths, source encodings, and illegal Special ops;
- stalls caused by missing selected data and reset during an in-flight op;
- public result data, metadata, handshake, and fault outputs only.

Run the position regressions and collect per-position VCS coverage with:

```bash
bash scripts/vcs_vxm_alu_positions.sh
```

The arithmetic-core regressions remain useful lower-level tests, but this
position suite is the primary acceptance test for an individual VXM ALU.
