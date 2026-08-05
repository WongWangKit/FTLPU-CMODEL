# Distributed ICU timing test

`icu_test.cpp` verifies the distributed local-i-MEM frontend independently of
the functional datapaths. The report scenario covers MEM, MXM Load, MXM
Compute, VXM, and SXM queues with parameterized instruction widths and finite
IQs.

Running `icu_test` regenerates:

- `results/distributed_icu_fetch_timing_brief_report.md`
- `results/distributed_icu_fetch_timing_detailed_trace.txt`
- `results/distributed_icu_fetch_timing_gantt.html`

The detailed trace and Gantt chart distinguish local i-MEM read start,
one-cycle read completion into the IQ, and queue issue/control activity. No
instruction event uses data MEM or a data Stream Register.
