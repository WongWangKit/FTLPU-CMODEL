# RAW Macro Level 2 shared input contract

`shared_input.hpp` is the single host-side definition of the five Level 2 MEM
cases. It constructs every command as a `MemInstruction` and obtains the
formal 64-bit machine word only through `isa::encode_mem_instruction()`.

The CModel trace test exports `shared_input_manifest.json` beside its JSONL
trace. The manifest contains those exact words, issue cycles, locations,
streams, directions, SRAM preload bytes, and SR-boundary write stimulus. An
RTL harness can consume that generated manifest instead of copying private
encodings or recreating data patterns.

The contract deliberately contains only native Read and Write commands. It
does not use bit 30, a private three-cycle path, ready/busy flow control,
cycle offsets, or response queues.
