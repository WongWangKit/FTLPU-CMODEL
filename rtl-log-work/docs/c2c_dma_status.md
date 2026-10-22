# C2C / DMA Current Status and Continuation Guide
## Purpose

This document records the current architectural and verification status of the
C2C peer datapath and DMA Store baseline.

It is not a final architecture specification, production protocol
specification, or PHY specification.
## Current checkpoint

The current Git checkpoint includes:
- C2C peer SRF-to-SRF normal path;
- C2C vector credit and serialization-delay abstraction;
- MEM-SRF DMA Store vector-sink baseline;
- single-lane DMA Store context;
- final regression runner.

The unified regression entry is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_final_regression.ps1
```

The current baseline result is:

```text
ALL KEY REGRESSIONS PASS
```

## C2C peer datapath

The implemented C2C peer path is:

```text
Source East sreg13
  -> TX SRF adapter
  -> 4 x 64-bit TX gather
  -> Completed FIFO
  -> credit / serialization abstraction
  -> peer transport
  -> RX-ready FIFO
  -> independent Receive pairing
  -> RX Replay
  -> Destination West sreg13
```

Implemented responsibilities are:

- source stream selection and SRF consume;
- four 64-bit segments gathered into one 256-bit vector;
- payload-only vector staging after TX gather;
- independent Send and Receive stream routing;
- RX-ready buffering and FIFO-order pairing with Receive commands;
- 256-bit vector replay into four 64-bit West SRF segments;
- destination SRF injection;
- vector-level credit tracking and serialization occupancy/delay abstraction;
- dual-SRF end-to-end verification.

This is an architectural peer datapath closure. It is not production C2C IP,
a real link protocol, a SerDes implementation, or a PHY implementation.

## Frozen C2C semantics

- ordinary SRF is static scheduled and readyless;
- Send stream and Receive stream are independent;
- for example, `Send E3` plus `Receive W6` transfers the source `E3` payload
  to destination `W6`; it does not imply `E3 -> W3`;
- the boundary after TX gather is payload-only;
- `vector_tag` is not part of the baseline;
- credit controls only Completed FIFO to launch admission;
- credit decrements at `launch_fire`;
- credit returns at RX-ready successful pairing / `ready_pop`;
- credit does not create ordinary SRF backpressure;
- serializer behavior is only a vector-level occupancy/delay abstraction.

## C2C not implemented

The following are NOT IMPLEMENTED / OPEN:

- real PHY and real SerDes;
- CDC;
- framing;
- CRC/FEC;
- retry and recovery;
- production link protocol;
- final Peer/DMA coexistence topology.

## DMA Store data-plane

The implemented Store-only baseline is:

```text
MEM group12 Read
  -> ordinary East SRF
  -> East sreg13
  -> TX adapter
  -> gather
  -> Completed FIFO
  -> abstract 256-bit vector sink
```

The abstract sink interface is:

- `vector_valid`;
- `vector_data[255:0]`;
- `vector_accept`.

The sink is an external-storage-facing abstraction only. Sink acceptance is
not external write completion.

The Store baseline has verified:

- basic single-vector transfer;
- source stream selection;
- sink hold behavior;
- SRF consume behavior;
- reset behavior;
- back-to-back II=1 overlap;
- 256-bit payload ordering.

This does not represent complete 8-lane DMA throughput.

## DMA Store context

`rtl/dma_native/dma_store_lane_context.sv` is a single-lane decoded-semantic
context primitive.

It stores:

- active descriptor state;
- fixed logical DMA lane identity;
- direction;
- external base address;
- `vector_count_minus_1`;
- `stride_bytes`.

It supports:

- accept;
- hold while active;
- explicit retirement lifecycle hook;
- same-cycle retire and replacement.

`retire_i` is a temporary lifecycle abstraction. It is not final external
write completion, gather completion, FIFO pop, or `vector_accept`.

## DMA frozen and known semantics

- a DMA descriptor represents a multi-vector transfer task;
- descriptor address semantics are `base + i * stride_bytes`;
- external base address is a 64-bit byte address;
- physical vector size is 32 bytes / 256 bits;
- `completed_vectors` advances on true external write completion;
- descriptor completion requires all vectors to be externally completed;
- the current Store baseline permits at most one outstanding write per lane;
- DMA descriptor does not directly issue ordinary MEM Read;
- ordinary MEM Read and DMA TX issue remain independent static controls;
- logical DMA lane `0..7` is not the ordinary East stream `E0..E7` namespace;
- `vector_tag` remains removed.

## DMA OPEN / TBD

### Progress

- exact `next_vector_index` increment event;
- formal request-launch acceptance event.

### Descriptor / ICU

- final 192-bit descriptor codec;
- DMA queue mapping;
- 416-bit ICU codec;
- descriptor FIFO;
- 8-lane dispatcher.

### External storage

- request valid/ready;
- request ID;
- response/completion identity;
- burst behavior;
- error handling;
- arbitration;
- outstanding management.

### Schedule association

- descriptor vector `i` to MEM/SRF/TX gathered-vector binding;
- compiler/static-schedule coordination contract.

### Remaining DMA work

- notification / Sync semantics;
- DMA Load;
- final endpoint integration.

## Why progress RTL is not ready

A progress/address module is NOT yet ready to implement because:

1. the `next_vector_index` increment event is not frozen;
2. the external request acceptance event is not frozen;
3. request/completion identity is not frozen;
4. descriptor-vector to MEM/SRF/TX-vector association remains partial.

Do not implement progress RTL by inventing these contracts.

## Regression coverage

The final runner covers:

| Group | Final marker |
| --- | --- |
| SRF | `VMODEL_SRF_PORT TEST_PASS` |
| MEM-SRF | `VMODEL_MEM_SRF_INTEGRATION TEST_PASS` |
| MEM-SRF-SXM | `VMODEL_MEM_SRF_SXM_ROUNDTRIP TEST_PASS` |
| C2C Credit | `C2C_VECTOR_CREDIT_SERIALIZER TEST_PASS` |
| C2C TX | `C2C_TX_PEER_PATH TEST_PASS` |
| C2C RX Pair | `C2C_RX_ISSUE_PAIR TEST_PASS` |
| C2C RX SRF | `C2C_RX_SRF_INTEGRATION TEST_PASS` |
| C2C E2E | `C2C_SRF_PEER_E2E TEST_PASS` |
| DMA Store | `DMA_STORE_VECTOR_SINK TEST_PASS` |
| DMA Context | `DMA_STORE_LANE_CONTEXT TEST_PASS` |

## Recommended continuation

1. Freeze `next_vector_index` and request-launch semantics.
2. Freeze external request and completion contract.
3. Freeze descriptor-to-MEM/TX static-schedule association.
4. Implement single-lane progress/address only after steps 1-3 close.
5. Extend Store external request/completion afterward.
6. Expand descriptor FIFO and 8-lane handling later.
7. Implement DMA Load after Store semantics close.
8. Integrate ICU codec/queues only after codec semantics are frozen.

## Do not redesign

- SRF static schedule and state ownership;
- MEM-SRF basic path;
- MEM-SRF-SXM round-trip;
- C2C gather;
- Completed FIFO;
- independent Send/Receive semantics;
- payload-only peer boundary;
- no `vector_tag`;
- credit return at RX `ready_pop`;
- RX Replay;
- dual-SRF C2C end-to-end path;
- DMA Store MEM/SRF/vector-sink baseline;
- single-lane DMA context lifecycle.
