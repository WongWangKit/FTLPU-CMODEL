# ICU local VMODEL baseline

## Scope

This baseline freezes the local RTL files used for the ICU instruction-path
alignment work. It is a disk-snapshot baseline: the local VMODEL working tree
is authoritative even when a file is modified or untracked. GitHub `main` is
not substituted for any file listed here.

Captured on 2026-11-15 before CModel alignment changes.

## Repository state

| Item | Value |
|---|---|
| Local root | `D:\LPU\FTLPU-VMODEL-main` |
| Branch | `rtl-handshake-naming` |
| HEAD | `2cbaa104090c6df099894559794c0f981d4aafda` |
| Modified entries | 116 |
| Deleted entries | 10 |
| Untracked entries | 55 |

The ProgramLoad implementation used by this work is mainly the untracked
replacement family under `rtl/program_load/lpu_program_load_*` and
`lpu_program_distribute_*`. The older tracked ProgramLoad family is deleted in
the working tree. Consequently, HEAD alone cannot reproduce this baseline.

## Actual filelist and integration

`sim/filelist_program_load_system.f` explicitly includes the current files in
this order:

1. shared packages and `rtl/queue/lpu_queue_fifo.sv`;
2. QCTL pool/controller/fabric;
3. ProgramLoad controller, config FIFO, instruction DMA, reassembler, word
   FIFO and load path;
4. distribution elastic stage, scope distributor, relay, relay chain and
   distribution system;
5. MEM local iMEM/fetch/Macro frontend/bank queue;
6. MXM local iMEM/fetch/three Macro frontends/three timed queues/coordinator;
7. SXM and VXM subsystem boundaries;
8. `lpu_fu_subsystem_cluster` and `lpu_top`.

Top-level integration is real, not inferred from filenames:

- `rtl/lpu_top.sv:276-341` instantiates `lpu_program_load_system` and connects
  its distribution record interface.
- `rtl/lpu_top.sv:342-430` instantiates and connects
  `lpu_fu_subsystem_cluster`.
- `rtl/core/lpu_fu_subsystem_cluster.sv:286-315` instantiates the distribution
  system; MEM, MXM, SXM and VXM subsystem boundaries are instantiated at
  `:386-396`, `:473-483`, `:429-439` and `:552-560` respectively.
- `rtl/program_load/lpu_program_load_system.sv:183-238` currently validates
  descriptors for MEM and the three MXM streams. VXM/SXM kinds exist in the
  transport encoding but are rejected at this integrated descriptor boundary.
- `rtl/vxm/lpu_vxm_subsystem.sv:5-7,30-31` and
  `rtl/sxm/lpu_sxm_subsystem.sv:5-7,32-33` retain Program endpoints with
  `ready=0`; their standalone Macro frontend sources are not integrated into
  this top-level load path.

The common RTL Golden scope is therefore **MEM and MXM ProgramLoad, local iMEM,
Macro frontend and timed issue boundary**. VXM/SXM CModel functionality is
retained but classified as CModel-only or RTL Not Integrated.

## Key-file SHA-256

| File | SHA-256 |
|---|---|
| `rtl/lpu_top.sv` | `1F035D0AA04E7B7E01AA2E0648700C3C13CF56CF46A74A871956F76380E70210` |
| `rtl/core/lpu_fu_subsystem_cluster.sv` | `B15002FBCA31E1AF8945B902024799D49B1AB5554172530B26C0632CD56F2BD8` |
| `rtl/program_load/lpu_program_load_system.sv` | `C21BB3B8D3573135B4E1B23F938E91E04A634B4324E7AD4B4F2C170D0FFE4AAF` |
| `rtl/program_load/lpu_program_load_path.sv` | `E3CF60C6C7FE8DA464D707573D890B8358AFE059B91ACC481F97A9D002290EC9` |
| `rtl/program_load/lpu_instruction_dma.sv` | `D013FD7C9A12B609D31C42936DA68A5A053F380437CC5F0582937FEE87C0B5F5` |
| `rtl/program_load/lpu_program_load_reassembler.sv` | `1248C0C29DF7ADED23FD84D0E46D3D74CB3FFE029DDA1EA6F768193EB28527EE` |
| `rtl/program_load/lpu_program_distribute_system.sv` | `62B7458A298591B0DD8C2916D285F9FB32608DE6A093BC56298B87DB89C177C0` |
| `rtl/mem/lpu_mem_subsystem.sv` | `DB9D35A6FF92EC6BA534586F7010EBCB04ADD80EBD2719079C89FC68C0E1BA5C` |
| `rtl/mxm/lpu_mxm_subsystem.sv` | `C8481393F5679E93FFB1448E41C7DF7139325C352B2BED3F8D87A549F9D18132` |
| `rtl/vxm/lpu_vxm_subsystem.sv` | `5832FE2FBDB108A65EF39F57F97151BA7277D4EFC8104E7583EDF47EDB26FC4C` |
| `rtl/sxm/lpu_sxm_subsystem.sv` | `001719222955C47BC909ADC843DD7ED014737A003A6A877476D734152ECEF9BC` |
| `sim/filelist_program_load_system.f` | `852F3AFB89C205CCE7767CD1213CD62C98C2C19DEF68156C6B82DE685F896255` |

## Verification boundary and executed RTL evidence

The local machine has ModelSim SE-64 10.1c at
`D:\modelsim\win64`.  The VMODEL tree remained read-only during all
compiles: the post-run Git status is still 116 modified, 10 deleted and 55
untracked entries, with no ModelSim output added to that tree.

The following tests were compiled directly from this frozen local source and
executed with all libraries, logs and traces under
`D:\LPU\FTLPU-CMODEL-clock\out\icu_rtl_alignment\rtl_golden`:

| RTL scope | Result |
|---|---|
| `lpu_queue_fifo_tb` | PASS |
| independent FIFO RTL/CModel trace, 12 cycles and 17 fields | PASS, exact match |
| `lpu_icu_queue_cycle_tb` legacy NOP/Repeat/MEM stride | PASS |
| Program config FIFO | PASS, 5 checks |
| Program reassembler | PASS, 7 checks |
| Program distribution elastic stage | PASS, 3 emitted records |
| seven-stage ProgramLoad relay chain | PASS, 80 checks |
| MEM Macro frontend | PASS, 2 scenarios |
| MXM Load/Dequant/Compute Macro frontends | PASS |

The complete `sim/filelist_frontend_cluster_top.f` was also attempted.
ModelSim 10.1c cannot parse several newer SystemVerilog constructs in the
full RTL core (for example parameter-port `localparam` declarations), so the
integrated ProgramLoad-to-cluster test did not compile with this 2012 tool.
This is recorded as a simulator-version limitation, not as a DUT test failure.
Consequently the executed cycle-by-cycle cross-model Golden is complete for
the common FIFO primitive, while full-cluster cycle equivalence remains
PARTIAL.
