# VXM Lane and Superlane datapath tests

`lpu_vxm_lane_chain_tb` fixes `LANES=1` and verifies one Lane containing 16
physical ALUs. Its fixed nearest-neighbor datapath is re-partitioned solely by
the global `chain_length` field.

The test loads the same Tile RTL with chain lengths 2, 4, and 8 and checks the
corresponding physical tail masks:

- length 2: stages 1, 3, 5, 7, 9, 11, 13, and 15;
- length 4: stages 3, 7, 11, and 15;
- length 8: stages 7 and 15.

The boundary sweep uses BYPASS. A separate length-4 arithmetic case proves the
actual inter-ALU path by calculating `1+2=3`, `3+original(1)=4`,
`4+auxiliary(2)=6`, and `6+immediate(1)=7`; it also checks that the original
and auxiliary metadata reach every physical chain tail unchanged. A final
length-4 sequence stores the first result in the tail Feedback registers,
reloads only the head source encoding, consumes Feedback on the next
execution, and checks that the Feedback token is cleared.

`lpu_vxm_superlane_lockstep_tb` fixes `LANES=8`, applies one shared length-4
instruction wave, and gives every Lane a different input. It checks that all
eight Lanes retire each chain tail in the same cycle and that every result and
Original value remains in its own Lane. The arithmetic is used only as a path
signature; ALU numerical coverage belongs to the per-ALU regressions.

Both tests drive and observe only public `lpu_vxm_tile_execution` ports.
