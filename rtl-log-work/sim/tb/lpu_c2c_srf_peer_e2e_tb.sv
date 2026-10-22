`timescale 1ns/1ps

// Two real SRF endpoints. No force, leaf writes, or TX metadata to RX routing.
module c2c_srf_peer_checker #(
  parameter integer LINK_LATENCY = 1,
  parameter integer FULL_SUITE = 1
) (output reg done, output integer single_span);
  localparam integer C = 16, SL = 4, ST = 32, SLOTS = 2;
  localparam integer SV = 2*C*SL*ST, IV = SV*SLOTS, BV = 2*SL*ST;
  // Exercise a nonzero slot: all unused slot bits must remain zero.
  localparam integer SLOT = 1;
  reg clk = 0;
  always #5 clk = ~clk;
  reg rst_n = 0, tx_valid = 0, cmd_valid = 0;
  reg [2:0] tx_stream = 0, cmd_stream = 0;
  reg [IV-1:0] source_inject_valid = '0;
  reg [IV*64-1:0] source_inject_data = '0;
  wire [SV-1:0] source_valid, destination_valid;
  wire [SV*64-1:0] source_data, destination_data;
  wire [IV-1:0] source_consume, destination_inject_valid;
  wire [IV*64-1:0] destination_inject_data;
  wire [2*C*SL-1:0] source_collision, destination_collision;
  wire [2*C*SL-1:0] source_invalid_consume, destination_invalid_consume;
  wire cmd_pop, ready_full, ready_empty;
  wire [1:0] ready_count;

  ftlpu_sr_hemisphere_fabric source_srf (
    .clk_i(clk), .rst_ni(rst_n),
    .boundary_valid_i({BV{1'b0}}), .boundary_data_i({BV*64{1'b0}}),
    .boundary_valid_o(), .boundary_data_o(),
    .inject_valid_i(source_inject_valid), .inject_data_i(source_inject_data),
    .consume_i(source_consume), .collision_o(source_collision),
    .invalid_consume_o(source_invalid_consume),
    .state_valid_o(source_valid), .state_data_o(source_data)
  );
  ftlpu_sr_hemisphere_fabric destination_srf (
    .clk_i(clk), .rst_ni(rst_n),
    .boundary_valid_i({BV{1'b0}}), .boundary_data_i({BV*64{1'b0}}),
    .boundary_valid_o(), .boundary_data_o(),
    .inject_valid_i(destination_inject_valid), .inject_data_i(destination_inject_data),
    .consume_i({IV{1'b0}}), .collision_o(destination_collision),
    .invalid_consume_o(destination_invalid_consume),
    .state_valid_o(destination_valid), .state_data_o(destination_data)
  );
  lpu_c2c_srf_peer_e2e #(
    .C2C_CONSUMER_SLOT(SLOT), .C2C_PRODUCER_SLOT(SLOT),
    .LINK_LATENCY(LINK_LATENCY)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .tx_issue_valid_i(tx_valid),
    .tx_stream_index_i(tx_stream), .rx_cmd_valid_i(cmd_valid),
    .rx_cmd_stream_index_i(cmd_stream), .rx_cmd_pop_o(cmd_pop),
    .source_state_valid_i(source_valid), .source_state_data_i(source_data),
    .source_consume_o(source_consume),
    .destination_inject_valid_o(destination_inject_valid),
    .destination_inject_data_o(destination_inject_data),
    .rx_ready_full_o(ready_full), .rx_ready_empty_o(ready_empty),
    .rx_ready_count_o(ready_count)
  );

  integer cycle_no = 0;
  always @(posedge clk) cycle_no = cycle_no + 1;
  integer count, seed, offset, mode;
  // Cycle labels refer to numbered rising edges: S samples TX tile0; G
  // captures tile3/completion into the completed FIFO; R is post-edge peer
  // visibility. Pair is combinational at PAIR_VISIBLE, sampled/popped at
  // P=PAIR_VISIBLE+1. SRF tile0 commits at that same edge, so D0=P (post-NBA).
  // Thus with a waiting Receive: G=S+3, R=G+L, P=R+2, Dt=P+t.
  integer send_at[0:3], send_stream[0:3], receive_stream[0:3];
  integer s_cycle[0:3], g_cycle[0:3], r_cycle[0:3], p_visible[0:3];
  integer d_cycle[0:3][0:3];
  integer queue_id[0:7], q_read, q_write, head;
  integer previous_gather, previous_peer, previous_pair;
  integer gathered, received, paired, delivered, wait_cycles, first_ready;
  integer overlap_seen, command_wait, vector_wait;
  reg held_command;
  reg [2:0] held_stream;
  reg [SV-1:0] expected_source, expected_destination, next_destination;
  reg [SV*64-1:0] expected_source_data, expected_destination_data, next_destination_data;
  reg [IV-1:0] expected_consume, expected_inject, previous_inject;
  reg [IV*64-1:0] expected_inject_data, previous_inject_data;

  function automatic integer state_index(input integer dir, col, tile, stream);
    state_index = ((dir*C+col)*SL+tile)*ST+stream;
  endfunction
  function automatic integer slot_index(input integer dir, col, tile, slot, stream);
    slot_index = (((dir*C+col)*SL+tile)*SLOTS+slot)*ST+stream;
  endfunction
  function automatic [255:0] payload(input integer id);
    integer b;
    begin
      for (b=0; b<32; b=b+1)
        payload[b*8 +: 8] = (seed + id*41 + b*3) & 255;
    end
  endfunction
  function automatic [63:0] segment(input integer id, tile);
    reg [255:0] v;
    begin v=payload(id); segment=v[tile*64 +: 64]; end
  endfunction
  task automatic check(input reg condition, input string reason);
    if (condition !== 1'b1)
      $fatal(1, "TEST_FAIL L=%0d cycle=%0d offset=%0d %s",
             LINK_LATENCY, cycle_no, offset, reason);
  endtask

  // Every candidate below is stable before its sampling edge. Native SRF
  // injection prepares current source state one edge ahead of Gather capture.
  task automatic tick(input integer n);
    integer id, t, col, st, ix, ox, gid, rid, pid, age, active;
    reg [3:0] tile_mask;
    begin
      offset=n;
      @(negedge clk);
      source_inject_valid='0; source_inject_data='0;
      expected_source='0; expected_source_data='0; expected_consume='0;
      tile_mask=0;
      for (id=0; id<count; id=id+1) begin
        t=n-send_at[id];
        if (t>=0 && t<4) begin
          ix=slot_index(0,13,t,0,send_stream[id]);
          source_inject_valid[ix]=1;
          source_inject_data[ix*64 +: 64]=segment(id,t);
          ix=state_index(0,13,t,send_stream[id]);
          expected_source[ix]=1;
          expected_source_data[ix*64 +: 64]=segment(id,t);
          expected_consume[slot_index(0,13,t,SLOT,send_stream[id])]=1;
          tile_mask[t]=1;
        end
      end
      // Independent state model: West propagation, then current local inject.
      next_destination='0; next_destination_data='0;
      for (col=0; col<C-1; col=col+1)
        for (t=0; t<SL; t=t+1)
          for (st=0; st<ST; st=st+1) begin
            ix=state_index(1,col,t,st); ox=state_index(1,col+1,t,st);
            next_destination[ix]=expected_destination[ox];
            next_destination_data[ix*64 +: 64]=expected_destination_data[ox*64 +: 64];
          end
      for (t=0; t<SL; t=t+1)
        for (st=0; st<ST; st=st+1) begin
          ix=slot_index(1,13,t,SLOT,st); ox=state_index(1,13,t,st);
          if (previous_inject[ix]) begin
            check(!next_destination[ox], "reference schedule collision");
            next_destination[ox]=1;
            next_destination_data[ox*64 +: 64]=previous_inject_data[ix*64 +: 64];
          end
        end
      @(posedge clk);
      // Check settled pre-edge candidates, not delta-cycle transients.
      check(source_collision==='0 && destination_collision==='0, "collision before commit");
      check(source_invalid_consume==='0, "invalid source consume before commit");
      #1;
      if (previous_gather>=0) begin
        g_cycle[previous_gather]=cycle_no; gathered=gathered+1;
      end
      if (previous_pair>=0) begin q_read=q_read+1; head=head+1; paired=paired+1; end
      if (previous_peer>=0) begin queue_id[q_write]=previous_peer; q_write=q_write+1; end
      expected_destination=next_destination; expected_destination_data=next_destination_data;
      check(source_valid===expected_source, "source state / passive propagation mismatch");
      check(destination_valid===expected_destination, "destination state valid/routing mismatch");
      for (ix=0; ix<SV; ix=ix+1) begin
        if (expected_source[ix]) check(source_data[ix*64 +: 64]===expected_source_data[ix*64 +: 64], "source payload");
        if (expected_destination[ix]) check(destination_data[ix*64 +: 64]===expected_destination_data[ix*64 +: 64], "destination payload/byte packing");
      end
      for (id=0; id<count; id=id+1)
        for (t=0; t<4; t=t+1)
          if (p_visible[id]>=0 && cycle_no==p_visible[id]+t+1) begin
            ix=state_index(1,13,t,receive_stream[id]);
            check(destination_valid[ix] && destination_data[ix*64 +: 64]===segment(id,t), "destination segment commit");
            d_cycle[id][t]=cycle_no; delivered=delivered+1;
          end
      if (count==4 && mode==0 && d_cycle[3][0]==cycle_no) begin
        check(d_cycle[2][1]==cycle_no && d_cycle[1][2]==cycle_no && d_cycle[0][3]==cycle_no, "RX overlap commits");
        overlap_seen=1;
        $display("RX_OVERLAP L=%0d cycle=%0d SL0/W7=V3.t0 SL1/W0=V2.t1 SL2/W2=V1.t2 SL3/W5=V0.t3", LINK_LATENCY, cycle_no);
      end

      tx_valid=0; tx_stream=0;
      for (id=0; id<count; id=id+1) if (n==send_at[id]) begin
        tx_valid=1; tx_stream=send_stream[id]; s_cycle[id]=cycle_no+1;
      end
      if (q_write>q_read && first_ready<0) first_ready=cycle_no;
      cmd_valid=(head<count) && (mode!=2 || (first_ready>=0 && cycle_no>=first_ready+3));
      cmd_stream=(head<count) ? receive_stream[head] : 0;
      if (held_command) check(cmd_valid && cmd_stream==held_stream, "command head changed before pop");
      #1;
      check(source_consume===expected_consume, "TX full consume bus mapping");
      check(dut.tx_tile_valid===tile_mask && dut.tx_tile_consume===tile_mask, "TX diagonal tile selection/consume");
      for (id=0; id<count; id=id+1) begin
        t=n-send_at[id];
        if (t>=0 && t<4) begin
          check(dut.tx_tile_data[t*64 +: 64]===segment(id,t), "TX tile data");
          check(dut.u_tx_adapter.tile_stream_idx_o[t*5 +: 5]===send_stream[id][4:0], "per-tile TX selector");
        end
      end
      gid=-1; rid=-1; pid=-1;
      for (id=0; id<count; id=id+1) begin
        if (n==send_at[id]+3) gid=id;
        if (g_cycle[id]>=0 && cycle_no==g_cycle[id]+LINK_LATENCY) rid=id;
      end
      check(dut.u_tx.gather_completed_valid===(gid>=0), "Gather completion timing/bubble");
      if (gid>=0) begin
        check(dut.u_tx.gather_completed_payload===payload(gid), "Gather vector integrity");
      end
      check(dut.peer_rx_valid===(rid>=0), "Peer latency/order/bubble");
      if (rid>=0) begin
        check(dut.peer_rx_payload===payload(rid), "Peer vector payload");
        r_cycle[rid]=cycle_no; received=received+1;
      end
      check(ready_count==q_write-q_read && ready_empty===(q_write==q_read) && ready_full===(q_write-q_read==2), "RX-ready FIFO occupancy");
      if (q_write>q_read) check(dut.u_rx.ready_payload===payload(queue_id[q_read]), "RX FIFO payload retention/order");
      if (cmd_valid && q_write>q_read) pid=queue_id[q_read];
      check(cmd_pop===(pid>=0) && dut.u_rx.ready_pop===(pid>=0) && dut.u_rx.pair_valid===(pid>=0), "atomic Receive/vector pairing");
      if (pid>=0) begin
        check(pid==head && p_visible[pid]<0, "pair duplicate or order");
        check(dut.u_rx.pair_stream===receive_stream[pid][4:0], "Receive stream independent of TX");
        check(dut.u_rx.pair_payload===payload(pid), "pair payload");
        p_visible[pid]=cycle_no;
      end
      if (cmd_valid && q_write==q_read) command_wait=command_wait+1;
      if (!cmd_valid && q_write>q_read) vector_wait=vector_wait+1;
      held_command=cmd_valid && !cmd_pop; held_stream=cmd_stream;

      expected_inject='0; expected_inject_data='0; active=0;
      for (id=0; id<count; id=id+1) begin
        age=cycle_no-p_visible[id];
        if (p_visible[id]>=0 && age>=0 && age<4) begin
          ix=slot_index(1,13,age,SLOT,receive_stream[id]);
          expected_inject[ix]=1; expected_inject_data[ix*64 +: 64]=segment(id,age);
          active=active+1;
        end
      end
      check(destination_inject_valid===expected_inject, "RX full inject valid bus/direction/slot");
      check(destination_inject_data===expected_inject_data, "RX full inject data bus");
      check(source_collision==='0 && destination_collision==='0, "collision in legal schedule");
      previous_inject=expected_inject; previous_inject_data=expected_inject_data;
      previous_gather=gid; previous_peer=rid; previous_pair=pid;
    end
  endtask

  task automatic reset_case;
    integer id,t;
    begin
      @(negedge clk);
      rst_n=0; tx_valid=0; cmd_valid=0; source_inject_valid='0; source_inject_data='0;
      repeat (2) @(posedge clk);
      #2;
      check(source_valid==='0 && destination_valid==='0, "SRF reset state");
      check(source_consume==='0 && destination_inject_valid==='0 && !cmd_pop, "reset interface valid");
      check({dut.u_tx_adapter.selector_valid_d1_q,dut.u_tx_adapter.selector_valid_d2_q,dut.u_tx_adapter.selector_valid_d3_q}===3'b0, "TX selector reset");
      check({dut.u_tx.u_gather.stage0_valid_q,dut.u_tx.u_gather.stage1_valid_q,dut.u_tx.u_gather.stage2_valid_q}===3'b0, "Gather reset");
      check(dut.u_tx.fifo_count==0 && dut.u_tx.u_transport.valid_q==='0 && ready_count==0, "FIFO/transport reset");
      check(dut.u_rx.u_replay.deferred_valid_q==='0 && !dut.u_rx.pair_valid, "pair/replay reset");
      expected_destination='0; expected_destination_data='0;
      previous_inject='0; previous_inject_data='0;
      previous_gather=-1; previous_peer=-1; previous_pair=-1;
      q_read=0; q_write=0; head=0; first_ready=-1;
      gathered=0; received=0; paired=0; delivered=0;
      command_wait=0; vector_wait=0; overlap_seen=0; held_command=0;
      for (id=0; id<4; id=id+1) begin
        s_cycle[id]=-1; g_cycle[id]=-1; r_cycle[id]=-1; p_visible[id]=-1;
        for (t=0;t<4;t=t+1) d_cycle[id][t]=-1;
      end
      @(negedge clk); rst_n=1;
    end
  endtask

  task automatic run_case;
    integer n,id,t,gap;
    begin
      reset_case(); n=0;
      // Bounded event-driven completion, followed by full SRF drain. No
      // assumed end-to-end delay is used to decide when data should match.
      while ((delivered<count*4 || expected_destination!='0 || n<2) && n<80) begin
        tick(n); n=n+1;
      end
      check(delivered==count*4 && gathered==count && received==count && paired==count, "completion count / timeout");
      check(expected_destination==='0 && ready_count==0 && previous_peer<0, "drain/no stale delivery");
      for (id=0; id<count; id=id+1) begin
        check(g_cycle[id]==s_cycle[id]+3, "S to G latency");
        check(r_cycle[id]==g_cycle[id]+LINK_LATENCY, "G to R latency");
        if (mode!=2) check(p_visible[id]==r_cycle[id]+1, "R to pair visibility latency");
        for (t=0;t<4;t=t+1) check(d_cycle[id][t]==p_visible[id]+1+t, "Pair/Replay/SRF capture latency");
        $display("CYCLE L=%0d V%0d E%0d->W%0d S=%0d G=%0d R=%0d P=%0d D0=%0d D1=%0d D2=%0d D3=%0d PAIR_VISIBLE=%0d",
          LINK_LATENCY,id,send_stream[id],receive_stream[id],s_cycle[id],g_cycle[id],r_cycle[id],p_visible[id]+1,
          d_cycle[id][0],d_cycle[id][1],d_cycle[id][2],d_cycle[id][3],p_visible[id]);
        if (id>0 && mode!=2) begin
          gap=send_at[id]-send_at[id-1];
          check(g_cycle[id]-g_cycle[id-1]==gap && r_cycle[id]-r_cycle[id-1]==gap && p_visible[id]-p_visible[id-1]==gap, "II/bubble preservation");
        end
      end
    end
  endtask

  initial begin
    done=0; single_span=0; offset=-1;
    count=1; seed=17; mode=0; send_at[0]=0; send_stream[0]=3; receive_stream[0]=6;
    run_case(); single_span=d_cycle[0][0]-s_cycle[0];
    if (FULL_SUITE) begin
      $display("C2C_SRF_PEER_STREAM_REMAP PASS");
      seed=49; send_stream[0]=2; receive_stream[0]=7; run_case();
      $display("C2C_SRF_PEER_SINGLE PASS");
      count=4; seed=83;
      send_at[0]=0; send_at[1]=1; send_at[2]=2; send_at[3]=3;
      send_stream[0]=3; send_stream[1]=6; send_stream[2]=7; send_stream[3]=1;
      receive_stream[0]=5; receive_stream[1]=2; receive_stream[2]=0; receive_stream[3]=7;
      run_case(); check(overlap_seen==1,"missing four-tile overlap");
      $display("C2C_SRF_PEER_II1 PASS");
      $display("C2C_SRF_PEER_MULTI_STREAM PASS");
      $display("C2C_SRF_PEER_RX_OVERLAP PASS");
      count=1; seed=117; mode=1; send_at[0]=3; send_stream[0]=3; receive_stream[0]=6;
      run_case(); check(command_wait>=3,"command-first wait coverage");
      $display("C2C_SRF_PEER_COMMAND_FIRST PASS");
      seed=151; mode=2; send_at[0]=0;
      run_case(); check(vector_wait>=3,"vector-first retention coverage");
      $display("C2C_SRF_PEER_VECTOR_FIRST PASS");
      count=4; seed=187; mode=3;
      send_at[0]=0; send_at[1]=1; send_at[2]=3; send_at[3]=4;
      send_stream[0]=3; send_stream[1]=6; send_stream[2]=7; send_stream[3]=1;
      receive_stream[0]=5; receive_stream[1]=2; receive_stream[2]=0; receive_stream[3]=7;
      run_case();
      $display("C2C_SRF_PEER_BUBBLE PASS");
      count=1; seed=211; mode=0; send_at[0]=0;
      reset_case(); tick(0); tick(1);
      check(dut.u_tx.u_gather.stage0_valid_q===1'b1,"reset in-flight coverage");
      // Abandon old vector; unique new payload and destination expose leakage.
      seed=239; send_at[0]=2; send_stream[0]=2; receive_stream[0]=4;
      run_case();
      $display("C2C_SRF_PEER_RESET PASS");
      $display("C2C_SRF_PEER_TX_CONSUME PASS");
      $display("C2C_SRF_PEER_RX_INJECT PASS");
      $display("C2C_SRF_PEER_COLLISION_FREE PASS");
      $display("C2C_SRF_PEER_CYCLE PASS");
    end else $display("C2C_SRF_PEER_LINK_LATENCY_3 PASS");
    done=1;
  end
endmodule

module lpu_c2c_srf_peer_e2e_tb;
  wire done1,done3;
  wire [31:0] span1,span3;
  c2c_srf_peer_checker main_check(.done(done1),.single_span(span1));
  c2c_srf_peer_checker #(.LINK_LATENCY(3),.FULL_SUITE(0)) latency_check(.done(done3),.single_span(span3));
  initial begin
    wait(done1 && done3);
    if (span3!==span1+2) $fatal(1,"TEST_FAIL LINK_LATENCY shift");
    $display("C2C_SRF_PEER_E2E TEST_PASS");
    $finish;
  end
  initial begin #20000; $fatal(1,"TEST_FAIL watchdog timeout"); end
endmodule
