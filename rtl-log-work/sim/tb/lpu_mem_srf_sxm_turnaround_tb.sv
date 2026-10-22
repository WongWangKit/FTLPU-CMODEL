`timescale 1ns/1ps
module lpu_mem_srf_sxm_turnaround_tb;
  import lpu_pkg::*;
  localparam integer MEM_SLICES=16, GROUPS=4, COLUMNS=16, SUPERLANES=4, STREAMS=32, BITS=64;
  logic clk_i,rst_ni;
  logic [MEM_SLICES*2-1:0] vmodel_mem_issue_valid_i,native_mem_issue_valid_o,mem_command_fault_o;
  logic [MEM_SLICES*2*47-1:0] vmodel_mem_issue_instruction_i;
  logic [MEM_SLICES*2*32-1:0] native_mem_issue_o;
  logic vmodel_transpose_valid_i,vmodel_permute_valid_i;
  logic [415:0] vmodel_transpose_instruction_i,vmodel_permute_instruction_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i;
  logic [2*SUPERLANES*STREAMS*BITS-1:0] boundary_data_i;
  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS*BITS-1:0] state_data_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o,srf_invalid_consume_o;
  logic [MEM_SLICES*8-1:0] mem_producer_valid_o,mem_producer_direction_o,mem_internal_collision_o;
  logic [MEM_SLICES*8*BITS-1:0] mem_producer_data_o;
  logic [MEM_SLICES*8*5-1:0] mem_producer_stream_o;
  logic [MEM_SLICES*8*4-1:0] mem_producer_boundary_o;
  logic [(GROUPS+1)*256-1:0] mem_boundary_consume_o;
  logic [MEM_SLICES*2-1:0] mem_bank_fault_valid_o;
  logic mem_fault_valid_o,mem_busy_o,sxm_fault_valid_o,sxm_permute_phase_fault_o,sxm_permute_selector_fault_o,sxm_permute_buffer_not_ready_o,sxm_busy_o,sxm_command_fault_o;
  logic [SUPERLANES-1:0] sxm_transpose_input_invalid_o,sxm_transpose_buffer_full_o;
  integer errors,tile,stream,lane,group,local_slice,w,cap_cycle[0:3],buffer_cycle[0:3],perm_cycle[0:3],west_cycle[0:3];
  lpu_mem_srf_sxm_turnaround_integration dut(.*);
  always #5 clk_i=~clk_i;
  function automatic [46:0] mp(input [2:0] op,input west,input [4:0] s,input [14:0] row); begin mp='0;mp[2:0]=op;mp[8:3]={west,s};mp[30:15]=row;end endfunction
  function automatic [415:0] sp(input [1:0] op,input sw,input [4:0] sb,input dw,input [4:0] db,input [2:0] ot,input integer ph);
    integer i,d,src; begin sp='0;sp[1:0]=op;sp[6+:5]=16;sp[11+:5]=16;sp[SXM_OUTPUT_TILE_LSB+:SXM_OUTPUT_TILE_WIDTH]=ot;
      for(i=0;i<16;i=i+1) begin sp[16+i*6+:5]=sb+i;sp[16+i*6+5]=sw;sp[112+i*6+:5]=db+i;sp[112+i*6+5]=dw;end
      for(i=0;i<32;i=i+1) begin d=i/8;src=(ph+4-d)%4;sp[240+i*5+:5]=src*8+i%8;end end endfunction
  function automatic [63:0] pat(input integer t,input integer s); begin pat='0;for(lane=0;lane<8;lane=lane+1)pat[lane*8+:8]=(t<<6)|(s<<2)|lane;end endfunction
  function automatic [63:0] ref_t(input integer t,input integer os); integer r,p,ss; begin ref_t='0;r=os/2;p=os%2;for(lane=0;lane<8;lane=lane+1)begin ss=2*lane+p;ref_t[lane*8+:8]=(t<<6)|(ss<<2)|r;end end endfunction
  function automatic integer bi(input integer d,input integer t,input integer s);bi=(d*SUPERLANES+t)*STREAMS+s;endfunction
  function automatic integer si(input integer d,input integer c,input integer t,input integer s);si=((d*COLUMNS+c)*SUPERLANES+t)*STREAMS+s;endfunction
  task automatic clear;begin vmodel_mem_issue_valid_i='0;vmodel_mem_issue_instruction_i='0;vmodel_transpose_valid_i=0;vmodel_transpose_instruction_i='0;vmodel_permute_valid_i=0;vmodel_permute_instruction_i='0;boundary_valid_i='0;boundary_data_i='0;end endtask
  task automatic step;begin @(posedge clk_i);#1;end endtask
  task automatic ck(input bit ok,input [8*60-1:0] s);begin if(!ok)begin $display("ERROR %0s",s);errors=errors+1;end end endtask
  task automatic reset;begin @(negedge clk_i);clear();rst_ni=0;@(negedge clk_i);rst_ni=1;step();end endtask
  task automatic drive_tile(input integer t);integer x;begin for(x=0;x<16;x=x+1)begin boundary_valid_i[bi(0,t,x)]=1;boundary_data_i[bi(0,t,x)*BITS+:BITS]=pat(t,x);end end endtask
  task automatic group_cmd(input integer g,input [2:0] op);integer s,b;begin for(local_slice=0;local_slice<4;local_slice=local_slice+1)begin s=g*4+local_slice;b=s*2;vmodel_mem_issue_valid_i[b]=1;vmodel_mem_issue_instruction_i[b*47+:47]=mp(op,0,s,0);end end endtask
  task automatic preload;begin @(negedge clk_i);clear();drive_tile(0);step();for(group=0;group<4;group=group+1)begin @(negedge clk_i);clear();if(group<3)drive_tile(group+1);group_cmd(group,3'd1);step();end repeat(4)begin @(negedge clk_i);clear();step();end end endtask
  task automatic reads;begin for(group=0;group<4;group=group+1)begin @(negedge clk_i);clear();group_cmd(group,3'd0);step();for(local_slice=0;local_slice<4;local_slice=local_slice+1)begin integer s,pidx;s=group*4+local_slice;pidx=s*8;ck(mem_producer_valid_o[pidx]&&mem_producer_data_o[pidx*BITS+:BITS]===pat(0,s)&&mem_producer_direction_o[pidx]==0&&mem_producer_stream_o[pidx*5+:5]==s&&mem_producer_boundary_o[pidx*4+:4]==group+1,"MEM Read producer mapping");end end end endtask
  task automatic east_check(input integer t);integer idx;begin for(stream=0;stream<16;stream=stream+1)begin idx=si(0,14,t,stream);ck(state_valid_o[idx]&&state_data_o[idx*BITS+:BITS]===pat(t,stream),"sreg14 East alignment");end end endtask
  task automatic west_check(input integer t);integer idx;begin for(stream=0;stream<16;stream=stream+1)begin idx=si(1,14,t,stream);ck(state_valid_o[idx]&&state_data_o[idx*BITS+:BITS]===ref_t(t,stream),"sreg14 West transpose result");end end endtask
  initial begin
    clk_i=0;rst_ni=1;errors=0;clear();reset();preload();reset();reads();
    for(w=0;w<11;w=w+1)begin @(negedge clk_i);clear();step();end
    east_check(0); cap_cycle[0]=$time/10;
    $display("TURNAROUND_MEM_READ PASS");$display("TURNAROUND_SRF_EAST_ALIGN PASS");
    @(negedge clk_i);clear();vmodel_transpose_valid_i=1;vmodel_transpose_instruction_i=sp(SXM_TRANSPOSE,0,0,0,16,0,0);#1;
    ck(dut.sxm_consume!='0,"SXM slot1 consume"); step();
    for(tile=1;tile<4;tile=tile+1)begin
      east_check(tile); cap_cycle[tile]=$time/10;
      @(negedge clk_i);clear();step();
    end
    ck(sxm_transpose_input_invalid_o==='0&&sxm_transpose_buffer_full_o==='0,"SXM capture status");
    $display("TURNAROUND_SXM_CAPTURE PASS");
    @(negedge clk_i);clear();step();
    for(tile=0;tile<4;tile=tile+1)begin
      buffer_cycle[tile]=$time/10;
      @(negedge clk_i);clear();vmodel_permute_valid_i=1;vmodel_permute_instruction_i=sp(SXM_PERMUTE,0,16,1,0,tile,(tile==1||tile==3)?2:0);perm_cycle[tile]=$time/10;step();west_cycle[tile]=$time/10;west_check(tile);
    end
    @(negedge clk_i); clear(); #1;
    ck(sxm_fault_valid_o==0&&sxm_permute_phase_fault_o==0&&sxm_permute_selector_fault_o==0&&sxm_permute_buffer_not_ready_o==0,"SXM Permute status");
    $display("TURNAROUND_RESULT_BUFFER PASS");$display("TURNAROUND_SXM_PERMUTE PASS");$display("TURNAROUND_SRF_WEST PASS");$display("TURNAROUND_DATA_COMPARE PASS");
    ck(mem_command_fault_o==='0&&sxm_command_fault_o==0&&mem_bank_fault_valid_o==='0&&mem_fault_valid_o==0&&srf_collision_o==='0&&srf_invalid_consume_o==='0,"legal turnaround status");
    if(errors==0)begin $display("TURNAROUND_CYCLE_CONTRACT PASS");$display("CYCLES cap=%0d,%0d,%0d,%0d buffer=%0d,%0d,%0d,%0d perm=%0d,%0d,%0d,%0d west=%0d,%0d,%0d,%0d",cap_cycle[0],cap_cycle[1],cap_cycle[2],cap_cycle[3],buffer_cycle[0],buffer_cycle[1],buffer_cycle[2],buffer_cycle[3],perm_cycle[0],perm_cycle[1],perm_cycle[2],perm_cycle[3],west_cycle[0],west_cycle[1],west_cycle[2],west_cycle[3]);$display("========================================");$display("VMODEL_MEM_SRF_SXM_TURNAROUND TEST_PASS");$display("========================================");end else $display("VMODEL_MEM_SRF_SXM_TURNAROUND TEST_FAIL errors=%0d",errors);$finish;
  end
endmodule
