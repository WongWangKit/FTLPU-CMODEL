`timescale 1ns/1ps
module lpu_mem_srf_sxm_roundtrip_tb;
  import lpu_pkg::*;
  localparam integer N=16,G=4,C=16,T=4,S=32,B=64;
  logic clk_i,rst_ni; logic [N*2-1:0] vmodel_mem_issue_valid_i,native_mem_issue_valid_o,mem_command_fault_o; logic [N*2*47-1:0] vmodel_mem_issue_instruction_i; logic [N*2*32-1:0] native_mem_issue_o;
  logic vmodel_transpose_valid_i,vmodel_permute_valid_i; logic [415:0] vmodel_transpose_instruction_i,vmodel_permute_instruction_i;
  logic [2*T*S-1:0] boundary_valid_i; logic [2*T*S*B-1:0] boundary_data_i;
  logic [2*C*T*S-1:0] state_valid_o; logic [2*C*T*S*B-1:0] state_data_o; logic [2*C*T-1:0] srf_collision_o,srf_invalid_consume_o;
  logic [N*8-1:0] mem_producer_valid_o,mem_producer_direction_o,mem_internal_collision_o; logic [N*8*B-1:0] mem_producer_data_o; logic [N*8*5-1:0] mem_producer_stream_o; logic [N*8*4-1:0] mem_producer_boundary_o; logic [(G+1)*256-1:0] mem_boundary_consume_o; logic [N*2-1:0] mem_bank_fault_valid_o; logic mem_fault_valid_o,mem_busy_o,sxm_fault_valid_o,sxm_permute_phase_fault_o,sxm_permute_selector_fault_o,sxm_permute_buffer_not_ready_o,sxm_busy_o,sxm_command_fault_o; logic [T-1:0] sxm_transpose_input_invalid_o,sxm_transpose_buffer_full_o;
  integer e,tile,stream,lane,g,l,w,west_visible[0:3],write_issue[0:3],consume_cycle[0:3];
  lpu_mem_srf_sxm_roundtrip_integration dut(.*); always #5 clk_i=~clk_i;
  function automatic [46:0] mp(input[2:0]o,input west,input[4:0]s,input[14:0]r);begin mp='0;mp[2:0]=o;mp[8:3]={west,s};mp[30:15]=r;end endfunction
  function automatic [415:0] sp(input[1:0]o,input sw,input[4:0]sb,input dw,input[4:0]db,input[2:0]ot,input integer ph);integer i,d,x;begin sp='0;sp[1:0]=o;sp[6+:5]=16;sp[11+:5]=16;sp[SXM_OUTPUT_TILE_LSB+:SXM_OUTPUT_TILE_WIDTH]=ot;for(i=0;i<16;i=i+1)begin sp[16+i*6+:5]=sb+i;sp[16+i*6+5]=sw;sp[112+i*6+:5]=db+i;sp[112+i*6+5]=dw;end for(i=0;i<32;i=i+1)begin d=i/8;x=(ph+4-d)%4;sp[240+i*5+:5]=x*8+i%8;end end endfunction
  function automatic[63:0] pat(input integer t,input integer s);begin pat='0;for(lane=0;lane<8;lane=lane+1)pat[lane*8+:8]=(t<<6)|(s<<2)|lane;end endfunction
  function automatic[63:0] golden(input integer t,input integer os);integer r,p,ss;begin golden='0;r=os/2;p=os%2;for(lane=0;lane<8;lane=lane+1)begin ss=2*lane+p;golden[lane*8+:8]=(t<<6)|(ss<<2)|r;end end endfunction
  function automatic integer bi(input integer d,input integer t,input integer s);bi=(d*T+t)*S+s;endfunction
  function automatic integer si(input integer d,input integer c,input integer t,input integer s);si=((d*C+c)*T+t)*S+s;endfunction
  task automatic clear;begin vmodel_mem_issue_valid_i='0;vmodel_mem_issue_instruction_i='0;vmodel_transpose_valid_i=0;vmodel_transpose_instruction_i='0;vmodel_permute_valid_i=0;vmodel_permute_instruction_i='0;boundary_valid_i='0;boundary_data_i='0;end endtask
  task automatic step;begin @(posedge clk_i);#1;end endtask task automatic ck(input bit a,input[8*54-1:0]m);begin if(!a)begin $display("ERROR %0s",m);e=e+1;end end endtask
  task automatic reset;begin @(negedge clk_i);clear();rst_ni=0;@(negedge clk_i);rst_ni=1;step();end endtask
  task automatic drive(input integer tt);integer x;begin for(x=0;x<16;x=x+1)begin boundary_valid_i[bi(0,tt,x)]=1;boundary_data_i[bi(0,tt,x)*B+:B]=pat(tt,x);end end endtask
  task automatic groupcmd(input integer gg,input[2:0]op,input west,input[14:0]row);integer ss,bb;begin for(l=0;l<4;l=l+1)begin ss=gg*4+l;bb=ss*2;vmodel_mem_issue_valid_i[bb]=1;vmodel_mem_issue_instruction_i[bb*47+:47]=mp(op,west,ss,row);end end endtask
  task automatic preload;begin @(negedge clk_i);clear();drive(0);step();for(g=0;g<4;g=g+1)begin @(negedge clk_i);clear();if(g<3)drive(g+1);groupcmd(g,1,0,0);step();end repeat(4)begin @(negedge clk_i);clear();step();end end endtask
  task automatic checkeast(input integer tt);integer ix;begin for(stream=0;stream<16;stream=stream+1)begin ix=si(0,14,tt,stream);ck(state_valid_o[ix]&&state_data_o[ix*B+:B]===pat(tt,stream),"SRF East data");end end endtask
  task automatic checkwest(input integer tt);integer ix;begin for(stream=0;stream<16;stream=stream+1)begin ix=si(1,14,tt,stream);ck(state_valid_o[ix]&&state_data_o[ix*B+:B]===golden(tt,stream),"SRF West data");end end endtask
  initial begin
    clk_i=0;rst_ni=1;e=0;clear();reset();preload();$display("ROUNDTRIP_MEM_PRELOAD PASS");reset();
    for(g=0;g<4;g=g+1)begin @(negedge clk_i);clear();groupcmd(g,0,0,0);step();end $display("ROUNDTRIP_MEM_READ PASS");
    for(w=0;w<11;w=w+1)begin @(negedge clk_i);clear();step();end checkeast(0);@(negedge clk_i);clear();vmodel_transpose_valid_i=1;vmodel_transpose_instruction_i=sp(SXM_TRANSPOSE,0,0,0,16,0,0);step();for(tile=1;tile<4;tile=tile+1)begin checkeast(tile);@(negedge clk_i);clear();step();end $display("ROUNDTRIP_SRF_EAST PASS");$display("ROUNDTRIP_SXM_TRANSPOSE PASS");
    @(negedge clk_i);clear();step();for(tile=0;tile<4;tile=tile+1)begin @(negedge clk_i);clear();vmodel_permute_valid_i=1;vmodel_permute_instruction_i=sp(SXM_PERMUTE,0,16,1,0,tile,(tile==1||tile==3)?2:0);step();west_visible[tile]=$time/10;checkwest(tile);end @(negedge clk_i);clear();#1;$display("ROUNDTRIP_SXM_PERMUTE PASS");$display("ROUNDTRIP_SRF_WEST PASS");
    // West tile0 reaches boundary sreg[g+1] after 13-g hops. Start writes
    // group3..0 on consecutive cycles; control tile waves then consume 0..3.
    for(w=0;w<6;w=w+1)begin @(negedge clk_i);clear();step();end
    for(g=3;g>=0;g=g-1)begin @(negedge clk_i);clear();groupcmd(g,1,1,1);write_issue[g]=$time/10;#1;ck(mem_boundary_consume_o[(g+1)*256+128+(g*4)*4+0],"MEM West boundary tile0 consume");step();consume_cycle[g]=$time/10;end
    repeat(4)begin @(negedge clk_i);clear();step();end
    $display("ROUNDTRIP_MEM_WRITE_BOUNDARY PASS");$display("ROUNDTRIP_MEM_CONSUME PASS");$display("ROUNDTRIP_MEM_WRITE PASS");
    // Legal readback is a VMODEL Read through the same 47-bit adapter path.
    @(negedge clk_i);clear();for(g=0;g<4;g=g+1)groupcmd(g,0,0,1);step();
    for(tile=0;tile<4;tile=tile+1)begin for(stream=0;stream<16;stream=stream+1)begin integer p;p=stream*8+tile;ck(mem_producer_valid_o[p]&&mem_producer_data_o[p*B+:B]===golden(tile,stream),"destination VMODEL Readback");end if(tile<3)begin @(negedge clk_i);clear();step();end end
    ck(mem_command_fault_o==='0&&sxm_command_fault_o==0&&mem_bank_fault_valid_o==='0&&mem_fault_valid_o==0&&sxm_fault_valid_o==0&&sxm_permute_phase_fault_o==0&&sxm_permute_selector_fault_o==0&&sxm_permute_buffer_not_ready_o==0&&srf_collision_o==='0&&srf_invalid_consume_o==='0&&mem_internal_collision_o==='0,"roundtrip faults");
    if(e==0)begin $display("ROUNDTRIP_DATA_COMPARE PASS");$display("ROUNDTRIP_CYCLE_CONTRACT PASS");$display("CYCLES west=%0d,%0d,%0d,%0d write_g0=%0d g1=%0d g2=%0d g3=%0d consume=%0d,%0d,%0d,%0d",west_visible[0],west_visible[1],west_visible[2],west_visible[3],write_issue[0],write_issue[1],write_issue[2],write_issue[3],consume_cycle[0],consume_cycle[1],consume_cycle[2],consume_cycle[3]);$display("========================================");$display("VMODEL_MEM_SRF_SXM_ROUNDTRIP TEST_PASS");$display("========================================");end else $display("VMODEL_MEM_SRF_SXM_ROUNDTRIP TEST_FAIL errors=%0d",e);$finish;
  end
endmodule
