`timescale 1ns/1ps
module mxm_core_oracle;
  reg clk=0,rst_n=0;
  reg ce=0,cl=0,le=0,ll=0,pe=0,pl=0,de=0,dl=0,be=0,bl=0;
  reg [12:0] backend=0;
  reg [4*8*16-1:0] activation={32{16'h3f80}};
  reg [4*32*8-1:0] stream_valid='1;
  reg [4*32*8*8-1:0] stream_data=0;
  wire [3:0] tv,rv,ov,bf;
  wire [4*8*32-1:0] top,returned,result;
  wire [63:0] top_tag,return_tag,result_tag;
  wire fault;
  lpu_mxm_core dut(.clk_i(clk),.rst_ni(rst_n),.run_i(1'b1),
    .compute_enqueue_valid_i(ce),.compute_launch_i(cl),.compute_instruction_i(7'b0),
    .load_enqueue_valid_i(le),.load_launch_i(ll),.load_instruction_i(2'b0),
    .param_load_enqueue_valid_i(pe),.param_load_launch_i(pl),.param_load_instruction_i(5'b01000),
    .dequant_enqueue_valid_i(de),.dequant_launch_i(dl),.dequant_instruction_i(3'b000),
    .backend_enqueue_valid_i(be),.backend_launch_i(bl),.backend_instruction_i(backend),
    .activation_valid_i(4'b1111),.activation_i(activation),
    .weight_stream_valid_i(stream_valid),.weight_stream_i(stream_data),
    .partial_valid_o(tv),.partial_o(top),.partial_tag_o(top_tag),
    .returned_valid_o(rv),.returned_o(returned),.returned_tag_o(return_tag),
    .result_valid_o(ov),.result_o(result),.result_tag_o(result_tag),.result_bf16_o(bf),.fault_o(fault));
  integer fd,t,c,lane,n;string output_path;
  initial begin
    if(!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"output path");
    for(integer r=0;r<4;r=r+1)for(integer l=0;l<8;l=l+1)begin
      stream_data[(r*32*8+l)*8+:8]=1;
      stream_data[(r*32*8+8+l)*8+:8]=8'h80;
      stream_data[(r*32*8+16+l)*8+:8]=8'h3f;
    end
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;
    fd=$fopen(output_path,"w");
    if (!fd) $fatal(1,"cannot open test trace file");
    for(t=0;t<175;t=t+1)begin
      for(integer r=0;r<4;r=r+1)for(integer l=0;l<8;l=l+1)begin
        // Match the unsigned, wraparound test-pattern generator in C++.
        stream_data[(r*32*8+l)*8+:8]=8'(((32'(t-3*r))*5+l*7+r*11)%17-8);
        stream_data[(r*32*8+8+l)*8+:8]=8'h80;
        stream_data[(r*32*8+16+l)*8+:8]=8'h3e+((r+l)%3);
        stream_data[(r*32*8+24+l)*8+:8]=8'(l%3-1);
        activation[(r*8+l)*16+:16]=16'h3f00+(((32'(t-3*r))*13+r*19+l*5)%192);
      end
      pe=t==0;pl=t==1;le=t==2;ll=t==3;
      de=t>=2&&t<34;dl=t>=3&&t<35;
      ce=t>=39&&t<135;cl=t>=40&&t<136;
      be=t>=62&&t<158;bl=t>=63&&t<159;
      n=t-62;
      backend=0;
      if(n>=32&&n<64)backend=((n-32)<<2)|1;
      if(n>=64&&n<96)backend=((n-64)<<2)|3;
      #2;clk=1;#1;
      for(c=0;c<4;c=c+1)begin
        $fwrite(fd,"%0d %0d %0d %0d %064x %0d %0d %064x %0d %0d %064x %0d %0d\n",t,c,fault,
          tv[c],tv[c]?top[c*256+:256]:0,tv[c]?top_tag[c*16+:16]:0,
          rv[c],rv[c]?returned[c*256+:256]:0,rv[c]?return_tag[c*16+:16]:0,
          ov[c],ov[c]?result[c*256+:256]:0,ov[c]?result_tag[c*16+:16]:0,bf[c]&&ov[c]);
      end
      clk=0;#1;
    end
    $fclose(fd);$display("MXM_CORE_ORACLE PASS cycles=175");$finish;
  end
endmodule
