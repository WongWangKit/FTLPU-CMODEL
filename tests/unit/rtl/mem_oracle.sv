`timescale 1ns/1ps
module mem_oracle;
  parameter integer ROWS=4;
  reg clk=0,rst_n=0,valid=0;reg [31:0] word=0;
  reg [ROWS-1:0] sv=0;reg [ROWS*64-1:0] data=0;
  wire [ROWS-1:0] consume,rv,fault,cv,dir;wire [ROWS*64-1:0] rd;wire [ROWS*5-1:0] si;
  wire busy;reg [ROWS-1:0] before_cv;reg before_busy;
  mem_logical_bank_column #(.P_TILE_ROWS(ROWS),.P_MEM_BANK_DEPTH_ROWS(64)) dut(
    .clk_i(clk),.rst_ni(rst_n),.south_cmd_valid_i(valid),.south_cmd_i(word),.stream_valid_i(sv),.stream_data_i(data),
    .stream_consume_o(consume),.read_valid_o(rv),.read_data_o(rd),.read_stream_dir_o(dir),.read_stream_idx_o(si),
    .fault_valid_o(fault),.current_cmd_valid_o(cv),.pipeline_busy_o(busy));
  integer t,r,s,short_start,n,fd;string output_path;
  initial begin
    if(!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"path");
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;
    fd=$fopen(output_path,"w");
    if (!fd) $fatal(1,"cannot open test trace file");
    s=32+3*(ROWS-1)+2;short_start=s+32+2*(ROWS-1);
    for(t=0;t<260;t=t+1)begin
      valid=0;word=0;
      if(t<32)begin valid=1;word=32'h40000001|(t<<15)|(7<<3);end
      if(t>=s&&t<s+32)begin valid=1;word=32'h40000100|((t-s)<<15)|(3<<3);end
      if(t>=short_start&&t<short_start+32)begin valid=1;word=32'h80000001|((t-short_start)<<15)|(5<<3);end
      if(t>=short_start+32&&t<short_start+64)begin valid=1;word=((t-short_start-32)<<15)|(5<<3);end
      if(t>=short_start+64&&t<short_start+80)begin valid=1;word=32'h40000000|((t-short_start-64)<<15)|(6<<3);end
      if(t==245)begin valid=1;word=0;end
      for(r=0;r<ROWS;r=r+1)begin sv[r]=((t+r)%13)!=0;data[r*64+:64]=64'h1020304050607080^(64'h0101010101010101*(t+17*r));end
      if(t==240)begin rst_n=0;#1;end
      #1;before_cv=cv;before_busy=busy;#1;clk=1;#1;
      for(r=0;r<ROWS;r=r+1)$fwrite(fd,"%0d %0d %0d %0d %016x %0d %0d %0d %0d\n",t,r,before_cv[r],rv[r],rv[r]?rd[r*64+:64]:0,consume[r],fault[r],rv[r]?dir[r]:0,rv[r]?si[r*5+:5]:0);
      clk=0;#1;if(t==240)rst_n=1;
    end
    $fclose(fd);$display("MEM_ORACLE PASS rows=%0d",ROWS);$finish;
  end
endmodule
