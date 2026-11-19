`timescale 1ns/1ps
module vxm_tile_oracle;
  parameter integer CHAIN=0,FORMAT=0,TAIL_OP=0;
  integer selected_chain=CHAIN,selected_format=FORMAT,selected_op=TAIL_OP,length,mode=0;
  reg clk=0,rst_n=0,load=0,execute=0;
  reg [7:0] active=0;reg [55:0] words=0;reg [14:0] global_word=0;
  reg [2047:0] data=0;wire [31:0] consume,ov;
  reg [31:0] input_valid=32'hffffffff,output_ready=32'hffffffff;reg [3:0] phase=4'b1011;
  wire [2047:0] out_data;wire cr,er,done,fault;wire [127:0] tail_valid;
  reg [15:0] tail_mask;
  lpu_vxm_tile_execution #(.LUT_BANK_COUNT(4)) dut(.clk_i(clk),.rst_ni(rst_n),
    .local_config_load_i(load),.local_config_ready_o(cr),.local_config_active_i(active),.local_config_instruction_i(words),
    .execute_valid_i(execute),.execute_ready_o(er),.repeat_control_i(phase),.config_done_o(done),
    .global_config_valid_i(1'b1),.global_config_i(global_word),.stream_valid_i(input_valid),.stream_data_i(data),
    .stream_consumed_o(consume),.immediate_valid_i(8'b0),.immediate_data_i(256'b0),
    .lut_config_valid_i(1'b0),.lut_config_bank_i(2'b0),.lut_config_input_min_i(16'b0),.lut_config_segment_width_i(16'h3c00),
    .lut_write_valid_i(1'b0),.lut_write_bank_i(2'b0),.lut_write_address_i(6'b0),.lut_write_k_i(16'b0),.lut_write_b_i(16'b0),.lut_write_log_i(50'b0),
    .output_ready_i(output_ready),.output_valid_o(ov),.output_data_o(out_data),.tail_valid_o(tail_valid),.fault_o(fault));
  integer t,s,l,q,fd;reg [31:0] value;reg [15:0] half_value;reg [31:0] consume_before;
  string output_path;
  initial begin
    if(!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"path");
    if($value$plusargs("CHAIN=%d",selected_chain))begin end
    if($value$plusargs("FORMAT=%d",selected_format))begin end
    if($value$plusargs("OP=%d",selected_op))begin end
    if($value$plusargs("MODE=%d",mode))begin end
    length=2<<selected_chain;
    global_word=(selected_chain<<12)|(selected_format<<8)|((selected_format==2)<<6)|(selected_format<<4)|((selected_format==2)<<2)|selected_format;
    for(q=0;q<length;q=q+1)active[q]=1;
    words[(length-1)*7+:7]=selected_op|(mode==4?24:0);
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;
    fd=$fopen(output_path,"w");
    if (!fd) $fatal(1,"cannot open test trace file");
    for(t=0;t<(mode==4?100:50);t=t+1)begin
      load=t==0;execute=t==1;
      output_ready=(mode==1&&t<28)?0:32'hffffffff;
      input_valid=(mode==2&&t<5)?0:32'hffffffff;phase=4'b1011;
      if(mode==3)begin phase[3]=t>=25;if(t==25)begin load=1;words[0+:7]=16;end if(t==26)execute=1;end
      if(mode==4)begin phase={1'b1,1'b1,t<33,t>=65};execute=t==1||t==33||t==65;end
      if(mode==5)begin rst_n=t!=10;if(t==12)load=1;if(t==13)execute=1;end
      for(s=0;s<32;s=s+1)for(l=0;l<8;l=l+1)begin
        value=selected_format==0?32'h3c00+l:selected_format==1?32'h3f80+l:32'h3f801234+l;
        half_value=(selected_format==2&&(t==3||(mode==2&&t==6)||(mode==3&&t==28)||
          (mode==4&&(t==35||t==67))||(mode==5&&t==15)))?value[31:16]:value[15:0];
        data[(s*8+l)*8+:8]=(s%2)?half_value[15:8]:half_value[7:0];
      end
      #1;consume_before=consume;#1;clk=1;#1;
      for(s=0;s<16;s=s+1)tail_mask[s]=&tail_valid[s*8+:8];
      $fwrite(fd,"%0d %0d %0d %0d %0d %08x %08x %04x %0256x\n",t,fault,cr,er,done,consume_before,ov,tail_mask,out_data[1023:0]);
      clk=0;#1;
    end
    $fclose(fd);$display("VXM_TILE_ORACLE PASS chain=%0d format=%0d",CHAIN,FORMAT);$finish;
  end
endmodule
