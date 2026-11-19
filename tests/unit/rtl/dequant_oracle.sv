`timescale 1ns/1ps
module dequant_oracle;
  reg clk=0,rst_n=0,run=1,request=0,wb=0,load=0;
  reg [4:0] param_word=0;
  reg [2:0] dq_word=0;
  reg [255:0] sv=0;
  reg [2047:0] data=0;
  wire accepted,pa,valid,buffer,fault;
  wire [1:0] ready;
  wire [255:0] consume;
  wire [127:0] weight;
  lpu_mxm_weight_dequant_pipe dut(.clk_i(clk),.rst_ni(rst_n),.run_i(run),.request_i(request),.buffer_i(wb),
    .param_write_i(load),.param_write_buffer_i(param_word[0]),.param_scale_mode_i(param_word[2:1]),
    .param_zero_mode_i(param_word[4:3]),.param_buffer_i(dq_word[0]),.format_i(dq_word[2:1]),
    .stream_valid_i(sv),.stream_i(data),.consume_o(consume),.accepted_o(accepted),.valid_o(valid),
    .buffer_o(buffer),.param_accepted_o(pa),.param_ready_o(ready),.weight_bf16_o(weight),.fault_o(fault));
  integer fi,fo,rc,t=0,ri,runi,rqi,wbi,li,index;
  reg [7:0] raw_byte;
  reg [47:0] mask,cc;
  reg ca,cp;
  string input_path,output_path;
  initial begin
    if(!$value$plusargs("INPUT=%s",input_path)||!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"paths");
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;
    fi=$fopen(input_path,"r");fo=$fopen(output_path,"w");
    if (!fi || !fo) $fatal(1,"cannot open test trace file");
    while(!$feof(fi))begin
      rc=$fscanf(fi,"%d %d %d %d %d %d %d %h",ri,runi,rqi,wbi,li,param_word,dq_word,mask);
      if(rc==8)begin
        rst_n=!ri;run=runi;request=rqi;wb=wbi;load=li;sv={208'b0,mask};
        for(index=0;index<48;index=index+1)begin
          if($fscanf(fi," %h",raw_byte)!=1)$fatal(1,"invalid stream byte");
          data[index*8+:8]=raw_byte;
        end
        #1;ca=rst_n&&accepted;cp=rst_n&&pa;cc=rst_n?consume[47:0]:0;#1;clk=1;#1;
        $fwrite(fo,"%0d %0d %0d %0d %012x %0d %0d %032x %0d\n",
          t,fault,ca,cp,cc,valid,valid&&buffer,valid?weight:0,ready);
        clk=0;#1;t=t+1;
      end
    end
    $fclose(fi);$fclose(fo);$finish;
  end
endmodule
