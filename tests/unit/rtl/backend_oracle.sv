`timescale 1ns/1ps
module backend_oracle;
  reg clk=0,rst_n=0,run=1,slot=0,pv=0,bf=0;
  reg [1:0] op=0;reg [15:0] addr=0,tag=0;reg [255:0] partial=0;
  wire rv,rb,fault;wire [255:0] result;wire [15:0] ra,rt;
  wire [1:0] req,wr;wire [7:0] ba;wire [511:0] wd;
  lpu_mxm_backend_column #(.DEPTH(32)) dut(.clk_i(clk),.rst_ni(rst_n),.run_i(run),
    .slot_valid_i(slot),.operation_i(op),.output_bf16_i(bf),.address_i(addr),
    .partial_valid_i(pv),.partial_i(partial),.partial_tag_i(tag),
    .result_valid_o(rv),.result_bf16_o(rb),.result_o(result),.result_address_o(ra),.result_tag_o(rt),
    .bank_request_valid_o(req),.bank_write_o(wr),.bank_address_o(ba),.bank_write_data_o(wd),.fault_o(fault));
  integer fi,fo,rc,t=0,runi,reseti,launch,operation,quant,address,valid,tagi,l;
  reg [31:0] word;reg [1:0] checked_req,checked_wr;reg [7:0] checked_ba;reg [511:0] checked_wd;
  string input_path,output_path;
  initial begin
    if(!$value$plusargs("INPUT=%s",input_path)||!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"paths");
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;
    fi=$fopen(input_path,"r");fo=$fopen(output_path,"w");
    if (!fi || !fo) $fatal(1,"cannot open test trace file");
    while(!$feof(fi))begin
      rc=$fscanf(fi,"%d %d %d %d %d %d %d %d",runi,reseti,launch,operation,quant,address,valid,tagi);
      if(rc==8)begin
        run=runi;rst_n=!reseti;pv=valid;tag=tagi;
        for(l=0;l<8;l=l+1)begin rc=$fscanf(fi," %h",word);partial[l*32+:32]=word;end
        if(reseti)begin slot=0;pv=0;end
        #1;checked_req=req;checked_wr=wr&req;checked_ba=0;checked_wd=0;
        for(l=0;l<2;l=l+1)if(req[l])begin
          checked_ba[l*4+:4]=ba[l*4+:4];if(wr[l])checked_wd[l*256+:256]=wd[l*256+:256];
        end
        #1;clk=1;#1;
        $fwrite(fo,"%0d %0d %0d %0d %0d %0d %064x %01x %01x %02x %0128x\n",
          t,fault,rv,rv&&rb,rv?ra:0,rv?rt:0,rv?result:0,checked_req,checked_wr,checked_ba,checked_wd);
        clk=0;slot=launch&&runi&&!reseti;op=operation;bf=quant;addr=address;#1;t=t+1;
      end
    end
    $fclose(fi);$fclose(fo);$finish;
  end
endmodule
