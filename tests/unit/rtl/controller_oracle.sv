`timescale 1ns/1ps
module controller_oracle;
  reg clk=0,rst_n=0,cv=0,acc=0,out=0,lr=0,er=0,cd=0;
  reg [9:0] count=0;reg [7:0] interval=0;
  wire ready,load,execute,busy,done,fault;wire [3:0] phase;
  lpu_vxm_instruction_controller dut(.clk_i(clk),.rst_ni(rst_n),.command_valid_i(cv),.command_ready_o(ready),
    .command_local_active_i(8'b0),.command_local_instruction_i(56'b0),.command_global_config_i(15'b0),
    .command_repeat_count_i(count),.command_repeat_interval_i(interval),.command_accumulator_enable_i(acc),.command_output_enable_i(out),
    .local_config_load_o(load),.local_config_ready_i(lr),.execute_valid_o(execute),.execute_ready_i(er),
    .repeat_control_o(phase),.config_done_i(cd),.busy_o(busy),.done_o(done),.fault_o(fault));
  integer fi,fo,rc,t=0,ri,vi,ci,ii,ai,oi,li,ei,di;string input_path,output_path;
  initial begin
    if(!$value$plusargs("INPUT=%s",input_path)||!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"paths");
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;fi=$fopen(input_path,"r");fo=$fopen(output_path,"w");
    if (!fi || !fo) $fatal(1,"cannot open test trace file");
    while(!$feof(fi))begin
      rc=$fscanf(fi,"%d %d %d %d %d %d %d %d %d",ri,vi,ci,ii,ai,oi,li,ei,di);
      if(rc==9)begin
        rst_n=!ri;cv=vi;count=ci;interval=ii;acc=ai;out=oi;lr=li;er=ei;cd=di;
        #2;clk=1;#1;$fwrite(fo,"%0d %0d %0d %0d %0d %01x %0d %0d\n",t,fault,ready,load,execute,phase,done,busy);
        clk=0;#1;t=t+1;
      end
    end
    $fclose(fi);$fclose(fo);$finish;
  end
endmodule
