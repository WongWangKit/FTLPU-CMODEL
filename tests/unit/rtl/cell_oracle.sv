`timescale 1ns/1ps
module cell_oracle;
  reg clk=0,rst_n=0,av=0,ab=0,wv=0,wb=0;reg [15:0] tag=0;
  reg [127:0] a=0,w=0;
  wire wa,wfv,wfb,afv,afb,fault;wire [127:0] wf,af;
  wire [15:0] aft;wire [1:0] ready;wire [7:0] dv,pv,lost;
  wire [255:0] dot,partial;wire [127:0] dt,pt;
  lpu_mxm_supercell #(.BOTTOM(1)) dut(.clk_i(clk),.rst_ni(rst_n),.activation_valid_i(av),.activation_i(a),
    .weight_buffer_i(ab),.tag_i(tag),.weight_write_valid_i(wv),.weight_write_buffer_i(wb),.weight_write_data_i(w),
    .weight_write_accepted_o(wa),.weight_forward_valid_o(wfv),.weight_forward_buffer_o(wfb),.weight_forward_data_o(wf),
    .weight_buffer_ready_o(ready),.activation_valid_o(afv),.activation_o(af),.weight_buffer_o(afb),.activation_tag_o(aft),
    .below_valid_i(8'b0),.below_partial_i(256'b0),.below_tag_i(128'b0),
    .dot_valid_o(dv),.dot_result_o(dot),.dot_tag_o(dt),.alignment_lost_o(lost),
    .partial_valid_o(pv),.partial_o(partial),.partial_tag_o(pt),.fault_o(fault));
  integer fd,fo,t=0,rc,ai,abi,wi,wbi,ti,l;reg [15:0] lane_value;
  reg [255:0] checked_dot,checked_partial;reg [7:0] checked_lost;
  string input_path,output_path;
  initial begin
    if(!$value$plusargs("INPUT=%s",input_path)||!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"paths");
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;
    fd=$fopen(input_path,"r");fo=$fopen(output_path,"w");
    if (!fd || !fo) $fatal(1,"cannot open test trace file");
    while(!$feof(fd))begin
      rc=$fscanf(fd,"%d %d %d %d %d",ai,abi,wi,wbi,ti);
      if(rc==5)begin
        av=ai;ab=abi;wv=wi;wb=wbi;tag=ti;
        for(l=0;l<8;l=l+1)begin rc=$fscanf(fd," %h",lane_value);a[l*16+:16]=lane_value;end
        for(l=0;l<8;l=l+1)begin rc=$fscanf(fd," %h",lane_value);w[l*16+:16]=lane_value;end
        #1;
        // Acceptance is combinational before the edge; tail/link data and
        // arithmetic outputs are observed after the same edge.
        wi=wa;#1;clk=1;#1;
        checked_dot=0;checked_partial=0;checked_lost=0;
        for(l=0;l<8;l=l+1)begin
          if(dv[l])begin checked_dot[l*32+:32]=dot[l*32+:32];checked_lost[l]=lost[l];end
          if(pv[l])checked_partial[l*32+:32]=partial[l*32+:32];
        end
        $fwrite(fo,"%0d %0d %0d %02x %064x %02x %02x %064x %0d %0d %032x %0d %0d %032x %0d %0d\n",
          t,fault,wi,dv,checked_dot,checked_lost,pv,checked_partial,wfv,wfv&&wfb,wfv?wf:0,
          afv,afv&&afb,afv?af:0,afv?aft:0,ready);
        clk=0;#1;t=t+1;
      end
    end
    $fclose(fo);$fclose(fd);$display("CELL_ORACLE PASS cycles=%0d",t);$finish;
  end
endmodule
