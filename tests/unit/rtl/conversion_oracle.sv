`timescale 1ns/1ps
module conversion_oracle;
  reg valid=0;reg [1:0] source=0,target=0;reg [31:0] raw=0;
  wire vo,active,fault;wire [31:0] bits;
  lpu_vxm_input_converter dut(.valid_i(valid),.source_dtype_i(source),.compute_dtype_i(target),.raw_data_i(raw),
    .valid_o(vo),.converted_data_o(bits),.conversion_active_o(active),.unsupported_conversion_o(fault));
  integer fi,fo,rc,t=0,v,s,d;string input_path,output_path;
  initial begin
    if(!$value$plusargs("INPUT=%s",input_path)||!$value$plusargs("OUTPUT=%s",output_path))$fatal(1,"paths");
    fi=$fopen(input_path,"r");fo=$fopen(output_path,"w");
    if (!fi || !fo) $fatal(1,"cannot open test trace file");
    while(!$feof(fi))begin
      rc=$fscanf(fi,"%d %d %d %h",v,s,d,raw);
      if(rc==4)begin valid=v;source=s;target=d;#1;$fwrite(fo,"%0d %0d %0d %0d %08x\n",t,vo,active,fault,bits);t=t+1;end
    end
    $fclose(fi);$fclose(fo);$finish;
  end
endmodule
