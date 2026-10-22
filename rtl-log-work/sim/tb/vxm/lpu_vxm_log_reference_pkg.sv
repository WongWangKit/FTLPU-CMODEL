// Test-only evaluator built from the production bit-level arithmetic helpers.
// Expected answers are supplied independently by the vector generator.
package lpu_vxm_log_reference_pkg;
  import lpu_pkg::*;
  import lpu_vxm_math_pkg::*;
  import lpu_vxm_log_math_pkg::*;

  function automatic logic [31:0] log_evaluate_bits(
    input logic [31:0] raw, input logic [1:0] dtype, input logic [49:0] row
  );
    logic sign, zero_value, nan_value, inf_value, near_value;
    logic [7:0] exponent;
    logic [22:0] fraction;
    logic [31:0] x, delta, inv, u, h, product, p, restored;
    logic [27:0] base;
    logic signed [47:0] fixed_value, base_value, e_value;
    integer e;
    begin
      exponent=0;fraction=0;sign=0;x=0;delta=0;e=0;
      case(dtype)
        VXM_FORMAT_FP16: begin
          sign=raw[15];exponent={3'b0,raw[14:10]};fraction={13'b0,raw[9:0]};
          x=fp16_to_fp32(raw[15:0]);e=integer'(exponent)-15;
          delta=log_fixed_to_fp32($signed({44'b0,fraction[3:0]}),10);
        end
        VXM_FORMAT_BF16: begin
          sign=raw[15];exponent=raw[14:7];fraction={16'b0,raw[6:0]};
          x=bf16_to_fp32(raw[15:0]);e=integer'(exponent)-127;
          delta=log_fixed_to_fp32($signed({47'b0,fraction[0]}),7);
        end
        default: begin
          sign=raw[31];exponent=raw[30:23];fraction=raw[22:0];x=raw;
          e=integer'(exponent)-127;
          delta=log_fixed_to_fp32($signed({31'b0,fraction[16:0]}),23);
        end
      endcase
      zero_value=exponent==0;
      inf_value=(dtype==VXM_FORMAT_FP16 ? exponent==31 : exponent==255) && fraction==0;
      nan_value=(dtype==VXM_FORMAT_FP16 ? exponent==31 : exponent==255) && fraction!=0;
      near_value=dtype==VXM_FORMAT_BF16 ?
        x>=32'h3f7e0000 && x<=32'h3f810000 :
        x>=32'h3f7c0000 && x<=32'h3f820000;
      base=log_active_base(row[49:22],dtype);
      inv=log_fixed_to_fp32($signed({26'b0,log_active_inverse(row[21:0],dtype)}),21);
      if(near_value) begin
        delta=fp32_add_rne(x,32'hbf800000);inv=32'h3f800000;base=0;e=0;
      end
      u=log_round_format(fp32_multiply_rne(delta,inv),dtype);
      h=32'h3f800000;
      if(dtype==VXM_FORMAT_FP16)
        h=log_round_format(fp32_add_rne(log_round_format(
          fp32_multiply_rne(32'hbf000000,u),dtype),32'h3f800000),dtype);
      else if(dtype==VXM_FORMAT_FP32) begin
        h=fp32_add_rne(fp32_multiply_rne(32'hbe800000,u),32'h3eaaaaab);
        h=fp32_add_rne(fp32_multiply_rne(h,u),32'hbf000000);
        h=fp32_add_rne(fp32_multiply_rne(h,u),32'h3f800000);
      end
      p=log_round_format(fp32_multiply_rne(h,u),dtype);
      base_value=$signed({8'b0,base,12'b0});
      e_value=e*LOG_LN2_Q40;
      fixed_value=e_value+base_value+log_fp32_to_q40(p);
      restored=near_value ? p : log_fixed_to_fp32(fixed_value,40);
      if(nan_value) restored=32'h7fc00000;
      else if(zero_value) restored=32'hff800000;
      else if(sign) restored=32'h7fc00000;
      else if(inf_value) restored=32'h7f800000;
      log_evaluate_bits=log_pack_result(restored,dtype);
    end
  endfunction

  function automatic logic log_within_ulp(
    input logic [31:0] got, expected, input logic [1:0] dtype
  );
    logic [31:0] a,b,distance;
    begin
      a=got;b=expected;
      if(dtype!=VXM_FORMAT_FP32) begin
        a={16'b0,got[15:0]};b={16'b0,expected[15:0]};
        if(dtype==VXM_FORMAT_FP16 ? b[14:10]==31 : b[14:7]==255)
          log_within_ulp=a==b;
        else begin
          distance=a>b?a-b:b-a;
          log_within_ulp=(a[15]==b[15]) && distance<=1;
        end
      end else if(b[30:23]==255) log_within_ulp=a==b;
      else begin
        distance=a>b?a-b:b-a;
        log_within_ulp=(a[31]==b[31]) && distance<=2;
      end
    end
  endfunction
endpackage
