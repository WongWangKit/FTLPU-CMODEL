// C3/C7/C11/C15: fixed eight-edge latency, one accepted request per cycle.
// Four spatial multiply stages: M0=22x18, M1/M2/M3=24x24.
// Function and format tags travel with each element; no resource arbitration.
module lpu_vxm_c3_pipeline #(
  parameter integer LUT_BANK_COUNT=4, LUT_ENTRY_COUNT=64,
  parameter integer LUT_BANK_WIDTH=LUT_BANK_COUNT<=1?1:$clog2(LUT_BANK_COUNT),
  parameter integer LUT_ADDRESS_WIDTH=LUT_ENTRY_COUNT<=1?1:$clog2(LUT_ENTRY_COUNT)
) (
  input logic clk_i,rst_ni,valid_i,
  input logic [1:0] data_format_i,
  input logic [3:0] opcode_i,
  input logic operand_format_valid_i,operand_sign_i,
  input logic [7:0] operand_exponent_i,
  input logic [22:0] operand_fraction_i,
  input logic operand_zero_i,operand_inf_i,operand_nan_i,
  input logic [LUT_BANK_COUNT-1:0] lut_configured_i,
  input logic [LUT_BANK_COUNT*16-1:0] lut_input_min_i,lut_segment_width_i,
  output logic lut_read_valid_o,
  output logic [LUT_BANK_WIDTH-1:0] lut_read_bank_o,
  output logic [LUT_ADDRESS_WIDTH-1:0] lut_read_address_o,
  input logic lut_read_valid_i,
  input logic [15:0] lut_read_k_i,lut_read_b_i,
  input logic [49:0] lut_read_log_i,
  output logic ready_o,valid_o,
  output logic [31:0] result_o,
  output logic illegal_opcode_o,lut_fault_o
);
  import lpu_pkg::*;
  import lpu_vxm_math_pkg::*;
  import lpu_vxm_log_math_pkg::*;
  import lpu_vxm_c3_math_pkg::*;

  typedef struct packed {
    logic valid, direct, near_one, sign_bit, lookup;
    logic [3:0] op;
    logic [1:0] fmt;
    logic [5:0] address;
    logic [1:0] bank;
    logic [17:0] residual;
    logic signed [8:0] exponent;
    logic [27:0] base;
    logic [31:0] m,y,u,h,x,direct_result;
  } token_t;
  token_t incoming,q0,q1,q2,q3,q4,q5,q6;
  logic supported,configured,configuration_fault;
  logic [31:0] operand32;
  integer fraction_bits,bias,input_exponent;
  logic [21:0] m0_a;
  logic [17:0] m0_b;
  logic [39:0] m0_product;
  logic [15:0] active_k,active_b;
  logic [21:0] active_inverse;
  integer coefficient_bits,coefficient_shift;
  logic signed [47:0] fixed_seed;
  logic [47:0] retained,remainder,mask,halfway;
  logic [31:0] seed,u_value;
  logic m1_enable,m2_enable,m3_enable;
  logic [31:0] m1_lhs,m1_rhs,m2_lhs,m2_rhs,m3_lhs,m3_rhs;
  logic [23:0] m1_a,m1_b,m2_a,m2_b,m3_a,m3_b;
  logic [47:0] m1_product,m2_product,m3_product;
  logic [31:0] m1_result,m2_result,m3_result,restored;
  logic [31:0] rsqrt_correction;
  logic rsqrt_correction_fault;
  logic [31:0] negative_quarter_u,negative_half_u;
  logic signed [47:0] log_sum;

  function automatic logic [15:0] coefficient_rne(input logic [15:0] value,input integer shift);
    logic [16:0] keep;
    logic [15:0] discard,cut,half;
    begin
      keep={1'b0,value}; discard=0;cut=0;half=0;
      if(shift!=0) begin
        keep={1'b0,value}>>shift;cut=(16'b1<<shift)-1'b1;
        discard=value&cut;half=16'b1<<(shift-1);
        if(discard>half || (discard==half && keep[0])) keep=keep+1'b1;
      end
      coefficient_rne=keep[15:0];
    end
  endfunction

  // Powers of two require exponent adjustment, never a significand multiplier.
  function automatic logic [31:0] scale_pow2(input logic [31:0] value,input integer shift);
    integer ex;
    begin
      ex=integer'(value[30:23])+shift;
      if(value[30:23]==0 || value[30:23]==255) scale_pow2=value;
      else if(ex<=0) scale_pow2={value[31],31'b0};
      else if(ex>=255) scale_pow2={value[31],8'hff,23'b0};
      else scale_pow2={value[31],8'(ex),value[22:0]};
    end
  endfunction

  always_comb begin
    supported=opcode_i==VXM_LOCAL_SPECIAL0 || opcode_i==VXM_LOCAL_SPECIAL1 || opcode_i==VXM_LOCAL_LOG;
    ready_o=1'b1;
    illegal_opcode_o=valid_i && !supported;
    incoming='0;
    incoming.valid=valid_i && supported && operand_format_valid_i;
    incoming.op=opcode_i;incoming.fmt=data_format_i;incoming.sign_bit=operand_sign_i;
    fraction_bits=data_format_i==VXM_FORMAT_FP16?10:(data_format_i==VXM_FORMAT_BF16?7:23);
    bias=data_format_i==VXM_FORMAT_FP16?15:127;
    input_exponent=integer'(operand_exponent_i)-bias;
    operand32={operand_sign_i,operand_exponent_i,operand_fraction_i};
    if(data_format_i==VXM_FORMAT_FP16)
      operand32=fp16_to_fp32({operand_sign_i,operand_exponent_i[4:0],operand_fraction_i[9:0]});
    else if(data_format_i==VXM_FORMAT_BF16)
      operand32=bf16_to_fp32({operand_sign_i,operand_exponent_i,operand_fraction_i[6:0]});
    incoming.m={1'b0,8'd127,operand32[22:0]};
    incoming.exponent=9'(-input_exponent);
    incoming.bank=opcode_i==VXM_LOCAL_LOG?2'd3:(opcode_i==VXM_LOCAL_SPECIAL1?2'd2:2'd1);
    incoming.address=6'(operand_fraction_i>>(fraction_bits-6));
    incoming.residual=18'(operand_fraction_i & ((23'b1<<(fraction_bits-6))-1'b1));
    if(opcode_i==VXM_LOCAL_SPECIAL1) begin
      incoming.address={input_exponent[0],5'(operand_fraction_i>>(fraction_bits-5))};
      incoming.residual=18'(operand_fraction_i & ((23'b1<<(fraction_bits-5))-1'b1));
      incoming.exponent=9'(-((input_exponent-(input_exponent&1))/2));
      incoming.m[30:23]=input_exponent[0]?8'd128:8'd127;
    end
    incoming.direct=operand_nan_i || operand_zero_i || operand_inf_i ||
      ((opcode_i!=VXM_LOCAL_SPECIAL0) && operand_sign_i);
    incoming.direct_result=32'h7fc00000;
    if(opcode_i==VXM_LOCAL_SPECIAL0) begin
      if(operand_nan_i) incoming.direct_result=32'h7fc00000;
      else if(operand_zero_i) incoming.direct_result={operand_sign_i,8'hff,23'b0};
      else if(operand_inf_i) incoming.direct_result={operand_sign_i,31'b0};
    end else if(opcode_i==VXM_LOCAL_SPECIAL1) begin
      if(operand_nan_i || (operand_sign_i && !operand_zero_i)) incoming.direct_result=32'h7fc00000;
      else if(operand_zero_i) incoming.direct_result=32'h7f800000;
      else if(operand_inf_i) incoming.direct_result=32'b0;
    end else begin
      incoming.exponent=9'(input_exponent);
      if(operand_nan_i) incoming.direct_result=32'h7fc00000;
      else if(operand_zero_i) incoming.direct_result=32'hff800000;
      else if(operand_sign_i) incoming.direct_result=32'h7fc00000;
      else if(operand_inf_i) incoming.direct_result=32'h7f800000;
      incoming.near_one=!incoming.direct && (data_format_i==VXM_FORMAT_BF16?
        (operand32>=32'h3f7e0000 && operand32<=32'h3f810000):
        (operand32>=32'h3f7c0000 && operand32<=32'h3f820000));
      if(incoming.near_one) begin
        incoming.exponent=0;
        if(operand32[30:23]==127)
          incoming.u=log_fixed_to_fp32($signed({25'b0,operand32[22:0]}),23);
        else incoming.u=log_fixed_to_fp32(-$signed(48'd8388608-{25'b0,operand32[22:0]}),24);
      end
    end
    configured=1'b0;
    for(integer bank=0;bank<LUT_BANK_COUNT;bank++)
      if(incoming.bank==bank) configured=lut_configured_i[bank];
    incoming.lookup=!incoming.direct && !incoming.near_one;
    configuration_fault=incoming.valid && incoming.lookup && !configured;
    if(incoming.lookup && !configured) begin
      incoming.direct=1'b1;incoming.lookup=1'b0;incoming.direct_result=32'h7fc00000;
    end
    lut_read_valid_o=q0.valid && q0.lookup;
    lut_read_bank_o=LUT_BANK_WIDTH'(q0.bank);
    lut_read_address_o=LUT_ADDRESS_WIDTH'(q0.address);
  end

  // M0: one physical multiply for all functions and formats.
  always_comb begin
    coefficient_shift=q1.fmt==VXM_FORMAT_BF16?6:(q1.fmt==VXM_FORMAT_FP16?3:0);
    coefficient_bits=15-coefficient_shift;
    active_k=coefficient_rne(lut_read_k_i,coefficient_shift);
    active_b=coefficient_rne(lut_read_b_i,coefficient_shift);
    active_inverse=log_active_inverse(lut_read_log_i[21:0],q1.fmt);
    m0_a=0;m0_b=0;
    if(q1.valid && q1.lookup && lut_read_valid_i) begin
      if(q1.op==VXM_LOCAL_LOG)
        m0_a=active_inverse>>(q1.fmt==VXM_FORMAT_BF16?12:(q1.fmt==VXM_FORMAT_FP16?9:0));
      else m0_a={6'b0,active_k};
      m0_b=q1.residual;
    end
    m0_product=m0_a*m0_b;
    fixed_seed=$signed({32'b0,active_b}) << (q1.fmt==VXM_FORMAT_BF16?7:(q1.fmt==VXM_FORMAT_FP16?10:23));
    fixed_seed=fixed_seed-$signed({8'b0,m0_product});
    retained=0;remainder=0;mask=0;halfway=0;
    if(fixed_seed>=0) begin
      retained=$unsigned(fixed_seed)>>(q1.fmt==VXM_FORMAT_BF16?7:(q1.fmt==VXM_FORMAT_FP16?10:23));
      mask=(48'b1<<(q1.fmt==VXM_FORMAT_BF16?7:(q1.fmt==VXM_FORMAT_FP16?10:23)))-1'b1;
      remainder=$unsigned(fixed_seed)&mask;
      halfway=48'b1<<(q1.fmt==VXM_FORMAT_BF16?6:(q1.fmt==VXM_FORMAT_FP16?9:22));
      if(remainder>halfway || (remainder==halfway && retained[0])) retained=retained+1'b1;
    end
    seed=log_fixed_to_fp32($signed(retained),coefficient_bits);
    if(q1.fmt==VXM_FORMAT_FP16) seed=log_round_format(seed,q1.fmt);
    u_value=log_round_format(log_fixed_to_fp32($signed({8'b0,m0_product}),
      q1.fmt==VXM_FORMAT_BF16?16:(q1.fmt==VXM_FORMAT_FP16?22:44)),q1.fmt);
    if(q1.near_one) u_value=log_round_format(q1.u,q1.fmt);
  end

  // M1/M2/M3 each have one explicit 24x24 core. Rounding tails consume its
  // product; Rsqrt M2 keeps all 48 bits through the fused correction.
  always_comb begin
    negative_quarter_u=scale_pow2(q2.u,-2);
    negative_quarter_u[31]=~q2.u[31];
    negative_half_u=scale_pow2(q2.u,-1);
    negative_half_u[31]=~q2.u[31];
    m1_enable=q3.valid && !q3.direct && q3.fmt==VXM_FORMAT_FP32;
    m1_lhs=0;m1_rhs=0;
    if(m1_enable) begin
      if(q3.op==VXM_LOCAL_LOG) begin m1_lhs=q3.h;m1_rhs=q3.u;end
      else begin m1_lhs=q3.op==VXM_LOCAL_SPECIAL1?q3.y:q3.m;m1_rhs=q3.y;end
    end
    m1_a=m1_enable?{1'b1,m1_lhs[22:0]}:24'b0;
    m1_b=m1_enable?{1'b1,m1_rhs[22:0]}:24'b0;
    m1_product=m1_a*m1_b;
    m1_result=fp32_product_rne(m1_lhs,m1_rhs,m1_product);

    m2_enable=q4.valid && !q4.direct && q4.fmt==VXM_FORMAT_FP32;
    m2_lhs=0;m2_rhs=0;
    if(m2_enable) begin
      if(q4.op==VXM_LOCAL_LOG) begin m2_lhs=q4.h;m2_rhs=q4.u;end
      else if(q4.op==VXM_LOCAL_SPECIAL1) begin m2_lhs=q4.m;m2_rhs=q4.h;end
      else begin m2_lhs=q4.y;m2_rhs=q4.h;end
    end
    m2_a=m2_enable?{1'b1,m2_lhs[22:0]}:24'b0;
    m2_b=m2_enable?{1'b1,m2_rhs[22:0]}:24'b0;
    m2_product=m2_a*m2_b;
    m2_result=fp32_product_rne(m2_lhs,m2_rhs,m2_product);
    rsqrt_correction=c3_rsqrt_correction(q4.m,q4.h,m2_product);
    rsqrt_correction_fault=m2_enable && q4.op==VXM_LOCAL_SPECIAL1 &&
      rsqrt_correction[30:23]==8'hff && |rsqrt_correction[22:0];

    m3_enable=q5.valid && !q5.direct &&
      ((q5.op==VXM_LOCAL_LOG && q5.fmt!=VXM_FORMAT_BF16) ||
       (q5.op==VXM_LOCAL_SPECIAL1 && q5.fmt==VXM_FORMAT_FP32));
    m3_lhs=0;m3_rhs=0;
    if(m3_enable) begin
      m3_lhs=q5.op==VXM_LOCAL_LOG?q5.u:q5.y;m3_rhs=q5.h;
    end
    m3_a=m3_enable?{1'b1,m3_lhs[22:0]}:24'b0;
    m3_b=m3_enable?{1'b1,m3_rhs[22:0]}:24'b0;
    m3_product=m3_a*m3_b;
    m3_result=fp32_product_rne(m3_lhs,m3_rhs,m3_product);
  end

  always_comb begin
    // The exponent-times-ln2 constant path is separate from the four shared
    // variable multipliers; retain Q40 guards until the final conversion.
    log_sum=$signed(q6.exponent)*LOG_LN2_Q40+
      $signed({8'b0,q6.base,12'b0})+log_fp32_to_q40(q6.x);
    restored=scale_pow2(q6.x,integer'($signed(q6.exponent)));
    if(q6.op==VXM_LOCAL_SPECIAL0) restored[31]=q6.sign_bit;
    if(q6.op==VXM_LOCAL_LOG) restored=q6.near_one?q6.x:log_fixed_to_fp32(log_sum,40);
    if(q6.direct) restored=q6.direct_result;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if(!rst_ni) begin
      q0<='0;q1<='0;q2<='0;q3<='0;q4<='0;q5<='0;q6<='0;
      valid_o<=0;result_o<=0;lut_fault_o<=0;
    end else begin
      // Hold payload registers on bubbles; valid bits always advance.
      if(incoming.valid)q0<=incoming;else q0.valid<=0;
      if(q0.valid)q1<=q0;else q1.valid<=0;
      if(q1.valid)q2<=q1;else q2.valid<=0;
      if(q2.valid)q3<=q2;else q3.valid<=0;
      if(q3.valid)q4<=q3;else q4.valid<=0;
      if(q4.valid)q5<=q4;else q5.valid<=0;
      if(q5.valid)q6<=q5;else q6.valid<=0;
      lut_fault_o<=configuration_fault ||
        (q1.valid && q1.lookup && !lut_read_valid_i) ||
        (q1.valid && q1.lookup && lut_read_valid_i && q1.op!=VXM_LOCAL_LOG && fixed_seed<0) ||
        rsqrt_correction_fault;
      if(q1.valid) begin
        q2.y<=seed;q2.x<=seed;q2.u<=u_value;
        q2.base<=q1.lookup && q1.op==VXM_LOCAL_LOG?log_active_base(lut_read_log_i[49:22],q1.fmt):28'b0;
        if(q1.lookup && (!lut_read_valid_i || (q1.op!=VXM_LOCAL_LOG && fixed_seed<0))) begin
          q2.direct<=1'b1;q2.direct_result<=32'h7fc00000;
        end
      end
      if(q2.valid && q2.op==VXM_LOCAL_LOG) begin
        if(q2.fmt==VXM_FORMAT_FP32)
          q3.h<=fp32_add_rne(32'h3eaaaaab,negative_quarter_u);
        else if(q2.fmt==VXM_FORMAT_FP16)
          q3.h<=log_round_format(fp32_add_rne(32'h3f800000,
            log_round_format(negative_half_u,q2.fmt)),q2.fmt);
        else q3.h<=32'h3f800000;
      end
      if(m1_enable) begin
        if(q3.op==VXM_LOCAL_LOG) q4.h<=fp32_add_rne(m1_result,32'hbf000000);
        else if(q3.op==VXM_LOCAL_SPECIAL1) q4.h<=m1_result;
        else q4.h<=fp32_add_rne(32'h40000000,{~m1_result[31],m1_result[30:0]});
      end
      if(m2_enable) begin
        if(q4.op==VXM_LOCAL_LOG) q5.h<=fp32_add_rne(m2_result,32'h3f800000);
        else if(q4.op==VXM_LOCAL_SPECIAL1) begin
          q5.h<=rsqrt_correction;
          if(rsqrt_correction_fault) begin q5.direct<=1'b1;q5.direct_result<=32'h7fc00000;end
        end
        else q5.x<=m2_result;
      end
      if(m3_enable) q6.x<=q5.op==VXM_LOCAL_LOG?log_round_format(m3_result,q5.fmt):m3_result;
      else if(q5.op==VXM_LOCAL_LOG) q6.x<=q5.u;
      valid_o<=q6.valid;
      if(q6.valid) result_o<=log_pack_result(restored,q6.fmt);
    end
  end
  initial if(LUT_ENTRY_COUNT!=64) $error("C3 approximation requires 64-entry LUTs");
endmodule
