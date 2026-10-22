module lpu_vxm_alu #(
  // 0: C0/C2 basic-only, 1: C1 with Exp, 2: C3 with Reciprocal/Rsqrt/Log.
  parameter integer SPECIAL_KIND = lpu_pkg::VXM_SPECIAL_NONE,
  parameter integer LUT_BANK_COUNT = 3,
  parameter integer LUT_ENTRY_COUNT = 64,
  parameter integer LUT_BANK_WIDTH =
    LUT_BANK_COUNT <= 1 ? 1 : $clog2(LUT_BANK_COUNT),
  parameter integer LUT_ADDRESS_WIDTH =
    LUT_ENTRY_COUNT <= 1 ? 1 : $clog2(LUT_ENTRY_COUNT)
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        input_valid_i,
  input  logic [1:0]  data_format_i,
  input  logic [3:0]  opcode_i,
  input  logic [31:0] lhs_i,
  input  logic [31:0] rhs_i,

  input  logic [LUT_BANK_COUNT-1:0]    lut_configured_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_input_min_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_segment_width_i,
  output logic                         lut_read_valid_o,
  output logic [LUT_BANK_WIDTH-1:0]    lut_read_bank_o,
  output logic [LUT_ADDRESS_WIDTH-1:0] lut_read_address_o,
  input  logic                         lut_read_valid_i,
  input  logic [15:0]                  lut_read_k_i,
  input  logic [15:0]                  lut_read_b_i,
  input  logic [49:0]                  lut_read_log_i,

  output logic        input_ready_o,
  output logic        output_valid_o,
  output logic [31:0] result_o,
  output logic        illegal_opcode_o,
  output logic        unsupported_format_o,
  output logic        result_collision_o,
  output logic        lut_fault_o
);
  import lpu_pkg::*;

  logic supported_request;
  logic basic_request;
  logic special_request;
  logic opcode_supported;
  logic special_opcode_supported;
  localparam bit C3_PIPELINED=SPECIAL_KIND==lpu_pkg::VXM_SPECIAL_RECIP_RSQRT;
  logic basic_ready;
  logic basic_raw_valid;
  logic [31:0] basic_raw_result;
  logic [5:0] basic_delay_valid_q;
  logic [31:0] basic_delay_result_q[0:5];
  logic basic_valid;
  logic [31:0] basic_result;
  logic basic_collision;
  logic special_ready;
  logic special_valid;
  logic [31:0] special_result;
  logic lhs_format_valid;
  logic lhs_sign;
  logic [7:0] lhs_exponent;
  logic [22:0] lhs_fraction;
  logic lhs_zero;
  logic lhs_inf;
  logic lhs_nan;
  logic rhs_format_valid;
  logic rhs_sign;
  logic [7:0] rhs_exponent;
  logic [22:0] rhs_fraction;
  logic rhs_zero;
  logic rhs_inf;
  logic rhs_nan;

  // One physical classification front end per source operand is shared by
  // every downstream operation.  Opcode decode remains parallel to unpack.
  lpu_vxm_float_unpack u_lhs_unpack (
    .data_format_i,
    .value_i(lhs_i),
    .format_valid_o(lhs_format_valid),
    .sign_o(lhs_sign),
    .exponent_o(lhs_exponent),
    .fraction_o(lhs_fraction),
    .effective_zero_o(lhs_zero),
    .was_subnormal_o(),
    .is_normal_o(),
    .is_inf_o(lhs_inf),
    .is_nan_o(lhs_nan)
  );

  lpu_vxm_float_unpack u_rhs_unpack (
    .data_format_i,
    .value_i(rhs_i),
    .format_valid_o(rhs_format_valid),
    .sign_o(rhs_sign),
    .exponent_o(rhs_exponent),
    .fraction_o(rhs_fraction),
    .effective_zero_o(rhs_zero),
    .was_subnormal_o(),
    .is_normal_o(),
    .is_inf_o(rhs_inf),
    .is_nan_o(rhs_nan)
  );

  always_comb begin
    special_opcode_supported =
      ((SPECIAL_KIND == VXM_SPECIAL_EXP) &&
       (opcode_i == VXM_LOCAL_SPECIAL0)) ||
      ((SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) &&
       ((opcode_i == VXM_LOCAL_SPECIAL0) ||
        (opcode_i == VXM_LOCAL_SPECIAL1) ||
        (opcode_i == VXM_LOCAL_LOG)));
    opcode_supported = (opcode_i <= VXM_LOCAL_MAX) ||
      special_opcode_supported;
    unsupported_format_o = input_valid_i &&
      (!lhs_format_valid || !rhs_format_valid);
    illegal_opcode_o = input_valid_i && lhs_format_valid &&
      rhs_format_valid && !opcode_supported;
    supported_request = input_valid_i && !unsupported_format_o &&
      !illegal_opcode_o;
    basic_request = supported_request && (opcode_i <= VXM_LOCAL_MAX);
    special_request = supported_request && special_opcode_supported;
    if (basic_request)
      input_ready_o = basic_ready;
    else if (special_request)
      input_ready_o = special_ready;
    else
      input_ready_o = 1'b1;

    output_valid_o = basic_valid || special_valid;
    result_o = 32'b0;
    if (special_valid)
      result_o = special_result;
    if (basic_valid)
      result_o = basic_result;
    result_collision_o = basic_collision || (basic_valid && special_valid);
  end

  lpu_vxm_basic_alu #(.UNIFORM_TWO_CYCLE(C3_PIPELINED)) u_basic (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .valid_i(basic_request),
    .data_format_i(data_format_i),
    .opcode_i(opcode_i[2:0]),
    .lhs_i(lhs_i),
    .rhs_i(rhs_i),
    .lhs_format_valid_i(lhs_format_valid),
    .lhs_sign_i(lhs_sign),
    .lhs_exponent_i(lhs_exponent),
    .lhs_fraction_i(lhs_fraction),
    .lhs_zero_i(lhs_zero),
    .lhs_inf_i(lhs_inf),
    .lhs_nan_i(lhs_nan),
    .rhs_format_valid_i(rhs_format_valid),
    .rhs_sign_i(rhs_sign),
    .rhs_exponent_i(rhs_exponent),
    .rhs_fraction_i(rhs_fraction),
    .rhs_zero_i(rhs_zero),
    .rhs_inf_i(rhs_inf),
    .rhs_nan_i(rhs_nan),
    .ready_o(basic_ready),
    .valid_o(basic_raw_valid),
    .result_o(basic_raw_result),
    .illegal_opcode_o(),
    .result_collision_o(basic_collision)
  );

  generate if(C3_PIPELINED) begin:g_basic_align
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if(!rst_ni) begin
        basic_delay_valid_q<='0;
        for(integer i=0;i<6;i++) basic_delay_result_q[i]<=0;
      end else begin
        basic_delay_valid_q<={basic_delay_valid_q[4:0],basic_raw_valid};
        if(basic_raw_valid) basic_delay_result_q[0]<=basic_raw_result;
        for(integer i=1;i<6;i++)
          if(basic_delay_valid_q[i-1]) basic_delay_result_q[i]<=basic_delay_result_q[i-1];
      end
    end
    assign basic_valid=basic_delay_valid_q[5];
    assign basic_result=basic_delay_result_q[5];
  end else begin:g_basic_direct
    assign basic_valid=basic_raw_valid;
    assign basic_result=basic_raw_result;
  end endgenerate

  lpu_vxm_special_alu #(
    .SPECIAL_KIND(SPECIAL_KIND),
    .LUT_BANK_COUNT(LUT_BANK_COUNT),
    .LUT_ENTRY_COUNT(LUT_ENTRY_COUNT),
    .LUT_BANK_WIDTH(LUT_BANK_WIDTH),
    .LUT_ADDRESS_WIDTH(LUT_ADDRESS_WIDTH)
  ) u_special (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .valid_i(special_request),
    .data_format_i(data_format_i),
    .opcode_i(opcode_i),
    .operand_format_valid_i(lhs_format_valid),
    .operand_sign_i(lhs_sign),
    .operand_exponent_i(lhs_exponent),
    .operand_fraction_i(lhs_fraction),
    .operand_zero_i(lhs_zero),
    .operand_inf_i(lhs_inf),
    .operand_nan_i(lhs_nan),
    .lut_configured_i,
    .lut_input_min_i,
    .lut_segment_width_i,
    .lut_read_valid_o,
    .lut_read_bank_o,
    .lut_read_address_o,
    .lut_read_valid_i,
    .lut_read_k_i,
    .lut_read_b_i,
    .lut_read_log_i,
    .ready_o(special_ready),
    .valid_o(special_valid),
    .result_o(special_result),
    .illegal_opcode_o(),
    .lut_fault_o
  );
endmodule
