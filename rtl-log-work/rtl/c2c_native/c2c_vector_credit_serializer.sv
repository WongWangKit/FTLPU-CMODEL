`timescale 1ns/1ps

// Vector-level admission control between the completed FIFO and peer flight.
// This is deliberately not a beat-level PHY serializer.  One accepted vector
// holds one credit through serializer occupancy, link flight, and RX-ready
// residence; the paired Receive pop supplies credit_return_i.
module c2c_vector_credit_serializer #(
  parameter integer P_VECTOR_CREDITS = 4,
  parameter integer D_LINK_SERIALIZATION_CYCLES = 1
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic tx_valid_i,
  input  logic [255:0] tx_payload_i,
  output logic tx_pop_o,
  input  logic credit_return_i,
  output logic launch_valid_o,
  output logic [255:0] launch_payload_o,
  output logic [$clog2(P_VECTOR_CREDITS+1)-1:0] credit_count_o,
  output logic serializer_busy_o,
  output logic credit_error_o
);
  localparam integer CREDIT_BITS = $clog2(P_VECTOR_CREDITS + 1);
  localparam integer SERIALIZER_BITS =
    (D_LINK_SERIALIZATION_CYCLES <= 1) ? 1 :
    $clog2(D_LINK_SERIALIZATION_CYCLES);

  logic [CREDIT_BITS-1:0] credit_count_q;
  logic [SERIALIZER_BITS-1:0] serializer_remaining_q;
  logic launch_fire;
  logic invalid_return;

  assign credit_count_o = credit_count_q;
  assign serializer_busy_o = (serializer_remaining_q != '0);
  assign launch_fire = tx_valid_i && (credit_count_q != '0) &&
                       !serializer_busy_o;
  assign tx_pop_o = launch_fire;
  assign launch_valid_o = launch_fire;
  assign launch_payload_o = tx_payload_i;

  // Returning a credit while all credits are already free is a protocol error.
  // A simultaneous launch makes a full-count return legal because the launch
  // consumes that credit at the same edge.
  assign invalid_return = credit_return_i &&
                          (credit_count_q == CREDIT_BITS'(P_VECTOR_CREDITS)) &&
                          !launch_fire;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      credit_count_q <= CREDIT_BITS'(P_VECTOR_CREDITS);
      serializer_remaining_q <= '0;
      credit_error_o <= 1'b0;
    end else begin
      if (launch_fire)
        serializer_remaining_q <=
          SERIALIZER_BITS'(D_LINK_SERIALIZATION_CYCLES - 1);
      else if (serializer_remaining_q != '0)
        serializer_remaining_q <= serializer_remaining_q - 1'b1;

      case ({launch_fire, credit_return_i})
        2'b10: credit_count_q <= credit_count_q - 1'b1;
        2'b01: begin
          if (credit_count_q != CREDIT_BITS'(P_VECTOR_CREDITS))
            credit_count_q <= credit_count_q + 1'b1;
        end
        default: credit_count_q <= credit_count_q;
      endcase

      if (invalid_return)
        credit_error_o <= 1'b1;
    end
  end

`ifndef SYNTHESIS
  initial begin
    if (P_VECTOR_CREDITS <= 0 || D_LINK_SERIALIZATION_CYCLES <= 0)
      $fatal(1, "TEST_FAIL invalid C2C vector credit serializer parameters");
  end

  always @(posedge clk_i) begin
    if (rst_ni && invalid_return)
      $error("TEST_FAIL C2C credit return overflow");
  end
`endif
endmodule
