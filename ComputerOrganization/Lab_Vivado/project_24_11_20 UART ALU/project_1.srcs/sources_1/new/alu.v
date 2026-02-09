`timescale 1ns / 1ps

module alu(
    input   wire [31:0] A_i,      // Operand A
    input   wire [31:0] B_i,      // Operand B
    input   wire [2:0]  op_i,     // Operation code
    output  reg  [31:0] result_o, // Result
    output  reg         Z,        // Zero flag
    output  reg         N,        // Negative flag
    output  reg         C,        // Carry flag
    output  reg         V         // Overflow flag
);

    always @(*) begin
        // Initialize outputs
        result_o = 32'b0;
        Z = 1'b0;
        N = 1'b0;
        C = 1'b0;
        V = 1'b0;

        case (op_i)
            3'b000: begin // Addition
                {C, result_o} = A_i + B_i; // Capture carry out
                V = (A_i[31] & B_i[31] & ~result_o[31]) | (~A_i[31] & ~B_i[31] & result_o[31]); // Overflow
            end
            3'b001: begin // Subtraction
                {C, result_o} = A_i - B_i; // Capture carry out
                V = (A_i[31] & ~B_i[31] & ~result_o[31]) | (~A_i[31] & B_i[31] & result_o[31]); // Overflow
            end
            3'b010: begin // AND
                result_o = A_i & B_i;
            end
            3'b011: begin // OR
                result_o = A_i | B_i;
            end
            3'b101: begin // ~
                result_o = A_i | B_i;
            end
            3'b101: begin // SLT (Set Less Than)
                result_o = (A_i < B_i) ? 32'b1 : 32'b0;
            end
            default: begin // Default case: do nothing
                result_o = 32'b0;
            end
        endcase

        // Set flags
        Z = (result_o == 32'b0);     // Zero flag
        N = result_o[31];            // Negative flag
    end

endmodule

