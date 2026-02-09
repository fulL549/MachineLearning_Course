`timescale 1ns / 1ps
//`default_nettype none
//////////////////////////////////////////////////////////////////////////////////
// Company: SYSU
// Engineer: Kafuuchino
// 
// Create Date: 2024/11/18 17:25:17
// Design Name: 
// Module Name: alu
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module alu(

    input       [31:0]              A_i,
    input       [31:0]              B_i,
    input       [2:0]               op_i,

    output      [31:0]              result_o,
    output  reg                     overflow_o,
    output                          zero_o

);
    wire[31:0] s,bout;
    reg[31:0] y;
    assign result_o=y;
	assign bout = op_i[2] ? ~B_i : B_i;
	assign s = A_i + bout + op_i[2];
	always @(*) begin
		case (op_i[2:0])
			3'b000: y <= A_i & bout;
			3'b001: y <= A_i | bout;
			3'b010: y <= s;
			3'b110: y <= s;
			3'b111: y <= s[31];
			3'b100: y <=A_i<<B_i;
			3'b101: y <=A_i>>B_i;
			default : y <= 32'b0;
		endcase	
	end
	assign zero_o = (y == 32'b0);

	always @(*) begin
		case (op_i[2:1])
			2'b01:overflow_o <= A_i[31] & B_i[31] & ~s[31] |
							~A_i[31] & ~B_i[31] & s[31];
			2'b11:overflow_o <= ~A_i[31] & B_i[31] & s[31] |
							A_i[31] & ~B_i[31] & ~s[31];
			default : overflow_o <= 1'b0;
		endcase	
	end
    

endmodule
