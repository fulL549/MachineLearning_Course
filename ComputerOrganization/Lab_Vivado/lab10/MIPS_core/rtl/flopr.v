`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/11/02 14:32:42
// Design Name: 
// Module Name: flopr
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


module flopr #(parameter WIDTH = 8)(
	input						clk_i,
	input						rst_n,
	input		[WIDTH-1:0] 	d,
	output	reg	[WIDTH-1:0] 	q
    );
	always @(negedge clk_i or negedge rst_n) begin
		if(~rst_n) begin
			q <= 0;
		end else begin
			q <= d;
		end
	end
endmodule
