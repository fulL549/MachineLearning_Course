`timescale 1ns / 1ps
//`default_nettype none
//////////////////////////////////////////////////////////////////////////////////
// Company: SYSU
// Engineer: Kafuuchino
// 
// Create Date: 2024/11/24 20:42:28
// Design Name: 
// Module Name: regfile
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


module regfile(
    input   wire            clk_i,
	input   wire            we3_i,
	input   wire            we_dbg_i,
	input   wire    [4:0]   ra1_i,
    input   wire    [4:0]   ra2_i,
    input   wire    [4:0]   ra_dbg_i,
    input   wire    [4:0]   wa3_i,
	input   wire    [31:0]  wd3_i,
	input   wire    [4:0]   wa_dbg_i,
	input   wire    [31:0]  wd_dbg_i,
	output  wire    [31:0]  rd1_o,
    output  wire    [31:0]  rd2_o,
	output  wire    [31:0]  rd_dbg_o
);
	reg             [31:0]  rf[31:0];
	wire  			[4:0]	wr_addr;
	wire  			[31:0]	wr_data;
	assign wr_addr = we_dbg_i ? wa_dbg_i : wa3_i;
	assign wr_data = we_dbg_i ? wd_dbg_i : wd3_i;
	always @(posedge clk_i) begin
		if(we3_i | we_dbg_i) begin
		    rf[wr_addr] <= wr_data;
		end
	end
	assign rd1_o = (ra1_i != 0) ? rf[ra1_i] : 0;
	assign rd2_o = (ra2_i != 0) ? rf[ra2_i] : 0;
	assign rd_dbg_o = (ra_dbg_i != 0) ? rf[ra_dbg_i] : 0;
endmodule
