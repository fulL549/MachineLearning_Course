`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/11/07 13:50:53
// Design Name: 
// Module Name: top
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


module top(
	input				clk_i,
	input  				rst_n,

	// pc run enable
	input  				pc_run_en_i,
	input  				pc_clr_i,

	output  	[31:0]	pc_o,

	// dbg connector
	output  	[10:0]	monitor_list_o,
	output  	[31:0]	instr_o,

	input		[4:0] 	ra_debug_i,
	output 		[31:0] 	ra_debug_data_o,
	input      [31:0]   reg_wdata_dbg,
	input wire          reg_wren_dbg,

    input       [31:0]  uart_reg_i,

    input               mem_debug_rden_i,
    input       [14:0]  mem_debug_addr_i,
    output      [31:0]  mem_debug_rdata_o
    );

	wire		[31:0] 	pc;
	wire		[31:0] 	instr;

	wire            	memwrite;
    wire    	[14:0]  memaddr;
    wire    	[31:0]  memwdata;
    wire    	[31:0]  memrdata;

	wire    	[14:0]  memaddr_actual;
    wire    	[31:0]  memwdata_actual;
    wire    	[31:0]  memrdata_actual;

	wire  		[31:0]	writedata;
	wire  		[31:0]	readdata;
	wire  		[31:0]	dataaddr;

	mips u_mips(
		.clk_i           (clk_i           ),
		.rst_n           (rst_n           ),
		.pc_run_en_i	 (pc_run_en_i	  ),
		.pc_clr_i 		 (pc_clr_i		  ),
		.pc_o            (pc              ),
		.instr_i         (instr           ),
		.memwrite_o      (memwrite        ),
		.aluout_o        (dataaddr        ),
		.writedata_o     (writedata       ),
		.readdata_i      (readdata        ),
		.monitor_list_o  (monitor_list_o  ),
		.ra_debug_i      (ra_debug_i      ),
		.ra_debug_data_o (ra_debug_data_o ),
		.reg_wdata_dbg   (reg_wdata_dbg   ),
		.reg_wren_dbg    (reg_wren_dbg    )
	);
	
	inst_rom u_inst_rom (
        .clka(clk_i),    // input wire clka
        .ena(1'b1),      // input wire ena
        .addra(pc[9:2]),  // input wire [3 : 0] addra
        .douta(instr)  // output wire [31 : 0] douta
    );

	data_ram u_data_ram (
        .clka(~clk_i),    // input wire clka
        .ena(1'b1),      // input wire ena
        .wea(memwrite),      // input wire [0 : 0] wea
        .addra(memaddr_actual[13:0]),  // input wire [13 : 0] addra
        .dina(memwdata_actual),    // input wire [31 : 0] dina
        .douta(memrdata)  // output wire [31 : 0] douta
    );

    assign memaddr_actual = mem_debug_rden_i ? mem_debug_addr_i : memaddr;
    assign memwdata_actual = writedata;
	assign memrdata_actual = (memaddr_actual[14]) ? uart_reg_i : memrdata;

    assign memwdata = writedata;
    assign memaddr = dataaddr[14:0];
    assign mem_debug_rdata_o = memrdata_actual;
	assign readdata = memrdata_actual;


	assign pc_o = pc;
	assign instr_o = instr;

endmodule
