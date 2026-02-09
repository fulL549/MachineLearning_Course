`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/11/07 13:54:42
// Design Name: 
// Module Name: testbench
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


module testbench();

	localparam 			period = 10;

	reg 				clk;
	reg 				rst_n;

	reg 				pc_run_en;
	reg 				pc_clr;

	reg  				pc_run_hold;

	wire 	[31:0]		pc;
	wire 				pc_ov;

	wire  	[31:0]		instr_num;

	wire   	[10:0]		monitor_list;
	wire 	[31:0]		instr;

	assign instr_num = 32'h12;

	top dut(
		.clk_i             (clk               ),
		.rst_n             (rst_n             ),
		.pc_run_en_i       (pc_run_hold),
		.pc_clr_i          (pc_clr),
		.pc_o              (pc),
		.monitor_list_o    (monitor_list),
		.instr_o           (instr),
		.ra_debug_i        (5'h0),
		.ra_debug_data_o   (),
		.reg_wren_dbg      (1'b0    ),
		.uart_reg_i        (32'h80018003),
		.mem_debug_rden_i  (1'b0),
		.mem_debug_addr_i  (14'h0),
		.mem_debug_rdata_o ()
	);
	

	initial begin 
		clk = 1'b1;
		rst_n = 1'b1;
		pc_run_en = 1'b0;
		pc_clr = 1'b0;
		#201;
		rst_n = 1'b0;

		#(period*2)
		pc_run_en = 1'b1;

		#(period*20)
		pc_run_en = 1'b0;
		pc_clr = 1'b1;

		#(period)
		pc_clr = 1'b0;

	end

	always #(period/2) clk = ~clk;

	assign pc_ov = (pc[31:2] == instr_num[29:0]);

	always @(posedge clk or negedge rst_n) begin
        if(rst_n) begin
            pc_run_hold	<= 'b0;
        end else begin
            casex({pc_ov, pc_run_en})
                2'b00:
                    pc_run_hold <= pc_run_hold;
                2'b01:
                    pc_run_hold	<= 1'b1;
                2'b1x:
                    pc_run_hold	<= 1'b0;
            endcase
        end
    end


endmodule
