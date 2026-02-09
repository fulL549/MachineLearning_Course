`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/11/07 10:58:03
// Design Name: 
// Module Name: mips
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


module mips(
	input				clk_i,
	input 				rst_n,
	input  				pc_run_en_i,
	input  				pc_clr_i,
	output 		[31:0] 	pc_o,
	input 		[31:0] 	instr_i,
	output 				memwrite_o,
	output 		[31:0] 	aluout_o,
	output 		[31:0] 	writedata_o,
	input 		[31:0] 	readdata_i,
	//output              branch,
	//add debug port 
	output  	[10:0]	monitor_list_o,
	input		[4:0] 	ra_debug_i,
	output 		[31:0] 	ra_debug_data_o,
	input      [31:0]  reg_wdata_dbg,
	input wire          reg_wren_dbg
    );
	
	wire				memtoreg;
	wire				alusrc;
	wire				regdst;
	wire				regwrite;
	wire				jump;
	wire				pcsrc;
	wire				zero;
	wire				overflow;
	wire		[2:0] 	alucontrol;
	wire                branch;

	//assign monitor_list_o = {memtoreg, memwrite_o, zero, alusrc, regdst, regwrite, jump, zero, alucontrol};
    assign monitor_list_o = {memtoreg, memwrite_o, pcsrc, alusrc, regdst, regwrite, jump, branch, alucontrol};
	controller u_controller(
		.op_i         (instr_i[31:26]),
		.funct_i      (instr_i[5:0]  ),
		.zero_i       (zero			 ),
		.memtoreg_o   (memtoreg      ),
		.memwrite_o   (memwrite_o    ),
		.pcsrc_o      (pcsrc         ),
		.alusrc_o     (alusrc        ),
		.regdst_o     (regdst        ),
		.regwrite_o   (regwrite      ),
		.jump_o       (jump          ),
		.alucontrol_o (alucontrol    ),
		.branch       (branch        )
	);

	datapath u_datapath(
		.clk_i           (clk_i           ),
		.rst_n           (rst_n           ),
		.pc_run_en_i	 (pc_run_en_i	  ),
		.pc_clr_i 		 (pc_clr_i		  ),
		.memtoreg_i      (memtoreg        ),
		.pcsrc_i         (pcsrc           ),
		.alusrc_i        (alusrc          ),
		.regdst_i        (regdst          ),
		.regwrite_i      (regwrite        ),
		.jump_i          (jump            ),
		.alucontrol_i    (alucontrol      ),
		.overflow_o      (overflow        ),
		.zero_o          (zero            ),
		.pc_o            (pc_o            ),
		.instr_i         (instr_i         ),
		.aluout_o        (aluout_o        ),
		.writedata_o     (writedata_o     ),
		.readdata_i      (readdata_i      ),
		.ra_debug_i      (ra_debug_i      ),
		.ra_debug_data_o (ra_debug_data_o ),
		.reg_wdata_dbg   (reg_wdata_dbg   ),
		.reg_wren_dbg    (reg_wren_dbg    )
	);
	
endmodule
