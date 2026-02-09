`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/10/23 15:21:30
// Design Name: 
// Module Name: controller
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


module controller(
	input		[5:0] 	op_i,
	input		[5:0] 	funct_i,
	input 				zero_i,
	output				memtoreg_o,
	output				memwrite_o,
	output				pcsrc_o,
	output				alusrc_o,
	output				regdst_o,
	output				regwrite_o,
	output				jump_o,
	output		[2:0] 	alucontrol_o,
	output              branch
    );

	wire		[1:0] 	aluop;
	//wire 				branch;

	maindec u_maindec(
		.op_i       (op_i       ),
		.memtoreg_o (memtoreg_o ),
		.memwrite_o (memwrite_o ),
		.branch_o   (branch     ),
		.alusrc_o   (alusrc_o   ),
		.regdst_o   (regdst_o   ),
		.regwrite_o (regwrite_o ),
		.jump_o     (jump_o     ),
		.aluop_o    (aluop      )
	);
	
	aludec u_aludec(
		.funct_i    	(funct_i      ),
		.aluop_i    	(aluop        ),
		.alucontrol_o	(alucontrol_o )
	);

	assign pcsrc_o = branch & zero_i;
endmodule
