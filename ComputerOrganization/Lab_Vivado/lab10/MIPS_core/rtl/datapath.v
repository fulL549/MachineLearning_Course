`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/11/02 15:12:22
// Design Name: 
// Module Name: datapath
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


module datapath(
	input				clk_i,
	input				rst_n,

	input  				pc_run_en_i,
	input  				pc_clr_i,

	input 				memtoreg_i,
	input 				pcsrc_i, 
	input				alusrc_i,
	input				regdst_i,
	input				regwrite_i,
	input				jump_i,
	input		[2:0] 	alucontrol_i,
	output				overflow_o,
	output				zero_o,
	output		[31:0] 	pc_o,
	input		[31:0] 	instr_i,
	output		[31:0] 	aluout_o,
	output		[31:0] 	writedata_o,
	input		[31:0] 	readdata_i,
	//
	input		[4:0] 	ra_debug_i,
	output		[31:0] 	ra_debug_data_o,
	input      [31:0]  reg_wdata_dbg,
	input wire          reg_wren_dbg
    );
	
	//add your code here
	reg[31:0] next_pc;		
	

	PC u_pc(clk_i,rst_n,pcsrc_i,jump_i,next_pc,pc_run_en_i,pc_o);
	wire[31:0] rd1,rd2;
	 
	//reg ra1,ra2,wa3,wd3,src_a,src_b;
	wire [4:0] ra1,ra2,wa3;
	wire [31:0] wd3,src_a,src_b;
	regfile u_regfile(clk_i,regwrite_i,reg_wren_dbg,ra1,ra2,ra_debug_i,wa3,wd3,ra_debug_i,reg_wdata_dbg,rd1,rd2,ra_debug_data_o);
	alu u_alu(src_a,src_b,alucontrol_i,aluout_o,overflow_o,zero_o);
	//mux2 #(32) u_mux2_1(rd2,aluout,memtoreg,writedata);
	//mux2 #(5) u_mux2_2(rd2,wa3,regdst,wa3);
	//reg signext_i,sl2_i;
	reg [15:0] signext_i;
	reg [31:0] sl2_i;
	
	//wire signext_o,sl2_o;
	wire [31:0] signext_o,sl2_o;
	
	signext u_signext(signext_i,signext_o);
	sl2 u_sl2(sl2_i,sl2_o);
	wire[31:0] pcplus;
	assign pcplus=pc_o+4;
	always @(*) begin
		case(instr_i[31:26])
			6'b000010:
				next_pc = {pcplus[31:28],instr_i[25:0],2'b00};
			6'b000100: begin
				signext_i = instr_i[15:0];
				sl2_i = signext_o; 
				next_pc = pcplus+sl2_o;
			end
		endcase
	end
	wire [15:0]signext_i_2;
	wire [31:0]signext_o_2;
	assign signext_i_2 = instr_i[15:0];
	signext u_signext_2(signext_i_2,signext_o_2);
	assign ra1 = instr_i[25:21];
	assign src_a = rd1;
	assign src_b = (alusrc_i) ? signext_o_2 : rd2;
	assign wa3 = regdst_i ? instr_i[15:11]: instr_i[20:16];
	assign wd3 = (memtoreg_i) ? readdata_i : aluout_o;
	assign writedata_o = rd2;
	assign ra2 = instr_i[20:16];

	
	

endmodule

module PC(
    input               clk_i,      
    input               rst_i,      
    input               pc_src_i,  
    input               jump_i,     
    input [31:0]        next_pc_i,  
	input  		    run_en_i,   
    output reg [31:0]   pc_o
);


always @(negedge clk_i or posedge rst_i) begin
    if (rst_i) begin  
        pc_o <= 32'b0; 
	
	end else if(!run_en_i) begin
		pc_o <= pc_o;

    end else begin
        if (jump_i) begin
            pc_o <= next_pc_i;
        end else if (pc_src_i) begin
            pc_o <= next_pc_i;
        end else begin
            pc_o <= pc_o + 32'd4;
        end
    end
end
	

endmodule
