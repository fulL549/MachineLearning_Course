`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/10/23 15:21:30
// Design Name: 
// Module Name: maindec
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


module maindec(
	input		[5:0] 	op_i,

	output 				memtoreg_o,
	output 				memwrite_o,
	output				branch_o,
	output				alusrc_o,
	output 				regdst_o,
	output 				regwrite_o,
	output 				jump_o,
	output 		[1:0] 	aluop_o
    );
	//add your code here according to lab 7
	reg memtoreg;
	reg memwrite;
	reg branch;
	reg alusrc;
	reg regdst;
	reg regwrite;
	reg jump;
	reg	[1:0] 	aluop;
	
    always@(*)begin
        case(op_i)
        6'b000000:begin
        regwrite<=1'b1;
        regdst<=1'b1;
        alusrc<=1'b0;
        branch<=1'b0;
        memwrite<=1'b0;
        memtoreg<=1'b0;
        aluop<=2'b10;
        jump<=1'b0;
        end
        
        6'b100011:begin
        regwrite<=1'b1;
        regdst<=1'b0;
        alusrc<=1'b1;
        branch<=1'b0;
        memwrite<=1'b0;
        memtoreg<=1'b1;
        aluop<=2'b00;
        jump<=1'b0;
        end
        
        6'b101011:begin
        regwrite<=1'b0;
        regdst<=1'b0;
        alusrc<=1'b1;
        branch<=1'b0;
        memwrite<=1'b1;
        memtoreg<=1'b0;
        aluop<=2'b00;
        jump<=1'b0;
        end
        
        6'b000100:begin
        regwrite<=1'b0;
        regdst<=1'b0;
        alusrc<=1'b0;
        branch<=1'b1;
        memwrite<=1'b0;
        memtoreg<=1'b0;
        aluop<=2'b01;
        jump<=1'b0;
        end
        
         6'b001000:begin
        regwrite<=1'b1;
        regdst<=1'b0;
        alusrc<=1'b1;
        branch<=1'b0;
        memwrite<=1'b0;
        memtoreg<=1'b0;
        aluop<=2'b00;
        jump<=1'b0;
        end

        
        6'b000010:begin
        regwrite<=1'b0;
        regdst<=1'b0;
        alusrc<=1'b0;
        branch<=1'b0;
        memwrite<=1'b0;
        memtoreg<=1'b0;
        aluop<=2'b00;
        jump<=1'b1;
        end
        
        default:begin
        regwrite<=1'b0;
        regdst<=1'b0;
        alusrc<=1'b0;
        branch<=1'b0;
        memwrite<=1'b0;
        memtoreg<=1'b0;
        aluop<=2'b0;
        jump<=1'b0;
        end
        
        
        endcase
        
    
    end
    
    assign memtoreg_o=memtoreg;
    assign memwrite_o=memwrite;
    assign branch_o=branch;
    assign alusrc_o=alusrc;
    assign regdst_o=regdst;
    assign regwrite_o=regwrite;
    assign jump_o=jump;
    assign aluop_o=aluop;

endmodule
