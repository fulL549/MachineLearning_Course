`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2017/10/23 15:27:24
// Design Name: 
// Module Name: aludec
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


module aludec(
	input		[5:0] 	funct_i,
	input		[1:0] 	aluop_i,
	output reg	[2:0] 	alucontrol_o
    );
	always@(*)begin
	    case(aluop_i)
	        2'b00:begin
	            alucontrol_o<=3'b010;
	        end 
	        2'b01:begin
	            alucontrol_o<=3'b110;
	        end
	        2'b10:begin
	            case(funct_i)
	            6'b000000:alucontrol_o<=3'b100;
	            6'b000010:alucontrol_o<=3'b101;
	            6'b100000:alucontrol_o<=3'b010;
	            6'b100010:alucontrol_o<=3'b110;
	            6'b100100:alucontrol_o<=3'b000;
	            6'b100101:alucontrol_o<=3'b001;
	            6'b101010:alucontrol_o<=3'b111;
	            default:
	            alucontrol_o<=3'b0;
	            endcase
	        end
	        default:
	            alucontrol_o<=3'b0;
	    endcase
	end
endmodule
