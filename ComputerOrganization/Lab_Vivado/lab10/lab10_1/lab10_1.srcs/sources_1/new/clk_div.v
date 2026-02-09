`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/27 11:28:16
// Design Name: 
// Module Name: clk_div
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


module clk_div(
    input clk_i,
    input rst_n,
    output reg clk_run
    );
     reg [7:0] count;
    
    always@ (posedge clk_i or negedge rst_n)begin
        if(!rst_n)begin
            count<=5'b0;
        end
        else if(count!=49)
            count<=count+1;
        else
            count<=5'b0;    
    end
    always@(posedge clk_i or negedge rst_n ) begin
	    if(!rst_n)
		    clk_run <= 0;
	    else if(count ==8'd24 | count ==8'd49  )begin
		    clk_run <= ~clk_run;
	    end
	    else
		    clk_run <= clk_run;
        end

    
endmodule
