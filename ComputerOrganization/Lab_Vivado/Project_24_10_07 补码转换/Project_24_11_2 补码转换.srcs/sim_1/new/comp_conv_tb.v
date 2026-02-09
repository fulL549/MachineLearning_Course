`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/02 21:31:06
// Design Name: 
// Module Name: comp_conv_tb
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


module comp_conv_tb;
reg[7:0]        a;
wire[7:0]       a_comp;
comp_conv comp_conv(
                .a(a),
                .a_comp(a_comp)
                    );

initial begin
        a<=-15;
        #30 $stop;
end                 

always #1 a<=a+1;           
                 


endmodule
