`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 21:12:27
// Design Name: 
// Module Name: m_gen_tb
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


module m_gen_tb;
reg          clk,res;
wire         y;
m_gen m_gen(
                .clk(clk),
                .res(res),
                .y(y)
    );
initial begin
        clk<=0;res<=0;
    #17 res<=1;
    #600 $stop;
end
always #5 clk<=~clk;
endmodule
