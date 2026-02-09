`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 12:51:17
// Design Name: 
// Module Name: tri_gen_tb
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


module tri_gen_tb;
reg         clk;
reg         res;
wire[8:0]   d_out;
tri_gen tri_gen(
                .clk(clk),
                .res(res),
                .d_out(d_out)
                 );

initial begin
    clk<=0;
    res<=0;
    #17 res<=1;
    #10000 $stop;
end                 

always #1 clk<=~clk;
      
endmodule
