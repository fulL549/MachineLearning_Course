`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/03 22:54:36
// Design Name: 
// Module Name: s_counter_tb
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


module s_counter_tb;
reg         clk;
reg         res;
wire[3:0]   s_num;
s_counter s_counter(
                .clk(clk),
                .res(res),
                .s_num(s_num)
                    );
                    
initial begin
    clk<=0;
    res<=0;
    #17      res<=1;//Ê±ÖÓ¸´Î»½â³ý
    #20000    $stop;
end                   

always #5 clk<=~clk;         
                    
endmodule
