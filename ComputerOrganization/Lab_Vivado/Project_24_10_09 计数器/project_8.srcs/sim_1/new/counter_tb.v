`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 16:00:15
// Design Name: 
// Module Name: counter_tb
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


module counter_tb;
reg         clk,res;
wire[7:0]   y;
counter counter(
                .clk(clk),
                .res(res),
                .y(y)
    );
    
initial begin
        clk<=0;res<=0;//一开始复位信号初始化为0
   #17  res<=1;//把复位信号释放
   #6000 $stop;
end

always #5 clk<=~clk;//5ns反转一次 则周期为10ns
    
endmodule
