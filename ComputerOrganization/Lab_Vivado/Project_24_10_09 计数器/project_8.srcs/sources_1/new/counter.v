`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 15:55:01
// Design Name: 
// Module Name: counter
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


module counter(
                clk,//时钟信号
                res,//复位信号
                y//输出
    );
input           clk;
input           res;
output[7:0]     y;//结果是8位的输出

reg[7:0]        y;//在always中对y进行赋值操作 需要将它变成reg型

wire[7:0]       sum;//+1的运算结果
assign          sum=y+1;//组合逻辑部分

always@(posedge clk or negedge res)
if(~res)begin//复位信号到来
  y<=0;
end
else begin//正常工作时候 将sum传给y
  y<=sum;
end

endmodule
