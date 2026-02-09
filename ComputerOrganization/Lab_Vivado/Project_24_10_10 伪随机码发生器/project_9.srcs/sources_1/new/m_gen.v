`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 21:04:32
// Design Name: 
// Module Name: m_gen
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

//四级伪随机码发生器
module m_gen(
                clk,
                res,//时序逻辑电路一定有clk和res
                y//输出
    );
input           clk;
input           res;
output          y;

reg[3:0]        d;//定义四位的触发器
assign          y=d[0];

always@(posedge clk or negedge res)
if(~res) begin
    d<=4'b1111;//不能为全0
end
else begin
    d[2:0]<=d[3:1];
    d[3]=d[3]+d[0];//模2加==加法时高位溢出，自动舍去进位
end
endmodule
