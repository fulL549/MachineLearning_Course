`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 10:19:12
// Design Name: 
// Module Name: test
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
//二选一组合逻辑
module fn_sw(//写入端口
a,
b,
sel,
y
    );
input a;//定义端口属性
input b;
input sel;
output y;

//定义y
//方法一：三目操作符
//assign y=sel?(a^b):(a&b);

//方法二:使用always
reg y;//将y定义为reg型
always@(a or b or y)begin
  if(sel==1) begin//sel==1
    y<=a^b;
    end
    else begin
    y<=a&b;
    end
    end
endmodule