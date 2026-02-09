`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 10:43:50
// Design Name: 
// Module Name: tb
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


module fn_sw_tb;

reg a,b,sel;//输入定义为reg
wire y;//输出定义为wire

fn_sw fn_sw(//匿名例化方式
.a(a),
.b(b),
.sel(sel),
.y(y)
    );

initial begin
      a<=0;b<=0;sel<=0;//0s时
  #10 a<=0;b<=0;sel<=1;//10ns后
  #10 a<=0;b<=1;sel<=0;//10ns后
  #10 a<=0;b<=1;sel<=1;//10ns后
  #10 a<=1;b<=0;sel<=0;//10ns后
  #10 a<=1;b<=0;sel<=1;//10ns后
  #10 a<=1;b<=1;sel<=0;//10ns后
  #10 a<=1;b<=1;sel<=1;//10ns后
  #10 $stop;//verilog系统任务
  end
  
endmodule
