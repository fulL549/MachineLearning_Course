`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 14:30:19
// Design Name: 
// Module Name: fn_sw_4
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


module fn_sw_4(
                a,
                b,
                sel,
                y
                );
input           a;
input           b;
input[1:0]      sel;//两位
output          y;

reg             y;
always@(a or b or y)begin
  case(sel)
      //代码规整性
      2'b00:begin y<=a&b;end
      2'b01:begin y<=a|b;end
      2'b10:begin y<=a^b;end
      2'b11:begin y<=~(a^b);end
  endcase
end
endmodule
