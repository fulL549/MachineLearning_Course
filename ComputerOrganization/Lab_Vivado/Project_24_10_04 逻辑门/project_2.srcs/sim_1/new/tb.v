`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 09:34:04
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


module inv_tb;
reg aa;
wire yy;
inv inv(
.A(aa),
.Y(yy)
        );
initial begin
aa<=0;
#10 aa<=1;
#10 aa<=0;
#10 aa<=1;
#10 $stop;
end
endmodule
