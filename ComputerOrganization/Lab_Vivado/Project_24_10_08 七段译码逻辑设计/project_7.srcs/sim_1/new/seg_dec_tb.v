`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 15:42:08
// Design Name: 
// Module Name: seg_dec_tb
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


module seg_dec_tb;
reg[3:0]        a;
wire[6:0]       b;
seg_dec seg_dec(
                .num(a),
                .a_g(b)
    );
initial begin
        a<=0;
   #150 $stop;
end
always #10 a<=a+1;

endmodule
