`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/15 14:51:23
// Design Name: 
// Module Name: fn_sw_4_tb
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


module fn_sw_4_tb;
reg[3:0]        absel;
wire            y;        
fn_sw_4 fn_sw_4(
                .a(absel[0]),//第一位给a
                .b(absel[1]),//第二位给b
                .sel(absel[3:2]),//2 3两高位给sel
                .y(y)
                );
initial begin
            absel<=0;
    #200    $stop;
end          
always #10 absel<=absel+1;
    
endmodule
