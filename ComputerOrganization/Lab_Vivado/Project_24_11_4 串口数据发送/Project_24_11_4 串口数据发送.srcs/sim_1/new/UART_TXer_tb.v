`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 14:41:14
// Design Name: 
// Module Name: UART_TXer_tb
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


module UART_TXer_tb;
reg                 clk;
reg                 res;
reg[7:0]            data_in;
reg                 en_data_in;
wire                TX;
wire                rdy;
UART_TXer UART_TXer(//อฌร๛ภýปฏ
                    clk,
                    res,
                    data_in,
                    en_data_in,
                    TX,
                    rdy
);

initial begin
    clk<=0;
    res<=0;
    data_in<=8'h0a;
    en_data_in<=0;
    #17 res<=1;
    #30 en_data_in<=1;
    #10 en_data_in<=0;
    #100000 $stop;
end

always #5 clk=~clk;

endmodule
