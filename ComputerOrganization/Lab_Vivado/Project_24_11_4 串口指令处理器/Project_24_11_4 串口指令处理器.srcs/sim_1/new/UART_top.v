`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 15:52:54
// Design Name: 
// Module Name: UART_top
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


module UART_top(
                clk,
                res,
                RX,
                TX
    );
input           clk;
input           res;
input           RX;
input           TX;

wire[7:0]       din_pro;
wire            en_din_pro;
wire[7:0]       dout_pro;
wire            en_dout_pro;
wire            rdy;//中间信号全部设置为wire

YART_RXer UART_RXer(
                .clk(clk),
                .res(res),
                .RX(RX),
                .data_out(din_pro),
                .en_data_out(en_din_pro)
);

UART_TXer UART_TXer(
                    .clk(clk),
                    .res(res),
                    .data_in(dout_pro),
                    .en_data_in(en_dout_pro),
                    .TX(TX),
                    .rdy(rdy)
);

cmd_pro cmd_pro(
            .clk(clk),
            .res(res),
            .din_pro(din_pro),
            .en_din_pro(en_din_pro),
            .dout_pro(dout_pro),
            .en_dout_pro(en_dout_pro),
            .rdy(rdy)
    );
    
endmodule
