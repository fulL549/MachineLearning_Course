`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 16:06:00
// Design Name: 
// Module Name: UART_top_tb
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


module UART_top_tb;
reg         clk;
reg         res;
wire         RX;
wire        TX;
reg[45:0]    RX_send;
assign      RX=RX_send[0];

reg[12:0]   con;

UART_top UART_top(
                clk,
                res,
                RX,
                TX
    );
    
initial begin
    clk<=0;
    res<=0;
    RX_send<={1'b1,8'h09,1'b0,1'b1,8'h06,1'b0,1'b1,8'h0a,1'b0,16'hffff};//操作码：+ 操作数1：6 操作数2：9 预期结果：15
    con<=0;
    #17 res<=1;
    #40000000 $stop;
end

always #5 clk=~clk;

always@(posedge clk)begin//利用clk触发模块
    if(con==5000-1)begin
        con<=0;
    end
    else begin
        con<=con+1;
    end
    
    if(con==0) begin
        RX_send[44:0]<=RX_send[45:1];//右移动
        RX_send[45]<=RX_send[0];
    end
end
    
endmodule
