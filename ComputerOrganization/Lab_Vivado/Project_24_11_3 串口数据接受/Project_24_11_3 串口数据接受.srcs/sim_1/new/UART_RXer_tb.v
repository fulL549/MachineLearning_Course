`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 13:39:04
// Design Name: 
// Module Name: UART_RXer_tb
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


module UART_RXer_tb;
reg             clk;
reg             res;
wire            RX;
wire[7:0]       data_out;
wire            en_data_out;
reg[25:0]       RX_send;//里面装有串口字节发送数据
assign          RX=RX_send[0];//链接RX

reg[12:0]       con;

UART_RXer UART_RXer(
                .clk(clk),
                .res(res),
                .RX(RX),
                .data_out(data_out),
                .en_data_out(en_data_out)
);

initial begin
    clk<=0;
    res<=0;
    RX_send<={1'b1,8'haa,1'b0,16'hffff};//利用RX_send的最低位作为发送端 依次右移
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
        RX_send[24:0]<=RX_send[25:1];//右移动
        RX_send[25]<=RX_send[0];
    end
end

endmodule
