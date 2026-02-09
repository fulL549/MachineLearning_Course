`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 14:25:05
// Design Name: 
// Module Name: UART_TXer
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


module UART_TXer(
                    clk,
                    res,
                    data_in,
                    en_data_in,
                    TX,
                    rdy
);
input              clk;
input              res;
input[7:0]         data_in;//准备发送的数据
input              en_data_in;//发送使能
output             TX;
output             rdy;//空闲标志 0表示空闲

reg[3:0]            state;//主状态机寄存器
reg[9:0]            send_buf;//发送给寄存器
assign             TX=send_buf[0];//链接TX

reg[9:0]            send_flag;//用于判断右移结束

reg[12:0]           con;//用于计算波特周期
reg                 rdy;//rdy表示是否空闲 在忙时不接受新的信号

always@(posedge  clk or negedge res)
if(~res) begin
    state<=0;
    send_buf=1;//TX空闲时候为1
    send_flag<=0;
    con<=0;
    rdy<=0;
end
else begin
    case(state)
    0:begin//等待使能信号
        if(en_data_in) begin
            send_buf={1'b1,data_in,1'b0};
            rdy<=1;       
            send_flag<=10'b10_0000_0000;//_可以省略
            state<=1; 
        end
    end
    1: begin //串口发送寄存器右移
        if(con==5000-1) begin
            con<=0;
        end
        else begin
            con<=con+1;
        end
        
        if(con==0)begin
            send_buf[8:0]=send_buf[9:1];
            send_flag[8:0]=send_flag[9:1];
        end
        
        if(send_flag[0]==1)begin
            rdy<=0;
            state<=0;
        end
    end
    endcase
end


endmodule
