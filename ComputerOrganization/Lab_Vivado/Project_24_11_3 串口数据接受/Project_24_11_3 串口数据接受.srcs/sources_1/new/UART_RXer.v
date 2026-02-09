`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 13:11:52
// Design Name: 
// Module Name: UART_RXer
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


module UART_RXer(
                clk,
                res,
                RX,
                data_out,
                en_data_out
);
input           clk;
input           res;
input           RX;
output[7:0]     data_out;//接受字节输出
output          en_data_out;//输出使能

reg[7:0]        state;//主状态机
reg[12:0]       con;//用于计算比特宽度
reg[3:0]        con_bits;//用于计算比特数
reg             RX_delay;//RX的延时
reg[7:0]        data_out;
reg             en_data_out;
always@(posedge  clk or negedge res)begin
if(~res) begin
    state<=0;con<=0;con_bits<=0;RX_delay<=0;data_out<=0;
end
else begin
    RX_delay<=RX;
    case(state)
        0:begin//等空闲
            if(con==4999)begin//比特率为5000
                con<=0;
            end
            else begin
                con<=con+1;
            end//con不停计数 转12圈
            
            if(con==0) begin
                if(RX) begin
                    con_bits<=con_bits+1;
                end
                else begin
                    con_bits<=0;
                end
            end
            
            if(con_bits==12) begin
                state<=1;
            end
        
        end
        
        1:begin//等起始位
            en_data_out<=0;
            if(~RX&RX_delay) begin
                state<=2;
            end
        end
        2:begin//收最低位b0
            if(con==7500-1)begin//收最低位等1.5个Bit 比特率为5000
                con<=0;
                data_out[0]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end
        
        end
        3:begin//收第一位b1
            if(con==5000-1)begin//收1-7位等1个Bit 比特率为5000
                con<=0;
                data_out[1]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end
        end
        4:begin
            if(con==5000-1)begin//收1-7位等1个Bit 比特率为5000
                con<=0;
                data_out[2]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end        
        end
        5:begin
            if(con==5000-1)begin//收1-7位等1个Bit 比特率为5000
                con<=0;
                data_out[3]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end        
        end        
        6:begin
            if(con==5000-1)begin//收1-7位等1个Bit 比特率为5000
                con<=0;
                data_out[4]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end        
        end
        7:begin
            if(con==5000-1)begin//收1-7位等1个Bit 比特率为5000
                con<=0;
                data_out[5]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end        
        end
        8:begin
            if(con==5000-1)begin//收1-7位等1个Bit 比特率为5000
                con<=0;
                data_out[6]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end        
        end 
        9:begin
            if(con==5000-1)begin//收1-7位等1个Bit 比特率为5000
                con<=0;
                data_out[7]<=RX;
                state<=3;
            end
            else begin
                con<=con+1;
            end        
        end 
        10:begin//字节收完了 产生使能脉冲
            en_data_out<=1;
            state<=1;
        end
        default:begin
            state<=0;
            con<=0;
            con_bits<=0;
            en_data_out<=0;
        end
                          
    endcase        
end
end

endmodule
