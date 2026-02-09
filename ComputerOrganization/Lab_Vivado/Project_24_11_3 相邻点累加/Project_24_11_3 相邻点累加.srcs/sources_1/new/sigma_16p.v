`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/03 23:20:22
// Design Name: 
// Module Name: sigma_16p
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

/*
相邻16点累加
*/
module sigma_16p(
                clk,
                res,
                date_in,
                syn_in,
                date_out,
                syn_out
                 );  
input            clk;
input            res;
input[7:0]       date_in;//采样信号
input            syn_in;//采样时钟
output[7:0]      date_out;//累加结果输出
output           syn_out;//累加结果同步脉冲

reg              syn_in_n1;//syn_in的方向延时
wire             syn_pulse;//采样时钟上升沿识别脉冲
assign           syn_pulse=syn_in & syn_in_n1;
reg[3:0]         con_syn;//脉冲尖循环计数 在0-15之间循环

wire[7:0]        comp_8;//补码
wire[11:0]       d_12;//升位结果
assign            comp_8=date_in[7]?{date_in[7],~date_in[6:0]+1}:date_in;//补码运算
assign            d_12={date_in[7],date_in[7],date_in[7],date_in[7],date_in};//升位 升了4个符号位

reg[11:0]         sigma;//累加计算
reg[11:0]         date_out;
reg               syn_out;
always@(posedge clk or negedge res)
if(~res) begin
        syn_in_n1<=0;
        con_syn<=0;
        sigma<=0;
        syn_out<=0;
    end
    else begin
        syn_in_n1<=~syn_in;
        
        if(syn_pulse==1) begin
            con_syn<=con_syn+1;
        end
        
        if(syn_pulse==1) begin
            if(con_syn==15) begin
                sigma<=d_12;
                date_out<=sigma;
                syn_out<=1;
            end
            else begin
                sigma<=sigma+d_12;
            end
        end        
        else begin
            syn_out<=0;
        end
    end


endmodule
