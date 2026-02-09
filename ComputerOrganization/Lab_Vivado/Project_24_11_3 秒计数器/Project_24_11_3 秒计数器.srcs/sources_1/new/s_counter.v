`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/03 22:41:41
// Design Name: 
// Module Name: s_counter
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
秒计数器
0-9循环
*/
module s_counter(
                clk,
                res,
                s_num
                    );
input           clk;
input           res;        
output[3:0]     s_num;//保证能够0-9计数
parameter        frequency_clk=24000;//参数设置 //观察不明显24000000->24000

reg[24:0]        con_t;//秒脉冲分频计数器
reg              s_pulse;//秒脉冲尖
reg[3:0]         s_num;//对s_num进行赋值 改为reg类型 触发器

always@(posedge clk or negedge res)begin
    if(~res)begin//复位情况下
        con_t<=0;
        s_pulse<=0;
        s_num<=0;
    end
    else begin
        if(con_t==frequency_clk-1)begin
            con_t<=0;
        end
        else begin
            con_t<=con_t+1;
        end
    end
    //con_t 从0一直数到24M就重新回到0
    
    if(con_t==0) begin
        s_pulse<=1;
    end
    else begin
        s_pulse<=0;
    end
    //s_pulse看到con_t=0就赋值为1 其他为0
    
    if(s_pulse==1) begin
        if(s_num==9) begin
            s_num<=0;
        end
        else begin
            s_num<=s_num+1;
        end
    end
    
end    
  
endmodule
