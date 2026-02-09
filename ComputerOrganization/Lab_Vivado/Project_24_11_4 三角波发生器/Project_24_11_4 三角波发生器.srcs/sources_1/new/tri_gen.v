`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 12:43:12
// Design Name: 
// Module Name: tri_gen
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


module tri_gen(
                clk,
                res,
                d_out
                 );
input           clk;
input           res;
output[8:0]     d_out;                      


reg            state;//主状态机寄存器 只有两个状态
reg[8:0]       d_out;//需要在always里面赋值 定义为reg类型

always@(posedge clk or negedge res)begin
    if(~res) begin
        state<=0;d_out<=0;
    end
    else begin
        case(state)//case语句
        0:begin
            d_out<=d_out+1;
            if(d_out==299) begin
                state<=1;
            end
        end
        1:begin
            d_out<=d_out-1;
            if(d_out==1) begin
                state<=0;
            end
        end
        endcase
    end
end

          
endmodule
