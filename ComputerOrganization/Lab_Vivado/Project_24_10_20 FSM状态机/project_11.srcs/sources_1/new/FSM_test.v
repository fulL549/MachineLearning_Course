`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/17 10:19:10
// Design Name: 
// Module Name: FSM_test
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
module sequence_detector (
    input wire clk,      // 时钟信号
    input wire reset,    // 异步复位信号
    input wire in,       // 输入信号 in
    output reg out       // 输出信号 out
);

localparam      STATE_S=3'd0,
                STATE_A=3'd1,
                STATE_B=3'd2,
                STATE_C=3'd3,
                STATE_D=3'd4;
                
reg[2:0] state,
         nxtState;

    // 状态寄存器：在时钟上升沿或复位信号有效时更新当前状态
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= STATE_S; // 复位时进入 IDLE 状态
        end else begin
            state <= nxtState; // 否则转移到下一状态              
       end
    end
  

    // 下一状态逻辑：根据当前状态和输入信号决定下一状态
    always @(*) begin
        case(state)
            STATE_S: begin
                if (in)
                    nxtState = STATE_A; // 如果在S状态收到 1信号，转到A状态
                else
                    nxtState = STATE_S;        // 否则保持S状态
            end
            STATE_A: begin
                if (in)
                    nxtState = STATE_A;     
                else
                    nxtState = STATE_B;   
            end
            STATE_B: begin
                if (in)
                    nxtState = STATE_C;    
                else
                    nxtState = STATE_S; 
            end
            STATE_C: begin
                if (in)
                    nxtState = STATE_D;   
                else
                    nxtState = STATE_B;  
            end
            STATE_D: begin
                if (in)
                      nxtState = STATE_A;     
                else
                    nxtState = STATE_B;     
            end
            default: begin
                nxtState = STATE_S;             // 默认转回S
            end
        endcase
    end
    
//状态输出
always @(posedge clk or posedge reset) begin
    if(reset) begin
    out<=0;
    end
    else if(nxtState==STATE_D) begin
    out<=1;
    end
    else begin
    out<=0;
    end
end

endmodule
