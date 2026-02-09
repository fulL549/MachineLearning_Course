`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 15:29:13
// Design Name: 
// Module Name: cmd_pro
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


module cmd_pro(
            clk,
            res,
            din_pro,
            en_din_pro,
            dout_pro,
            en_dout_pro,
            rdy
    );
input       clk;
input       res;
input[7:0]  din_pro;//指令和数据输入端口
input       en_din_pro;//输入使能
output[7:0] dout_pro;//指令执行结果
output      en_dout_pro;//指令输出使能
output      rdy;//串口发送空闲标志 0表示空闲

parameter  add_ab=8'h0a;
parameter  sub_ab=8'h0b;
parameter  and_ab=8'h0c;
parameter  or_ab=8'h0d;
reg[3:0]    state;//主状态机寄存器
reg[7:0]    cmd_reg,A_reg,B_reg;//存放指令、A和B
reg[7:0]    dout_pro;
reg         en_dout_pro;

always@(posedge clk or negedge res)
if(~res) begin
    state<=0;cmd_reg<=0;A_reg<=0;B_reg<=0;dout_pro<=0;en_dout_pro<=0;
end
else begin
    case(state)
    0:begin//等指令
        en_dout_pro<=0;
        if(en_din_pro)begin//指令来了
            cmd_reg<=din_pro;//接受指令
            state<=1;
        end
    end
    1:begin//收A
        if(en_din_pro)begin
            A_reg<=din_pro;
            state<=2;
        end
    end
    2:begin
        if(en_din_pro)begin
            B_reg<=din_pro;
            state<=3;
        end
    end
    3:begin//指令译码和执行
        case(cmd_reg)
            add_ab:begin
                dout_pro<=A_reg+B_reg;
            end
            sub_ab:begin
                dout_pro<=A_reg-B_reg;
            end
            and_ab:begin
                dout_pro<=A_reg&B_reg;
            end
            or_ab:begin
                dout_pro<=A_reg|B_reg;
            end
        endcase        
    end
    4:begin//通过串口发送指令执行结果
        if(rdy==1) begin//空闲
            en_dout_pro<=1;
            state<=0;
        end
    end 
    default:begin
        state<=0;
        en_dout_pro<=0;    
    end
    
    endcase
end

endmodule
