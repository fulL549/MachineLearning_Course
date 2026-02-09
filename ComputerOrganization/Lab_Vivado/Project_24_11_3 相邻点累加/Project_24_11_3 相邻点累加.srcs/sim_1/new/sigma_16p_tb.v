`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/03 23:52:22
// Design Name: 
// Module Name: sigma_16p_tb
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


module sigma_16p_tb;
reg             clk;
reg             res;
reg[7:0]        date_in;
reg             syn_in;
wire[11:0]      date_out;
wire            syn_out;    
             
sigma_16p sigma_16p(
                .clk(clk),
                .res(res),
                .date_in(date_in),
                .syn_in(syn_in),
                .date_out(date_out),
                .syn_out(syn_out)
                 );  
                 
initial begin
        clk<=0;
        res<=0;
        date_in<=1;
        syn_in<=0;
        #17 res<=1;
        #3000 $stop;
end

always#5    clk<=~clk;
always#100  syn_in<=~syn_in;
                 
endmodule
