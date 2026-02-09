`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/17 10:19:33
// Design Name: 
// Module Name: FSM_test_tb
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
module sequence_detector_tb;

reg         clk;
reg         reset; 
reg         in;

wire        out;

sequence_detector sequence_detector (
               .clk(clk),     
               .reset(reset),    
               .in(in),      
               .out(out)      
);
initial begin
        clk = 0;
end

always #1 clk=~clk;//周期为2ns

initial begin
    reset=1;in=0;//先复位
    #2;
    reset=0;//再释放
end

initial begin
        #2;        in=1;
        #2;        in=1;
        #2;        in=1;
        #2;        in=0;
        #2;        in=1;
        #2;        in=1;
        #2;        in=0;
        #2;        in=1;
        #2;        in=1;
        #2;        in=0;
        #2;        in=1;
        #2;        in=1;
        #2;        in=1;
        #2;        in=0;
        #2;        in=1;
        #2;        in=1;
        #20        $stop;
      
end
endmodule