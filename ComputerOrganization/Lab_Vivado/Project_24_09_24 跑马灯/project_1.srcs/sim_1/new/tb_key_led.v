`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/09/24 21:30:03
// Design Name: 
// Module Name: tb_key_led
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


module tb_led_chaser();
    reg clk;
    reg reset;
    reg sw;
    wire [2:0] led;
    
    always#5 clk = ~clk;
    
    led_chaser u_led_chaser(
        .clk(clk),
        .reset(reset),
        .sw_i(sw),
        
        .led_o(led)
    );
    
    initial begin
        clk = 1'b1;
        reset = 1'b0;
        
        #20
        reset = 1'b1;
        #80
        sw = 1;
        #80
        sw = 0;
        #100
        $stop;
    end
endmodule
