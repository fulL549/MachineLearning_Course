`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/09/24 21:23:08
// Design Name: 
// Module Name: key_led_test
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


module led_chaser(
    input clk,
    input reset,
    
    input sw_i,
    
    output [2:0] led_o
    );
    reg [2:0] led;
    reg [24:0] count;
    assign led_o = led;
    
    always @(posedge clk or negedge reset)begin
        if(~reset)
            count <= 0;
        else if(count == 10000000)
            count <= 0;
        else
            count <= count+1;
    end
    
    always @(posedge clk or negedge reset)begin
        if(~reset)
            led <= 3'b000;
        else if(count == 10000000)begin
            if(sw_i == 1)begin
                if(led == 3'b000)
                    led <= 3'b100;
                else
                    led <= {led[0],led[2:1]};
            end
            else if(sw_i == 0)begin
                if(led == 3'b000)
                    led <= 3'b001;
                else
                    led <= {led[1:0],led[2]};
            end
        end
    end
endmodule
