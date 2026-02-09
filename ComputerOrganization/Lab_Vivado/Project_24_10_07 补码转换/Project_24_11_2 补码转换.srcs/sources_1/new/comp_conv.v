`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/02 21:21:10
// Design Name: 
// Module Name: comp_conv
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
补码转换
正数不变
负数非符号位取反+1
*/

module comp_conv(
                a,
                a_comp
    );
input[7:0]      a;//输入
output[7:0]     a_comp;//输出

//wire[6:0]       b;//按位取反的幅度位
//wire[7:0]       y;//负数的补码
//assign          b=~a[6:0];
//assign          y[6:0]=b+1;//按位取反+1
//assign          y[7]=a[7];//符号位不变

assign          a_comp=a[7]?{a[7],~a[6:0]+1}:a;//三目操作符

endmodule
