`timescale 1ns / 1ps
//`default_nettype none
//////////////////////////////////////////////////////////////////////////////////
// Company: SYSU
// Engineer: Kafuuchino
// 
// Create Date: 2024/11/24 20:48:44
// Design Name: 
// Module Name: sys_monitor
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


module sys_monitor #(
    parameter                       monitor_len = 11,
    parameter                       RX_byte = 1,
    parameter                       TX_byte = 4
)(
    input							clk_i,
    input							rst_n,

    input                           clk_run_i,  // proc run freq

    input                           rx_i,
    output                          tx_o,

    input       [monitor_len-1:0]   monitor_list_i,
    input       [31:0]              pc_i,

    input       [31:0]              proc_cur_inst_i,
    // output                          proc_inst_print_o,

    output                          proc_run_en_o,
    output                          proc_reset_o,
    input                           pc_ov_i,

    output                          reg_wren_o,
    output      [4:0]               reg_addr_o,
    output      [31:0]              reg_wdata_o,
    input       [31:0]              reg_rdata_i,

    output      [31:0]              uart_reg_o,

    output                          mem_wren_o,
    output                          mem_rden_o,
    output      [14:0]              mem_addr_o,
    output      [31:0]              mem_wdata_o,
    input       [31:0]              mem_rdata_i

);

    reg                             r2u_fifo_wren;
    wire                            r2u_fifo_rden;
    wire                            r2u_fifo_full;
    wire                            r2u_fifo_empty;
    wire        [63:0]              r2u_fifo_rdata;
    reg         [63:0]              r2u_fifo_wdata;

    wire                            u2r_fifo_wren;
    wire                            u2r_fifo_rden;
    wire                            u2r_fifo_full;
    wire                            u2r_fifo_empty;
    wire        [7:0]               u2r_fifo_rdata;
    wire        [7:0]               u2r_fifo_wdata;

    wire        [RX_byte*8-1:0]     rx_data;
    wire                            rx_valid;
    wire                            rx_ready;
    wire        [TX_byte*8-1:0]     tx_data;
    reg                             tx_valid;
    wire                            tx_ready;

    // reg         [7:0]               cmd_reg;
    reg                             u2r_fifo_rden_dly1;
    reg                             u2r_fifo_rden_dly2;
    reg                             mem_rdata_tx_dly1;

    wire                                                   reg_rdata_tx;
    wire                                                   mem_rdata_tx; 
    // reg                                                    mem_rdata_tx_dly1;

    wire                                                   proc_inst_print;

    wire                                                   uart_reg_wren;
    reg         [31:0]                                     uart_reg;

    // wire                            proc_run_en;
    // wire                            proc_reset;
    // wire                            proc_inst_print_req;
    // wire                            proc_break_point_set;

    // reg                             proc_run_hold;

    // clock domain: clk_run - User define

    // r2u fifo
    always @(posedge clk_run_i or negedge rst_n) begin
        if(~rst_n) begin
            r2u_fifo_wdata <= 32'h0;
        end else begin
            casex({proc_inst_print, mem_rdata_tx, reg_rdata_tx})
                3'b000:
                    r2u_fifo_wdata <= {
                                monitor_list_i, 
                                {(TX_byte*8-monitor_len-8){1'b0}}, 
                                pc_i[7:0], 
                                proc_cur_inst_i};
                3'b001:
                    r2u_fifo_wdata <= {reg_rdata_i, proc_cur_inst_i};
                3'b01x:
                    r2u_fifo_wdata <= {mem_rdata_i, proc_cur_inst_i};
                3'b1xx:
                    r2u_fifo_wdata <= {proc_cur_inst_i, proc_cur_inst_i};
            endcase
        end
    end
    
    always @(posedge clk_run_i or negedge rst_n) begin
        if(~rst_n) begin
            r2u_fifo_wren	    <= 'b0;
            mem_rdata_tx_dly1   <= 'b0;
        end else begin
            mem_rdata_tx_dly1   <= mem_rdata_tx;
            r2u_fifo_wren	    <= proc_run_en_o | mem_rdata_tx | reg_rdata_tx | proc_inst_print;
        end
    end

    assign mem_rden_o = mem_rdata_tx;

    // u2r fifo
    assign u2r_fifo_rden = ~u2r_fifo_empty;

    always @(posedge clk_run_i or negedge rst_n) begin
        if(~rst_n) begin
            u2r_fifo_rden_dly1	<= 'b0;
            u2r_fifo_rden_dly2	<= 'b0;
            mem_rdata_tx_dly1	<= 'b0;
        end else begin
            u2r_fifo_rden_dly1	<= u2r_fifo_rden;
            u2r_fifo_rden_dly2	<= u2r_fifo_rden_dly1;
            mem_rdata_tx_dly1	<= mem_rdata_tx;
        end
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            uart_reg	<= 'b0;
        end else begin
            if(uart_reg_wren)
                uart_reg	<= reg_wdata_o;
        end
    end

    assign uart_reg_o = uart_reg;

    debug_core u_debug_core(
        .clk_i             (clk_run_i         ),
        .rst_n             (rst_n             ),
        .cmd_data_i        (u2r_fifo_rdata),
        .cmd_data_valid_i  (u2r_fifo_rden_dly1),
        .pc_i              (pc_i              ),
        .pc_ov_i           (pc_ov_i           ),
        .proc_run_en_o     (proc_run_en_o     ),
        .proc_reset_o      (proc_reset_o      ),
        .proc_inst_print_o (proc_inst_print   ),
        .reg_rdata_tx_o    (reg_rdata_tx      ),
        .mem_rdata_tx_o    (mem_rdata_tx      ),
        .reg_wren_o        (reg_wren_o        ),
        .uart_reg_wren_o   (uart_reg_wren     ),
        .reg_addr_o        (reg_addr_o        ),
        .reg_wdata_o       (reg_wdata_o       ),
        .mem_wren_o        (mem_wren_o        ),
        .mem_addr_o        (mem_addr_o        ),
        .mem_wdata_o       (mem_wdata_o       )
    );

    // clock domain convert

    r2u_data_fifo u_run_2_uart_clock (
        .rst(~rst_n),        // input wire rst
        .wr_clk(clk_run_i),  // input wire wr_clk
        .rd_clk(clk_i),  // input wire rd_clk
        .din(r2u_fifo_wdata),        // input wire [63 : 0] din
        .wr_en(r2u_fifo_wren),    // input wire wr_en
        .rd_en(r2u_fifo_rden),    // input wire rd_en
        .dout(r2u_fifo_rdata),      // output wire [63 : 0] dout
        .full(r2u_fifo_full),      // output wire full
        .empty(r2u_fifo_empty)    // output wire empty
    );

    u2r_cmd_fifo u_u2r_cmd_convert (
        .rst(~rst_n),        // input wire rst
        .wr_clk(clk_i),  // input wire wr_clk
        .rd_clk(clk_run_i),  // input wire rd_clk
        .din(u2r_fifo_wdata),        // input wire [7 : 0] din
        .wr_en(u2r_fifo_wren),    // input wire wr_en
        .rd_en(u2r_fifo_rden),    // input wire rd_en
        .dout(u2r_fifo_rdata),      // output wire [7 : 0] dout
        .full(u2r_fifo_full),      // output wire full
        .empty(u2r_fifo_empty)    // output wire empty
    );

    // clock domain: clk_i - 50Mhz

    // r2u fifo
    assign r2u_fifo_rden = ~r2u_fifo_empty & tx_ready & ~tx_valid;

    // u2r fifo
    assign u2r_fifo_wdata = rx_data;
    assign u2r_fifo_wren = rx_ready & rx_valid;
    
    // uart interface
    assign tx_data = r2u_fifo_rdata[63:32];
    assign rx_ready = ~u2r_fifo_full;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            tx_valid	<= 'b0;
        end else begin
            tx_valid	<= r2u_fifo_rden;
        end
    end

    //uart instance
    uart_top #(
        .rx_bytelen (1),
        .tx_bytelen (4)
    )u_uart_top(
        .clk_i      (clk_i      ),
        .rst_n      (rst_n      ),
        .rx_i       (rx_i       ),
        .tx_o       (tx_o       ),
        .tx_valid_i (tx_valid   ),
        .tx_ready_o (tx_ready   ),
        .tx_data_i  (tx_data    ),
        .rx_valid_o (rx_valid   ),
        .rx_ready_i (rx_ready   ),
        .rx_data_o  (rx_data    )
    );
    

    // ila_monitor u_ila_monitor (
    //     .clk(clk_i), // input wire clk


    //     .probe0(rx_data), // input wire [7:0]  probe0  
    //     .probe1({rx_valid, rx_ready, u2r_fifo_wren}) // input wire [2:0]  probe1
    // );

//    ila_monitor u_ila_memdbg (
//        .clk(clk_run_i), // input wire clk


//        .probe0({mem_rdata_i[3:0], r2u_fifo_wdata[35:32]}), // input wire [7:0]  probe0  
//        .probe1({r2u_fifo_wren, mem_rdata_tx_dly1, mem_rdata_tx}) // input wire [2:0]  probe1
//    );

endmodule
