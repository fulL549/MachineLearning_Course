`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/24 19:04:27
// Design Name: 
// Module Name: proc_top
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


module proc_top(
    input   clk_i,
    input   rst_n,

    input   rx_i,
    output  tx_o
);

    wire            clk_run;
    wire            clk_uart;

    wire    [31:0]  pc;

    wire            pc_run_en;
    wire            pc_jump;
    wire            pc_clr;

    wire    [10:0]  monitor_list;
    wire            pc_ov;


    wire    [31:0]  instr;

    wire    [13:0]  memaddr_actual;
    wire    [31:0]  memwdata_actual;

    wire    [31:0]  data_addr;

    wire            memwrite;
    wire    [14:0]  memaddr;
    wire    [31:0]  memwdata;
    wire    [31:0]  memrdata;

    wire            reg_wren_dbg;
    wire    [4:0]   reg_addr_dbg;
    wire    [31:0]  reg_wdata_dbg;
    wire    [31:0]  reg_rdata_dbg;

    wire            memwrite_dbg;
    wire            memread_dbg;
    wire    [14:0]  memaddr_dbg;
    wire    [31:0]  memwdata_dbg;
    wire    [31:0]  memrdata_dbg;

    wire    [31:0]  uart_reg;

    // clock divide
    clk_div u_clk_div(
        .clk_i(clk_i),
        .rst_n(rst_n),
        .clk_run(clk_run)
    );

    // mips core top
    top u_top(
        .clk_i             (clk_run           ),
        .rst_n             (pc_clr            ),//把rst修改为pc_clr
        .pc_run_en_i       (pc_run_en         ),
        .pc_clr_i          (pc_clr            ),
        .pc_o              (pc                ),
        .monitor_list_o    (monitor_list      ),
        .instr_o           (instr             ),
        .ra_debug_i        (reg_addr_dbg      ),
        .ra_debug_data_o   (reg_rdata_dbg     ),
        .reg_wdata_dbg     (reg_wdata_dbg     ),
        .reg_wren_dbg      (reg_wren_dbg      ),
        .uart_reg_i        (uart_reg          ),
        .mem_debug_rden_i  (memread_dbg       ),
        .mem_debug_addr_i  (memaddr_dbg       ),
        .mem_debug_rdata_o (memrdata_dbg      )
    );
    

    // uart monitor

    sys_monitor #(
        .monitor_len            (11),
        .RX_byte                (1),
        .TX_byte                (4)
    )u_sys_monitor(
        .clk_i                  (clk_i),//修改clk_uart为clk_i;
        .rst_n                  (rst_n),
        .clk_run_i              (clk_run),
        .rx_i                   (rx_i),
        .tx_o                   (tx_o),
        .monitor_list_i         (monitor_list),
        .pc_i                   (pc),
        .proc_cur_inst_i        (instr),
        .proc_run_en_o          (pc_run_en),
        .proc_reset_o           (pc_clr),
        .pc_ov_i                (pc_ov),
        .reg_addr_o             (reg_addr_dbg        ),
        .reg_wdata_o            (reg_wdata_dbg       ),
        .reg_rdata_i            (reg_rdata_dbg       ),
        .reg_wren_o             (reg_wren_dbg        ),
        .uart_reg_o             (uart_reg),
        .mem_wren_o             (),
        .mem_rden_o             (memread_dbg),
        .mem_addr_o             (memaddr_dbg        ),
        .mem_wdata_o            (),
        .mem_rdata_i            (memrdata_dbg       )
    );

    assign pc_ov = pc[7:2] == 6'd45;


endmodule
