`timescale 1ns / 1ps
//`default_nettype none
//////////////////////////////////////////////////////////////////////////////////
// Company: SYSU
// Engineer: Kafuuchino
// 
// Create Date: 2024/11/28 15:25:42
// Design Name: 
// Module Name: debug_core
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: Convert uart code to debug operation
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module debug_core(
    input							clk_i,
    input							rst_n,

    input       [7:0]               cmd_data_i,
    input                           cmd_data_valid_i,

    input       [31:0]              pc_i,
    input                           pc_ov_i,

    output                          proc_run_en_o,
    output                          proc_reset_o,
    output                          proc_inst_print_o,

    output  reg                     reg_rdata_tx_o,
    output  reg                     mem_rdata_tx_o,

    output                          reg_wren_o,
    output                          uart_reg_wren_o,
    output      [4:0]               reg_addr_o,
    output      [31:0]              reg_wdata_o,
    // input       [31:0]              reg_rdata_o,

    output                          mem_wren_o,
    output      [14:0]              mem_addr_o,
    output      [31:0]              mem_wdata_o
    // input       [31:0]              mem_rdata_o

);

    wire                            proc_run_en;
    reg                             proc_run_hold;
    wire                            proc_reset;
    wire                            proc_inst_print_req;
    wire                            proc_break_point_set;
    wire                            proc_reg_oper;
    wire                            proc_mem_oper;
    wire                            proc_step_one;

    wire                            reg_wr_flag;
    wire                            uart_reg_wr_flag;
    wire                            reg_rd_flag;
    wire                            reg_rd_all_flag;
    wire                            mem_wr_flag;
    wire                            mem_rd_flag;

    reg         [7:0]               cmd_reg;
    reg                             cmd_data_valid_dly1;

    reg         [31:0]              qword_cmd;

    reg         [31:0]              bp_pc;
    reg         [4:0]               reg_addr;
    reg         [14:0]              mem_addr;

    reg                             reg_wren;
    reg                             uart_reg_wren;
    reg                             mem_wren;

    reg                             reg_rden;
    reg                             mem_rden;
    reg                             mem_rden_dly1;
    reg                             mem_rden_dly2;
    
    reg                             long_cmd_recv;

    wire                            breakpoint;
    reg                             bp_set;
    reg                             bp_enable;
    reg                             mem_addr_set_en;

    reg         [2:0]               cmd_recv_cnt;
    wire                            cmd_recv_cnt_en;
    wire                            cmd_recv_cnt_clr;
    wire                            cmd_recv_cnt_ov;

    reg         [4:0]               reg_rd_cnt;
    reg                             reg_rd_cnt_en;
    wire                            reg_rd_cnt_clr;
    wire                            reg_rd_cnt_ov;

    reg                             reg_rd_sel;

    reg         [2:0]               cmd_recv_cnt_ov_thold;

    reg         [8:0]               c_state;
    reg         [8:0]               n_state;

    localparam                      s_idle              = 9'b0_0000_0001,
                                    s_bp                = 9'b0_0000_0010,
                                    s_reg_wr            = 9'b0_0000_0100,
                                    s_reg_wdata         = 9'b0_0000_1000,
                                    s_uart_reg_wdata    = 9'b0_0001_0000,
                                    s_reg_rdall         = 9'b0_0010_0000,
                                    s_mem_addr          = 9'b0_0100_0000,
                                    s_mem_wdata         = 9'b0_1000_0000,
                                    s_fin               = 9'b1_0000_0000;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            c_state	<= 'b0;
        end else begin
            c_state	<= n_state;
        end
    end

    always @(*) begin
        reg_wren = 1'b0;
        uart_reg_wren = 1'b0;
        reg_rden = 1'b0;
        mem_wren = 1'b0;
        mem_rden = 1'b0;
        long_cmd_recv = 1'b1;
        reg_rd_cnt_en = 1'b0;
        reg_rd_sel = 1'b0;
        cmd_recv_cnt_ov_thold = 3'b000;
        case(c_state)
            s_idle:begin
                long_cmd_recv = 1'b0;
                if(cmd_data_valid_dly1) begin
                    casex({proc_break_point_set, proc_reg_oper, proc_mem_oper})
                        3'b000:
                            n_state = s_idle;
                        3'b001:
                            n_state = s_mem_addr;
                        3'b01x:
                            n_state = s_reg_wr;
                        3'b1xx:
                            n_state = s_bp;
                    endcase
                end else begin
                    n_state = s_idle;
                end
            end
            s_bp:begin
                cmd_recv_cnt_ov_thold = 3'b100;
                if(cmd_recv_cnt_ov) begin
                    n_state = s_fin;
                end else begin
                    n_state = s_bp;
                end
            end
            s_reg_wr:begin
                cmd_recv_cnt_ov_thold = 3'b000;
                if(cmd_data_valid_dly1) begin
                    casex({reg_rd_all_flag, reg_wr_flag, uart_reg_wr_flag})
                        3'b000: begin
                            reg_rden = 1'b1;
                            n_state = s_fin;
                        end
                        3'b001:
                            n_state = s_uart_reg_wdata;
                        3'b01x:
                            n_state = s_reg_wdata;
                        3'b1xx:
                            n_state = s_reg_rdall;
                    endcase
                end else begin
                    n_state = s_reg_wr;
                end
            end
            s_reg_wdata: begin
                cmd_recv_cnt_ov_thold = 3'b100;
                if(cmd_recv_cnt_ov) begin
                    reg_wren = 1'b1;
                    n_state = s_fin;
                end else begin
                    n_state = s_reg_wdata;
                end
            end
            s_uart_reg_wdata:begin
                cmd_recv_cnt_ov_thold = 3'b100;
                if(cmd_recv_cnt_ov) begin
                    uart_reg_wren = 1'b1;
                    n_state = s_fin;
                end else begin
                    n_state = s_uart_reg_wdata;
                end
            end
            s_reg_rdall:begin
                reg_rden = 1'b1;
                reg_rd_cnt_en = 1'b1;
                reg_rd_sel = 1'b1;
                if(reg_rd_cnt_ov)
                    n_state = s_fin;
                else
                    n_state = s_reg_rdall;
            end
            s_mem_addr:begin
                cmd_recv_cnt_ov_thold = 3'b010;
                if(cmd_recv_cnt_ov) begin
                    if(mem_wr_flag) begin
                        n_state = s_mem_wdata;
                    end else begin
                        mem_rden = 1'b1;
                        n_state = s_fin;
                    end
                end else begin
                    n_state = s_mem_addr;
                end
            end
            s_mem_wdata: begin
                cmd_recv_cnt_ov_thold = 3'b100;
                if(cmd_recv_cnt_ov) begin
                    mem_wren = 1'b1;
                    n_state = s_fin;
                end else begin
                    n_state = s_mem_wdata;
                end
            end
            s_fin:begin
                long_cmd_recv = 1'b0;
                n_state = s_idle;
            end
            default:
                n_state = s_idle;
        endcase
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            cmd_data_valid_dly1	<= 'b0;
        end else begin
            cmd_data_valid_dly1	<= cmd_data_valid_i;
        end
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            cmd_reg	<= 'b0;
        end else begin
            casex({cmd_data_valid_dly1, cmd_data_valid_i})
                2'b00:
                    cmd_reg <= cmd_reg;
                2'b01:
                    cmd_reg	<= cmd_data_i;
                2'b10:
                    cmd_reg <= 4'h0;
                2'b11:
                    cmd_reg	<= cmd_data_i;
            endcase
        end
    end

    assign proc_run_en          = (cmd_reg[2:0] == 3'b000) & ~long_cmd_recv;
    assign proc_reset           = (cmd_reg[2:0] == 3'b001) & ~long_cmd_recv;
    assign proc_inst_print_req  = (cmd_reg[2:0] == 3'b010) & ~long_cmd_recv;
    assign proc_break_point_set = (cmd_reg[2:0] == 3'b011) & ~long_cmd_recv;
    assign proc_reg_oper        = (cmd_reg[2:0] == 3'b100) & ~long_cmd_recv;
    assign proc_mem_oper        = (cmd_reg[2:0] == 3'b101) & ~long_cmd_recv;
    assign proc_step_one        = (cmd_reg[2:0] == 3'b110) & ~long_cmd_recv;

    assign reg_wr_flag          = (cmd_reg[7:5] == 3'b100);
    assign uart_reg_wr_flag     = (cmd_reg[7:5] == 3'b111);
    assign reg_rd_flag          = (cmd_reg[7:5] == 3'b010);
    assign reg_rd_all_flag      = (cmd_reg[7:5] == 3'b110);
    
    assign mem_wr_flag          = (qword_cmd[31] == 1'b1);
    assign mem_rd_flag          = (qword_cmd[31] == 1'b0);


    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            proc_run_hold	<= 'b0;
        end else begin
            casex({pc_ov_i, proc_run_en & cmd_data_valid_dly1, breakpoint})
                3'b000:
                    proc_run_hold   <= proc_run_hold;
                3'b001:
                    proc_run_hold	<= 1'b0;
                    // proc_run_hold	<= proc_run_hold;
                3'b01x:
                    proc_run_hold	<= 1'b1;
                3'b1xx:
                    proc_run_hold   <= 1'b0;
            endcase
        end
    end

    assign breakpoint = (pc_i == bp_pc) & bp_enable;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            cmd_recv_cnt	<= 'b0;
        end else begin
            casex({cmd_recv_cnt_clr, cmd_recv_cnt_en})
                2'b00:
                    cmd_recv_cnt	<= cmd_recv_cnt;
                2'b01:
                    cmd_recv_cnt	<= cmd_recv_cnt + 1'b1;
                2'b1x:
                    cmd_recv_cnt	<= 'b0;
            endcase
        end
    end

    assign cmd_recv_cnt_en = cmd_data_valid_dly1 & (|cmd_recv_cnt_ov_thold);
    assign cmd_recv_cnt_clr = cmd_recv_cnt_ov;

    assign cmd_recv_cnt_ov = (cmd_recv_cnt == cmd_recv_cnt_ov_thold);

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            reg_rd_cnt	<= 'b0;
        end else begin
            casex({reg_rd_cnt_clr, reg_rd_cnt_en})
                2'b00:
                    reg_rd_cnt	<= reg_rd_cnt;
                2'b01:
                    reg_rd_cnt	<= reg_rd_cnt + 1'b1;
                2'b1x:
                    reg_rd_cnt	<= 'b0;
            endcase
        end
    end

    assign reg_rd_cnt_ov = (reg_rd_cnt == 5'h1f);
    assign reg_rd_cnt_clr = reg_rd_cnt_ov;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            qword_cmd   <=  32'h0;
        end else begin
            case({cmd_recv_cnt[1:0]})
                2'b00:
                    qword_cmd <= {cmd_reg, qword_cmd[23:0]};
                2'b01:
                    qword_cmd <= {qword_cmd[31:24], cmd_reg, qword_cmd[15:0]};
                2'b10:
                    qword_cmd <= {qword_cmd[31:16], cmd_reg, qword_cmd[7:0]};
                2'b11:
                    qword_cmd <= {qword_cmd[31:8], cmd_reg};
            endcase
        end
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            bp_set	<= 'b0;
        end else begin
            bp_set	<= (c_state == s_bp) & cmd_recv_cnt_ov;
        end
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            bp_enable	<= 'b0;
            mem_addr_set_en <= 'b0;
        end else begin
            casex({bp_set & (qword_cmd == 32'hffff_ffff), bp_set})
                2'b00:
                    bp_enable   <= bp_enable;
                2'b01:
                    bp_enable   <= 1'b1;
                2'b1x:
                    bp_enable   <= 1'b0;
            endcase
            mem_addr_set_en <= (c_state == s_mem_addr) & cmd_recv_cnt_ov;
        end
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            bp_pc	<= 'b0;
        end else begin
            if(bp_set)
                bp_pc	<= qword_cmd;
        end
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            reg_addr	<= 'b0;
        end else begin
            if((c_state == s_reg_wr) & cmd_data_valid_dly1)
                reg_addr	<= cmd_reg[4:0];
        end
    end

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            mem_addr	<= 'b0;
        end else begin
            if(mem_addr_set_en)
                mem_addr	<= qword_cmd[30:16];
        end
    end

    assign proc_run_en_o = proc_run_hold | proc_step_one;
    assign proc_reset_o = proc_reset;
    assign proc_inst_print_o = proc_inst_print_req;

    // always @(posedge clk_i or negedge rst_n) begin
    //     if(~rst_n) begin
    //         reg_wren_o	    <= 'b0;
    //         uart_reg_wren_o	<= 'b0;
    //         mem_wren_o	    <= 'b0;
    //     end else begin
    //         reg_wren_o	    <= reg_wren;
    //         uart_reg_wren_o	<= uart_reg_wren;
    //         mem_wren_o	    <= mem_wren;
    //     end
    // end

    assign reg_wren_o       = reg_wren;
    assign uart_reg_wren_o	= uart_reg_wren;
    assign mem_wren_o	    = mem_wren;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            reg_rdata_tx_o	<= 'b0;
            mem_rdata_tx_o	<= 'b0;
            mem_rden_dly1	<= 'b0;
            mem_rden_dly2	<= 'b0;
        end else begin
            reg_rdata_tx_o	<= reg_rden;
            mem_rdata_tx_o	<= mem_rden_dly1;
            mem_rden_dly1	<= mem_rden;
            mem_rden_dly2	<= mem_rden_dly1;
        end
    end
    
    assign reg_addr_o = reg_rd_sel ? reg_rd_cnt : reg_addr;
    assign reg_wdata_o = qword_cmd;
    assign mem_addr_o = mem_addr[14:0];
    assign mem_wdata_o = qword_cmd;

//    ila_debug_port u_ila_debug_core (
//        .clk(clk_i), // input wire clk


//        .probe0({proc_run_en, proc_run_hold, proc_reset, proc_inst_print_req, proc_break_point_set, proc_reg_oper, proc_mem_oper, proc_step_one}), // input wire [7:0]  probe0  
//        .probe1({reg_wr_flag, uart_reg_wren, uart_reg_wren_o, mem_wr_flag, mem_rd_flag, cmd_data_valid_dly1}), // input wire [5:0]  probe1 
//        .probe2(cmd_reg), // input wire [7:0]  probe2 
//        .probe3({qword_cmd[31:16], mem_addr, mem_addr_set_en}), // input wire [31:0]  probe3 
//        .probe4(c_state), // input wire [5:0]  probe4 
//        .probe5(n_state), // input wire [5:0]  probe5 
//        .probe6(cmd_recv_cnt), // input wire [2:0]  probe6 
//        .probe7({cmd_recv_cnt_en, cmd_recv_cnt_clr, cmd_recv_cnt_ov}), // input wire [2:0]  probe7 
//        .probe8({mem_rden, mem_rdata_tx_o, cmd_recv_cnt_ov_thold}), // input wire [4:0]  probe8 
//        .probe9(bp_pc), // input wire [31:0]  probe9 
//        .probe10(pc_i[6:2]) // input wire [4:0]  probe10
//    );

endmodule
