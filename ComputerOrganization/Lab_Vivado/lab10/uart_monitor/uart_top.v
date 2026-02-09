// Copyright 2017 ETH Zurich and University of Bologna.
// -- Adaptable modifications made for hbirdv2 SoC. -- 
// Copyright 2020 Nuclei System Technology, Inc.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

module uart_top#(
    parameter                           baudrate = 115200,
    parameter                           tx_bytelen = 4,
    parameter                           rx_bytelen = 4
)(
    input  wire                         clk_i,
    input  wire                         rst_n,
    
    input  wire                         rx_i,     // Receiver input
    output wire                         tx_o,     // Transmitter output

    input  wire                         tx_valid_i,
    output wire                         tx_ready_o,
    input  wire [8*tx_bytelen-1:0]      tx_data_i,

    output wire                         rx_valid_o,
    input  wire                         rx_ready_i,
    output reg  [8*rx_bytelen-1:0]      rx_data_o

);

    localparam  baud_div = 434 ;

    // receive buffer register, read only
    wire [7:0]  rx_data;
    wire [7:0]  tx_data;
    // parity error
    wire        parity_error;
    wire [3:0]  IIR_o;
    reg  [3:0]  clr_int;
    // tx flow control
    // wire        tx_ready;
    // rx flow control

    // wire        rx_valid;


    reg         fifo_tx_valid;
    reg         tx_valid;
    wire        fifo_rx_valid;
    reg         fifo_rx_ready;
    wire        rx_ready;

    wire        fifo_wren;
    wire        fifo_rden;
    wire        fifo_full;
    wire        fifo_empty;

    wire        inner_tx_valid;
    wire        inner_tx_ready;
    wire  [7:0] inner_tx_data;

    wire        inner_rx_valid;
    wire        inner_rx_ready;
    wire  [7:0] inner_rx_data;

    reg         inner_tx_busy;

    reg   [8*tx_bytelen-1:0]      tx_data_reg;
    wire  [8*rx_bytelen-1:0]      rx_data_sel;

    reg   [$clog2(tx_bytelen):0]    tx_cnt;
    wire                            tx_cnt_en;
    wire                            tx_cnt_clr;
    wire                            tx_cnt_ov;

    reg   [$clog2(rx_bytelen):0]    rx_cnt;
    wire                            rx_cnt_en;
    wire                            rx_cnt_clr;
    wire                            rx_cnt_ov;

    uart_rx u_uart_rx(
        .clk_i            ( clk_i                                                      ),
        .rstn_i           ( rst_n                                                     ),
        .rx_i             ( rx_i                                                     ),
        .cfg_en_i         ( 1'b1 ),     
        .cfg_div_i        ( 16'd0431 ), 
        .cfg_parity_en_i  ( 1'b0 ),     
        .cfg_parity_sel_i ( 2'b00 ),    
        .cfg_bits_i       ( 2'b11 ),    
        // .cfg_stop_bits_i    ( regs_q[(LCR * 8) + 2]                               ),
        .busy_o           (                                                          ),
        .err_o            ( parity_error                                             ),
        .err_clr_i        ( 1'b0                                                     ),
        .rx_data_o        ( inner_rx_data                                                ),
        .rx_valid_o       ( inner_rx_valid                                               ),
        .rx_ready_i       ( inner_rx_ready                                               )
    );

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            rx_cnt	<= 'b0;
        end else begin
            casex({rx_cnt_clr, rx_cnt_en})
                2'b00:
                    rx_cnt  <= rx_cnt;
                2'b01:
                    rx_cnt	<= rx_cnt + 1'b1;
                2'b1x:
                    rx_cnt  <= 'b0;
            endcase
        end
    end

    genvar i;
    genvar j;

    assign rx_cnt_en = inner_rx_valid & inner_rx_ready;
    assign rx_cnt_clr = rx_valid_o & rx_ready_i;
    assign rx_cnt_ov = rx_cnt == rx_bytelen;

    assign inner_rx_ready = ~rx_cnt_ov;
    assign rx_valid_o = rx_cnt_ov;

    generate 

        for (i = 0; i < rx_bytelen; i = i + 1) begin: rx_data_input_sel
            for (j = 0; j < 8; j = j + 1) begin: rx_data_input_byte_copy
                assign rx_data_sel[i*8+j] = (rx_cnt == i) ? inner_rx_data[j] : rx_data_o[i*8+j];
            end
        end
    endgenerate

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            rx_data_o	<= 'b0;
        end else begin
            rx_data_o   <= rx_data_sel;
        end
    end

    // always @(posedge clk_i or negedge rst_n) begin
    //     if(~rst_n) begin
    //         rx_data_o	<= 'b0;
    //     end else begin
    //         casex({rx_cnt})
    //             0:
	// 				rx_data_o[7:0]	    <=	inner_rx_data;
	// 			1:
	// 				rx_data_o[15:8]	    <=	inner_rx_data;
	// 			2:
	// 				rx_data_o[23:16]	<=	inner_rx_data;
	// 			3:
	// 				rx_data_o[31:24]	<=	inner_rx_data;
	// 			4:
	// 				rx_data_o[39:32]	<=	inner_rx_data;
	// 			5:
	// 				rx_data_o[47:40]	<=	inner_rx_data;
	// 			6:
	// 				rx_data_o[55:48]	<=	inner_rx_data;
	// 			7:
	// 				rx_data_o[63:56]	<=	inner_rx_data;
	// 			8:
	// 				rx_data_o[71:64]	<=	inner_rx_data;
	// 			9:
	// 				rx_data_o[79:72]	<=	inner_rx_data;
    //             default:
    //                 rx_data_o           <=  rx_data_o;
    //         endcase
    //     end
    // end

    uart_tx u_uart_tx(
        .clk_i            ( clk_i                                                      ),
        .rstn_i           ( rst_n                                                     ),
        .tx_o             ( tx_o                                                     ),
        .busy_o           (                                                          ),
        .cfg_en_i         ( 1'b1 ),     
        .cfg_div_i        ( 16'd0431 ), 
        .cfg_parity_en_i  ( 1'b0 ),     
        .cfg_parity_sel_i ( 2'b00 ),    
        .cfg_bits_i       ( 2'b11 ),    
        .cfg_stop_bits_i  ( 1'b1 ),     
        .tx_data_i        ( inner_tx_data                                                ),
        .tx_valid_i       ( inner_tx_valid                                              ),
        .tx_ready_o       ( inner_tx_ready                                             )
    );

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            tx_cnt	<= 'b0;
        end else begin
            casex({tx_cnt_clr, tx_cnt_en})
                2'b00:
                    tx_cnt  <= tx_cnt;
                2'b01:
                    tx_cnt	<= tx_cnt + 1'b1;
                2'b1x:
                    tx_cnt  <= 'b0;
            endcase
        end
    end

    assign tx_cnt_en = inner_tx_valid & inner_tx_ready;
    assign tx_cnt_clr = tx_cnt_ov;
    assign tx_cnt_ov = tx_cnt == tx_bytelen;

    assign inner_tx_valid = inner_tx_busy;
    assign tx_ready_o = ~inner_tx_busy;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            tx_data_reg	<= 'b0;
        end else begin
            tx_data_reg	<= tx_data_i;
        end
    end

    generate
        for (i = 0; i < 8; i = i + 1) begin: tx_data_sel
            assign inner_tx_data[i] = tx_data_reg[(tx_bytelen-tx_cnt-1)*8+i];
        end
    endgenerate

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            inner_tx_busy   <= 'b0;
        end else begin
            casex({tx_cnt_ov, tx_valid_i & tx_ready_o})
                2'b00:
                    inner_tx_busy   <= inner_tx_busy;
                2'b01:
                    inner_tx_busy   <= 1'b1;
                2'b1x:
                    inner_tx_busy   <= 1'b0;
            endcase
        end
    end

    // ila_monitor u_ila_uart_inner (
    //     .clk(clk_i), // input wire clk


    //     .probe0(inner_rx_data), // input wire [7:0]  probe0  
    //     .probe1({inner_rx_valid, inner_rx_ready, rx_cnt_clr}) // input wire [2:0]  probe1
    // );

    
    // ila_uart_inner u_ila_uart_inner_rx (
    //     .clk(clk_i), // input wire clk


    //     .probe0(rx_i), // input wire [0:0]  probe0  
    //     .probe1(inner_rx_valid) // input wire [0:0]  probe1
    // );
    
endmodule