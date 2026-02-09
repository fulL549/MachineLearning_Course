`timescale 1ns / 1ps
module alu_top(
    input							clk_i,
    input							rst_n,


    input  wire                     rx_i,     // Receiver input
    output wire                     tx_o     // Transmitter output

);

    reg                             tx_valid;
    wire                            tx_ready;

    wire                            rx_valid;
    wire                            rx_ready;
    wire    [7:0]                   rx_data;

    wire    [31:0]                  calc_data;


    wire                            fifo_wren;
    wire                            fifo_rden;
    wire                            fifo_full;
    wire                            fifo_empty;

    wire        [31:0]              data_A;
    wire        [31:0]              data_B;
    reg         [2:0]               op;
    wire        [31:0]              alu_result;

    wire                            op_valid;
    reg                             op_valid_dly1;

    

    uart_top u_uart_top(
        .clk_i      (clk_i      ),
        .rst_n      (rst_n      ),
        .rx_i       (rx_i       ),
        .tx_o       (tx_o       ),
        .tx_valid_i (tx_valid   ),
        .tx_ready_o (tx_ready   ),
        .tx_data_i  (alu_result ),
        .rx_valid_o (rx_valid   ),
        .rx_ready_i (rx_ready   ),
        .rx_data_i  (rx_data    )
    );

    assign rx_ready = ~fifo_full;

    fifo_8b_2_32b u_op_convert_fifo (
        .clk(clk_i),      // input wire clk
        .srst(~rst_n),    // input wire srst
        .din(rx_data),      // input wire [7 : 0] din
        .wr_en(fifo_wren),  // input wire wr_en
        .rd_en(fifo_rden),  // input wire rd_en
        .dout(calc_data),    // output wire [31 : 0] dout
        .full(fifo_full),    // output wire full
        .empty(fifo_empty)  // output wire empty
    );

    assign fifo_wren = rx_valid & rx_ready & fifo_empty;
    assign fifo_rden = op_valid;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            op	<= 'b0;
        end else begin
            if(op_valid)
                op	<= rx_data[2:0];
        end
    end

    assign data_A = {{16{calc_data[31]}}, calc_data[31:16]};
    assign data_B = {{16{calc_data[15]}}, calc_data[15:0]};

    assign op_valid = rx_valid & rx_ready & ~fifo_empty;

    always @(posedge clk_i or negedge rst_n) begin
        if(~rst_n) begin
            tx_valid	    <= 'b0;
            op_valid_dly1   <= 'b0;
        end else begin
            tx_valid	    <= op_valid_dly1;
            op_valid_dly1	<= op_valid;
        end
    end

    
    // your ALU instantiation here
        alu u_alu(
        .A_i        (data_A),
        .B_i        (data_B),
        .op_i       (op),
        .result_o   (alu_result)
    );
    // your ALU instantiation end
    

endmodule

