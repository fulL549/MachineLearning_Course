module uart_top #(
    parameter                           baudrate = 115200
)(
    input  wire                         clk_i,
    input  wire                         rst_n,
    input  wire                         rx_i,         // Receiver input
    output wire                         tx_o,         // Transmitter output

    input  wire                         tx_valid_i,
    output wire                         tx_ready_o,
    input  wire [31:0]                  tx_data_i,

    output wire                         rx_valid_o,
    input  wire                         rx_ready_i,
    output wire [7:0]                   rx_data_i
);

    localparam  baud_div = 434;

    // Receive buffer register, read only
    wire [7:0]  rx_data;
    wire [7:0]  tx_data;

    // Parity error
    wire        parity_error;
    wire [3:0]  IIR_o;
    reg  [3:0]  clr_int;

    // TX flow control
    reg         fifo_tx_valid;
    reg         tx_valid;
    wire        fifo_rx_valid;
    reg         fifo_rx_ready;

    // RX flow control
    wire        rx_ready;

    wire        fifo_wren;
    wire        fifo_rden;
    wire        fifo_full;
    wire        fifo_empty;

    reg         inner_tx_valid;
    wire        inner_tx_ready;
    wire  [7:0] inner_tx_data;

    // UART RX instance
    uart_rx u_uart_rx (
        .clk_i            (clk_i),
        .rstn_i           (rst_n),
        .rx_i             (rx_i),
        .cfg_en_i         (1'b1),
        .cfg_div_i        (16'd431),
        .cfg_parity_en_i  (1'b0),
        .cfg_parity_sel_i (2'b00),
        .cfg_bits_i       (2'b11),
        .busy_o           (),
        .err_o            (parity_error),
        .err_clr_i        (1'b0),
        .rx_data_o        (rx_data_i),
        .rx_valid_o       (rx_valid_o),
        .rx_ready_i       (rx_ready_i)
    );

    // UART TX instance
    uart_tx u_uart_tx (
        .clk_i            (clk_i),
        .rstn_i           (rst_n),
        .tx_o             (tx_o),
        .busy_o           (),
        .cfg_en_i         (1'b1),
        .cfg_div_i        (16'd431),
        .cfg_parity_en_i  (1'b0),
        .cfg_parity_sel_i (2'b00),
        .cfg_bits_i       (2'b11),
        .cfg_stop_bits_i  (1'b1),
        .tx_data_i        (inner_tx_data),
        .tx_valid_i       (inner_tx_valid),
        .tx_ready_o       (inner_tx_ready)
    );

    assign fifo_wren = tx_ready_o & tx_valid_i;
    assign tx_ready_o = ~fifo_full;

    assign fifo_rden = ~fifo_empty & inner_tx_ready & ~inner_tx_valid;

    // Inner TX valid control
    always @(posedge clk_i or negedge rst_n) begin
        if (~rst_n) begin
            inner_tx_valid <= 1'b0;
        end else begin
            inner_tx_valid <= fifo_rden;
        end
    end

    // FIFO instance
    fifo_32b_2_8b u_convert_fifo (
        .clk(clk_i),
        .srst(~rst_n),
        .din(tx_data_i),
        .wr_en(fifo_wren),
        .rd_en(fifo_rden),
        .dout(inner_tx_data),
        .full(fifo_full),
        .empty(fifo_empty)
    );

endmodule
