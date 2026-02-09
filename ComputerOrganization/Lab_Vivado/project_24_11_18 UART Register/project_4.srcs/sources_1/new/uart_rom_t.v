module top_module (
    input  wire clk,         // System clock
    input  wire rst_n,       // Active-low reset
    input  wire rx,          // UART receiver input
    output wire tx           // UART transmitter output
);

    // Parameters
    parameter BAUDRATE = 115200;

    // Signals
    wire        rx_valid;      // Data valid signal from RX
    wire [7:0]  rx_data;       // Received data
    wire        tx_ready;      // Ready to transmit
    wire        tx_valid;      // Valid signal for TX
    reg  [31:0] tx_data;       // Data to transmit
    reg         rx_ready;      // Ready signal for RX

    // Instantiate UART top module
    uart_top #(
        .baudrate(BAUDRATE)
    ) u_uart_top (
        .clk_i(clk),
        .rst_n(rst_n),
        .rx_i(rx),
        .tx_o(tx),
        .tx_valid_i(tx_valid),
        .tx_ready_o(tx_ready),
        .tx_data_i(tx_data),
        .rx_valid_o(rx_valid),
        .rx_ready_i(rx_ready),
        .rx_data_i(rx_data)
    );

    // ROM output
    wire [31:0] rom_data;
    reg  [7:0]  rom_addr;

    // ROM instance
    inst_rom u_inst_rom (
        .clka(clk),
        .ena(rom_enable),
        .addra(rom_addr),
        .douta(rom_data)
    );

    // State encoding using parameters
    localparam [1:0] IDLE       = 2'b00,
                     SEND_DATA  = 2'b01;

    // State register
    reg [1:0] state;

    // Single always block for state transition and outputs
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            state <= IDLE;
            rom_addr <= 8'd0;
            tx_data <= 32'd0;
            rx_ready <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    rx_ready <= 1'b1; // Prepare to receive data
                    if (rx_valid) begin
                        rom_addr <= rx_data; // Capture address
                        rx_ready <= 1'b0;    // Disable RX
                        state <= SEND_DATA;  // Transition to SEND_DATA
                    end
                end
                SEND_DATA: begin
                    if (tx_ready) begin
                                        tx_data <= rom_data; // Load ROM data for TX
                        state <= IDLE; // Return to IDLE
                    end
                end
            endcase
        end
    end

    // Output assignments
    assign rom_enable = (state == SEND_DATA);
    assign tx_valid = (state == SEND_DATA);

endmodule
