module uart_loop_tb; 
 // Internal signals 
 parameter clk_freq = 10; 
 
 reg clk; 
 reg rst_n; 
 reg tx_release; 
 reg [7:0] rx_data; 
 reg rx_valid; 
 wire rx_ready; 
 wire [7:0] tx_data; 
 reg tx_valid; 
 reg tx_ready; 
 wire fifo_wren; 
 wire fifo_rden; 
 wire fifo_full; 
 wire fifo_empty; 
 
 // Clock process to generate clock signals 
 always begin 
 # (clk_freq / 2) clk = ~clk; 
 end 
 // Reset signals 
 initial begin 
 clk = 1'b1; 
 rst_n = 1'b0; 
 #50 rst_n = 1'b1; 
 end 
 // DUT instantiation 
 fifo_8b u_fifo_8b ( 
 .clk(clk), // input wire clk 
 .srst(~rst_n), // input wire srst 
 .din(rx_data), // input wire [7 : 0] din 
 .wr_en(fifo_wren), // input wire wr_en 
 .rd_en(fifo_rden), // input wire rd_en 
 .dout(tx_data), // output wire [7 : 0] dout 
 .full(fifo_full), // output wire full 
 .empty(fifo_empty) // output wire empty 
 ); 
 // simulation code 
 initial begin 
 rx_data = 8'h25; 
 rx_valid = 1'b0; 
 tx_ready = 1'b1; 
 tx_release = 1'b0; 
 // oper after rst release 
#80 tx_release = 1'b0; 

 // your code here 
     // 模拟接收数据
    repeat (10) begin
        #50; // 每次发送间隔50时间单位
        rx_data[6:0] = rx_data[7:1]; // 发送递增的数据
        
        rx_valid = 1'b1; // 标记数据有效
        #10; // 保持有效信号一拍
        rx_valid = 1'b0; // 清除有效信号
    end
 // your code end 
 
 // this signal used for release tx_ready 
 #80 tx_release = 1'b1; 
 #10 tx_release = 1'b0; 
 end 
 
 always @(posedge clk) begin 
 if(tx_valid) 
 tx_ready <= 1'b0; 
 else if (tx_release) 
 tx_ready <= 1'b1; 
 end 

//loop here
    // FIFO 写入控制逻辑
    assign fifo_wren = rx_valid && rx_ready && !fifo_full; // 当接收到有效数据且FIFO不满时写入
    
    // FIFO 读取控制逻辑
    assign fifo_rden = tx_ready && !fifo_empty; // 当发送器准备好且FIFO不空时读取
    
    // tx_valid 数据有效性管理
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_valid <= 1'b0; // 复位时，tx_valid 设为低
        end else begin
            tx_valid <= fifo_rden; // 只有在读取数据时，tx_valid 才为高
        end
    end
    
    // 控制 rx_ready 信号，表示是否可以接收数据
    assign rx_ready = !fifo_full; // FIFO 不满时 rx_ready 为高

endmodule

