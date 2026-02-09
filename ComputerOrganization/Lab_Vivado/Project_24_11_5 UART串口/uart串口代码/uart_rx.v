module uart_rx (
    input  wire        clk_i,//时钟信号
    input  wire        rstn_i,//复位信号
    input  wire        rx_i,//串行接收数据线
    input  wire [15:0] cfg_div_i,//波特率配置除数，用于生成接收时钟
    input  wire        cfg_en_i,//配置使能信号，激活接收器
    input  wire        cfg_parity_en_i,//奇偶校验使能信号
    input  wire [1:0]  cfg_parity_sel_i,//奇偶校验选择信号 选择何种校验
    input  wire [1:0]  cfg_bits_i,//数据位数配置
    // input  wire        cfg_stop_bits_i,
    output wire        busy_o,//指示接收器是否忙碌
    output reg         err_o,//错误标志输出
    input  wire        err_clr_i,//错误清除信号
    output wire [7:0]  rx_data_o,//接收到的数据
    output reg         rx_valid_o,//只是接收到的数据是否有效
    input  wire        rx_ready_i
);

    //状态机
    localparam [2:0] IDLE      = 0;//空闲状态，等待开始位
    localparam [2:0] START_BIT = 1;//检测到开始位
    localparam [2:0] DATA      = 2;//接收数据位
    localparam [2:0] SAVE_DATA = 3;//保存接收到的数据
    localparam [2:0] PARITY    = 4;//检测奇偶校验位
    localparam [2:0] STOP_BIT  = 5;//检测停止位

    reg [2:0]  CS, NS;

    reg [7:0]  reg_data;
    reg [7:0]  reg_data_next;
    reg [2:0]  reg_rx_sync;
    reg [2:0]  reg_bit_count;
    reg [2:0]  reg_bit_count_next;

    reg [2:0]  s_target_bits;

    reg        parity_bit;
    reg        parity_bit_next;

    reg        sampleData;

    reg [15:0] baud_cnt;
    reg        baudgen_en;
    reg        bit_done;

    reg        start_bit;
    reg        set_error;
    wire       s_rx_fall;

    assign busy_o = (CS != IDLE);
    //当 CS 不是 IDLE 状态时，busy_o 被赋值为 1，表示接收模块正在处理数据，处于忙碌状态。

    always @(*) begin
        case (cfg_bits_i)//是一个 2 位的输入信号，用于配置数据位的数量
            2'b00: s_target_bits = 3'h4;//配置为接收 4 位数据
            2'b01: s_target_bits = 3'h5;//5位
            2'b10: s_target_bits = 3'h6;//6位
            2'b11: s_target_bits = 3'h7;//7位
        endcase
    end


    always @(*) begin
        NS = CS;//下一状态NS默认为当前状态CS
        sampleData = 1'b0;//默认不进行数据采样
        reg_bit_count_next = reg_bit_count;//默认下一位计数保持不变
        reg_data_next = reg_data;//接收的数据寄存器保持不变
        rx_valid_o = 1'b0;//默认输出的接收数据无效
        baudgen_en = 1'b0;//默认波特率生成器未启用
        start_bit = 1'b0;//默认不标识起始位
        parity_bit_next = parity_bit;//奇偶校验位保持不变
        set_error = 1'b0;//默认不设置错误标志

        case (CS)
        IDLE: begin//空闲状态 等待接收数据开始信号
                if (s_rx_fall) begin//这个条件用来检测接收数据线 rx_i 的下降沿
                    NS         = START_BIT;
                    baudgen_en = 1'b1;//启用波特率生成器，以便为接下来的数据接收提供时钟信号。
                    start_bit  = 1'b1;// 这个信号被设置为高，表明当前正在处理起始位
                end
        end
            START_BIT: begin
                parity_bit_next = 1'b0;//在接收起始位时，奇偶校验位的计数初始化为 0
                baudgen_en      = 1'b1;
                start_bit       = 1'b1;
                if (bit_done) NS = DATA;//该条件用于检查波特率生成器是否已完成起始位的处理
            end
            DATA: begin
                baudgen_en      = 1'b1;
                parity_bit_next = parity_bit ^ reg_rx_sync[2];
                //更新奇偶校验位。接收到的当前数据位 reg_rx_sync[2] 会与之前的奇偶校验结果进行异或运算，以计算新的奇偶校验位。

                case (cfg_bits_i)
                //根据配置的位数 (cfg_bits_i) 将接收到的数据位插入到数据寄存器 reg_data_next 中
                    2'b00: reg_data_next = {3'b0, reg_rx_sync[2], reg_data[4:1]};// 5 位数据，填充 3 个零
                    2'b01: reg_data_next = {2'b0, reg_rx_sync[2], reg_data[5:1]};// 6 位数据，填充 2 个零
                    2'b10: reg_data_next = {1'b0, reg_rx_sync[2], reg_data[6:1]};//7 位数据，填充 1 个零
                    2'b11: reg_data_next = {reg_rx_sync[2], reg_data[7:1]};//8 位数据，直接存储
                endcase

                if (bit_done) begin//该条件检查波特率生成器是否已经处理完当前位的接收
                    sampleData = 1'b1;//如果当前位接收完成，设置 sampleData 信号为高，表示需要将接收到的数据样本存入寄存器
                    if (reg_bit_count == s_target_bits) begin//检查 reg_bit_count 是否达到了目标位数 s_target_bits
                    	reg_bit_count_next = 'h0;
                    	NS = SAVE_DATA;//下一个状态 NS 设置为 SAVE_DATA，表示数据接收完成
	            end else begin
                    	reg_bit_count_next = reg_bit_count + 1;//继续接收下一个数据位
		    end
                end
            end
            SAVE_DATA: begin
                baudgen_en = 1'b1;
                rx_valid_o = 1'b1;//高 表示接收到的数据现在是有效的
                if (rx_ready_i) begin//检查接收端是否准备好接受新数据
                    if (cfg_parity_en_i) //如果配置中启用了奇偶校验（cfg_parity_en_i 为高）
                        NS = PARITY;//状态机转到 PARITY 状态
                    else 
                        NS = STOP_BIT;//未启用奇偶校验，则直接转到 STOP_BIT 状态
                end
            end
            PARITY: begin
                baudgen_en = 1'b1;
                if (bit_done) begin
                //表示当前的波特率计数已经达到了设定值，准备进行奇偶校验的检查。
                    case (cfg_parity_sel_i)
                    //语句根据配置的奇偶校验类型检查接收到的奇偶校验位
                        2'b00://偶校验
                            if (reg_rx_sync[2] != ~parity_bit) set_error = 1'b1;
                            //收到的奇偶校验位不等于计算得到的偶校验位（取反），则设置错误标志。
                        2'b01://奇校验
                            if (reg_rx_sync[2] != parity_bit) set_error = 1'b1;
                        2'b10://无奇偶校验
                            if (reg_rx_sync[2] != 1'b0) set_error = 1'b1;
                        2'b11://强制奇偶校验
                            if (reg_rx_sync[2] != 1'b1) set_error = 1'b1;
                    endcase
                    //完成奇偶校验位的检测后，状态机转移到 STOP_BIT 状态，准备检测停止位
                    NS = STOP_BIT;
                end
            end
            STOP_BIT: begin
                baudgen_en = 1'b1;
                if (bit_done) NS = IDLE;//如果 bit_done 信号为高，表示已经完成了停止位的接收
            end
            default: NS = IDLE;
        endcase
    end


    always @(posedge clk_i or negedge rstn_i) begin
        if (rstn_i == 1'b0) begin
            CS            <= IDLE;
            reg_data      <= 8'hff;
            reg_bit_count <= 'h0;
            parity_bit    <= 1'b0;
	end else begin
            if (bit_done)
                parity_bit <= parity_bit_next;//更新奇偶校验位为 parity_bit_next 的值
            if (sampleData)
                reg_data <= reg_data_next;//接收到的数据被存储到 reg_data 寄存器中

            reg_bit_count <= reg_bit_count_next;//更新位计数器，跟踪当前接收到的数据位数量

            if (cfg_en_i)//接收器配置使能信号 cfg_en_i 为高，状态寄存器 CS 更新为下一个状态 NS
                CS <= NS;
            else
                CS <= IDLE;
        end
    end

    assign s_rx_fall = ~reg_rx_sync[1] & reg_rx_sync[2];//s_rx_fall 将被设置为 1，表示发生了下降沿。
    /*
    ~reg_rx_sync[1]: 这是 reg_rx_sync 中的第二位取反，表示如果该位的值为 0（即之前的值为 1），那么当前接收到的信号是由高变低。
    reg_rx_sync[2]: 这是 reg_rx_sync 中的第三位，表示当前的接收到的信号值。如果该位为 1，则表示当前的信号是高电平。
    */

    always @(posedge clk_i or negedge rstn_i) begin
        if (rstn_i == 1'b0)
            reg_rx_sync <= 3'b111;//将 reg_rx_sync 初始化为 3'b111，表示高电平
        else if (cfg_en_i)//如果配置使能信号 (cfg_en_i) 为高，则将 reg_rx_sync 更新为当前接收信号和之前两位的组合
            reg_rx_sync <= {reg_rx_sync[1:0], rx_i};
        else
            reg_rx_sync <= 3'b111;
    end

    always @(posedge clk_i or negedge rstn_i) begin//波特率计数
        if (rstn_i == 1'b0) begin//复位信号有效
            baud_cnt <= 'h0;//波特率计数器 0
            bit_done <= 1'b0;//位完成信号 0
	end else if (baudgen_en) begin//波特率生成使能信号 (baudgen_en) 为高，表示正在计数
            if (!start_bit && (baud_cnt == cfg_div_i)) begin
            //当前不是起始位，并且计数器达到配置的波特率除数 (cfg_div_i)
                baud_cnt <= 'h0;
                bit_done <= 1'b1;//表示一个位的计数完成
        end else if (start_bit && (baud_cnt == {1'b0, cfg_div_i[15:1]})) begin
        //当前是起始位，计数器需要达到 cfg_div_i 的一半（即波特率的调整）
                baud_cnt <= 'h0;
                bit_done <= 1'b1;
        end else begin
                baud_cnt <= baud_cnt + 1;
                bit_done <= 1'b0;
            end
	end else begin
            baud_cnt <= 'h0;
            bit_done <= 1'b0;
        end
    end


    always @(posedge clk_i or negedge rstn_i)//错误标志处理
        if (rstn_i == 1'b0)//复位信号有效
            err_o <= 1'b0;//输出0
        else if (err_clr_i)//错误清除信号 err_clr_i 为高
            err_o <= 1'b0;//输出0 表示清除错误状态
        else if (set_error)//设置错误信号 set_error 为高
            err_o <= 1'b1;//将错误输出 err_o 置为 1，表示发生了错误

    assign rx_data_o = reg_data;
    //将内部寄存器 reg_data 的值直接赋给输出信号 rx_data_o。这意味着 rx_data_o 始终反映 reg_data 的当前值

endmodule