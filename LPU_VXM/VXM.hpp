#pragma once
#include "pipeline.hpp"   // 包含 Decoder, bias_stage, convert_stage, act_stage
#include "decoder.hpp"
#include <fstream>
#include <iomanip>
#include <string>
extern INT32 input_data;
class VXM {
public:
    VXM()
        : inst_fifo_head(0), inst_fifo_tail(0), inst_fifo_count(0)
        , in_fifo_head(0), in_fifo_tail(0), in_fifo_count(0)
        , out_fifo_head(0), out_fifo_tail(0), out_fifo_count(0)
        , m_input(0)
        , vector_number(0)
        , vector_counter_input(0)
        , vector_counter_output(0)
        , cycle_count(0)
    {   
        out_file.open("VXM_test_result.txt");
        if (out_file.is_open()) {
            out_file << "cycle\tbias\tconvert\tact\toutput\tstatus\n";
        }
    }
    ~VXM() {
        if (out_file.is_open()) {
            out_file.close();
        }
    }
    // ---- 向 FIFO 推送一条指令（最多 3 条） ----
      bool push_instruction(const VXMInstruction& inst) {
        if (inst_fifo_count >= FIFO_SIZE) return false;
        inst_fifo[inst_fifo_tail] = inst;
        inst_fifo_tail = (inst_fifo_tail + 1) % FIFO_SIZE;
        inst_fifo_count++;
        return true;
    }

    bool push_input(INT32 data) {
        if (in_fifo_count >= FIFO_SIZE) return false;
        in_fifo[in_fifo_tail] = data;
        in_fifo_tail = (in_fifo_tail + 1) % FIFO_SIZE;
        in_fifo_count++;
        return true;
    }

    bool pop_input(INT32& data) {
        if (in_fifo_count == 0) return false;
        data = in_fifo[in_fifo_head];
        in_fifo_head = (in_fifo_head + 1) % FIFO_SIZE;
        in_fifo_count--;
        return true;
    }

    bool push_output(FP16 data) {
        if (out_fifo_count >= FIFO_SIZE) return false;
        out_fifo[out_fifo_tail] = data;
        out_fifo_tail = (out_fifo_tail + 1) % FIFO_SIZE;
        out_fifo_count++;
        return true;
    }

    bool pop_output(FP16& data) {
        if (out_fifo_count == 0) return false;
        data = out_fifo[out_fifo_head];
        out_fifo_head = (out_fifo_head + 1) % FIFO_SIZE;
        out_fifo_count--;

        return true;
    }

    // ---- 指令配置：解码并配置各流水级，重置状态 ----
    void try_configure_from_fifo() {
        if (inst_fifo_count == 0) {
            return;   // 无指令可配置
        }

        // 从指令 FIFO 弹出指令
        VXMInstruction inst = inst_fifo[inst_fifo_head];
        inst_fifo_head = (inst_fifo_head + 1) % FIFO_SIZE;
        inst_fifo_count--;

        // 解码并配置各阶段
        decoder.decoder_inst(inst);
        bias.set_config(decoder.bias_reg);
        convert.set_config(decoder.convert_reg);
        act.set_config(decoder.act_reg);

        // 设置计数器
        vector_number = decoder.vector_num_reg;
        vector_counter_input = 0;
        vector_counter_output = 0;

        // 记录配置事件
        log_status("configure");
        // 流水线有效标志由各模块内部维护，配置后均为 false，故无需额外清空
    }

    // ---- 设置各阶段所需的参数（由外部控制提供） ----
    void set_bias_value(INT32 bias_val) {
        bias.set_bias(bias_val);
    }
    void set_scale(float scale) {
        convert.set_scale(scale);
    }
    void set_zero_point(INT32 zp) {
        convert.set_zero_point(zp);
    }
    // ---- 每周期注入输入数据 ----
    void feed_input(INT32 input) {
        m_input = input;
    }
    
void tick() {
    // 1. 检查是否所有工作已完成（无指令在执行，且计数器均为0）
    //    此时表示空闲状态，可进行指令装配（由外部通过 configure() 完成）
    cycle_count++;  // 增加周期计数
    FP16 result;
    bool bias_in_valid = false;
    if (vector_number == 0 && vector_counter_input == 0 && vector_counter_output == 0) {
        // 空闲状态，通常由外部调用 configure() 来装载新指令。
        // 这里不做任何操作，仅返回，让外部有机会配置。
        try_configure_from_fifo();
        return;
    }
    else if (vector_number != 0 && vector_counter_output < vector_number) {
        bias_in_valid = false;
        //2.判断本周期是否需要进行输入数据
        if(vector_counter_input<vector_number){
            bias_in_valid = true;   // 还有数据待注入
        }
        // 3. 按数据流方向执行各阶段（每个阶段内部已处理倒序和有效标志）
        act.execute(convert.get_result(), convert.get_enable());
        convert.execute(bias.get_result(), bias.get_enable());
        // test, 实际这个数据来自前一级流水线
        if(bias_in_valid)
        {
            input_data++;
            feed_input(input_data);
            vector_counter_input++;
        }
        bias.execute(m_input, bias_in_valid);
        if (act.get_enable()) {
            vector_counter_output++;  // 输出一个有效结果
            push_output(act.get_result());
            pop_output(result);
        }
        //print message
        FP16 act_out = FP16{0.0f};
        FP16 convert_out = FP16{0.0f};
        INT32 bias_out = 0;
        bool bias_en = false, convert_en = false, act_en = false;
        bias_out = bias.get_result();
        bias_en = bias.get_enable();
        convert_out = convert.get_result();
        convert_en = convert.get_enable();
        act_out = act.get_result();
        act_en = act.get_enable();
        log_pipeline_status(bias_out, bias_en, convert_out, convert_en, act_out, act_en);

        return;
    } else {
        // 当前指令已完成，保留 FIFO 中的后续指令，等待下一周期配置
        vector_number = 0;
        vector_counter_input = 0;
        vector_counter_output = 0;
        m_input = 0;
        log_status("complete");
        return;
    }
}

    // ---- 读取输出结果（仅当 act.get_enable() 为 true 时有效） ----
    FP16 get_output() const {
        return act.get_result();
    }
/*
    // ---- 查询输出是否有效 ----
    bool has_output() const {
        return act.get_enable();
    }
    // ---- 获取剩余未输出的向量数 ----
    UINT32 get_remaining() const {
        if (vector_number == 0) return 0;
        return (vector_counter_output >= vector_number) ? 0 : (vector_number - vector_counter_output);
    }
*/
    // ---- 指令是否已完成（所有数据已输出） ----
    bool is_complete() const {
        return (vector_number != 0) && (vector_counter_output >= vector_number);
    }
    // ---- 重置流水线状态（可选） ----
    void reset() {
    // 1. 重置指令计数器
    vector_number = 0;
    vector_counter_input = 0;
    vector_counter_output = 0;
    m_input = 0;

    // 2. 重置所有 FIFO（清空队列）
    inst_fifo_head = 0;
    inst_fifo_tail = 0;
    inst_fifo_count = 0;

    in_fifo_head = 0;
    in_fifo_tail = 0;
    in_fifo_count = 0;

    out_fifo_head = 0;
    out_fifo_tail = 0;
    out_fifo_count = 0;

    log_status("reset");
    // 3. 注意：bias、convert、act 模块内部的 enable 标志和结果寄存器未重置。
    //    但这些模块会在下次配置新指令后，由新的数据流自然覆盖旧状态。
    //    如果需要完全干净的流水线，可在调用 reset 后，再连续调用 tick() 若干次（传入无效输入）直到流水线排空。
    //    更彻底的方式是为各模块添加 reset 方法，但当前设计中不包含。
}
 void log_pipeline_status(INT32 bias_val, bool bias_en,
                             FP16 convert_val, bool convert_en,
                             FP16 act_val, bool act_en) {
        if (!out_file.is_open()) return;

        out_file << cycle_count << "\t";
        // bias
        if (bias_en) out_file << bias_val;
        else out_file << "N/A";
        out_file << "\t";
        // convert
        if (convert_en) out_file << convert_val;
        else out_file << "N/A";
        out_file << "\t";
        // act
        if (act_en) out_file << act_val;
        else out_file << "N/A";
        out_file << "\t";
        // output (从输出 FIFO 中取出的值？我们直接打印 act_en 时的 act_val，因为 push 了)
        // 为了显示实际输出的值，我们可以在这里打印 act_val 当 act_en 为真时。
        // 但既然已经打印了 act 列，重复。不如打印输出 FIFO 的头部，但可能外部还没读。
        // 我们简单打印 act_val（如果 act_en 为真）。
        if (act_en) out_file << act_val;
        else out_file << "N/A";
        out_file << "\t";
        out_file << "running\n";
    }

    void log_status(const std::string& status) {
        if (!out_file.is_open()) return;
        // 当状态为 idle, configure, reset, complete 时，各列打印 N/A
        out_file << cycle_count << "\tN/A\tN/A\tN/A\tN/A\t" << status << "\n";
    }

private:
    static constexpr int FIFO_SIZE = 3;
     // 指令 FIFO
    VXMInstruction inst_fifo[FIFO_SIZE];
    int inst_fifo_head, inst_fifo_tail, inst_fifo_count;

    // 输入数据 FIFO (INT32)
    INT32 in_fifo[FIFO_SIZE];
    int in_fifo_head, in_fifo_tail, in_fifo_count;

    // 输出数据 FIFO (FP16)
    FP16 out_fifo[FIFO_SIZE];
    int out_fifo_head, out_fifo_tail, out_fifo_count;

    Decoder decoder;
    bias_stage bias;
    convert_stage convert;
    act_stage act;

    INT32 m_input;              // 当前周期注入的数据
    UINT32 vector_number;       // 指令总向量数
    UINT32 vector_counter_input;  // 已注入流水线的向量数
    UINT32 vector_counter_output; // 已从流水线输出的向量数

    unsigned long long cycle_count;
    std::ofstream out_file;
};