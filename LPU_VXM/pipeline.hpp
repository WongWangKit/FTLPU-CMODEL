#pragma once
#include <cstdint>
#include "half.hpp" 
#include <utility>   // for std::pair
//enable信号和数据计算同步
//define data type
using INT32 = std::int32_t;
using INT8 = std::int8_t;
// FP16 placeholder: change to your half type if available
using FP16 = half;

// each pipeline function
enum class BiasOp {
    BYPASS,
    ADD
};
enum class ConvertOp {
    BYPASS,
    INT32_TO_FP16
};
enum class ActOp {
    BYPASS,
    RELU,
    GELU
};

// STAGE 0
class bias_stage {
public:
    bias_stage() 
        : m_bias(0), m_result(0), m_config(BiasOp::BYPASS), m_enable(false) {}

    // 设置偏置值（可动态更新）
    void set_bias(INT32 bias) {m_bias = bias;}
    // 设置工作模式
    void set_config(BiasOp config) {m_config = config;}
    // 获取当前结果寄存器值（用于后续阶段读取）
    INT32 get_result() const {return m_result;}
    bool get_enable() const { return m_enable; }
    // 核心运算：输入一个 INT32，根据配置计算并存入结果寄存器，同时返回结果
    INT32 bias_int32(INT32 input) {
        return m_result = input + m_bias;   // 有符号整数加法，溢出行为由实现定义，符合需求
    }
    void execute(INT32 input, bool valid_in) {
        m_enable = valid_in;
        if (m_config == BiasOp::ADD) {
            if(m_enable)
                m_result = bias_int32(input);   // 调用运算
        } else if(m_config==BiasOp::BYPASS) { // BYPASS
            if(m_enable)
                m_result = input;                 // 直通
        }
        // 若需延迟一拍，可在额外时钟函数中控制，这里直接组合逻辑更新
    }

private:
    INT32   m_bias;      // 偏置值
    INT32   m_result;    // 结果寄存器（流水线输出）
    BiasOp  m_config;    // 当前配置
    bool    m_enable;    // 寄存器
};

// STAGE 1
class convert_stage {
public:
    convert_stage()
        : m_scale(1.0f)
        , m_zero_point(0)
        , m_config(ConvertOp::BYPASS)
        , m_result(FP16{0.0f})
        , m_enable(false) {}

    // ---- 配置接口 ----
    void set_scale(float scale)         { m_scale = scale; }
    void set_zero_point(INT32 zp)       { m_zero_point = zp; }
    void set_config(ConvertOp config)   { m_config = config; }
    bool get_enable() const             { return m_enable; }
    // ---- 读取结果寄存器 ----
    FP16 get_result() const { return m_result; }

    // ---- 纯计算函数：仅做反量化，不修改成员变量 ----
    FP16 compute_convert(INT32 input) const {
        // 1. 整数减法（仍为 INT32）
        INT32 diff = input - m_zero_point;
        // 2. 提升为 float（保持精度）
        float diff_f32 = static_cast<float>(diff);
        // 3. 乘以 scale
        float result_f32 = diff_f32 * m_scale;
        // 4. 饱和到 FP16 可表示的最大有限值
        constexpr float FP16_MAX = 65504.0f;
        if (result_f32 > FP16_MAX)   result_f32 = FP16_MAX;
        if (result_f32 < -FP16_MAX)  result_f32 = -FP16_MAX;
        // 5. 转换为 FP16（half），并返回
        return static_cast<FP16>(result_f32);
    }

    // ---- 执行函数：根据配置更新结果寄存器 ----
    void execute(INT32 input, bool valid_in) {
        m_enable = valid_in;  
        if (m_config == ConvertOp::INT32_TO_FP16) {
            if(m_enable)
                m_result = compute_convert(input);
        } else if(m_config == ConvertOp::BYPASS){ // BYPASS
            // 直通：输入是 INT32，但结果寄存器是 FP16，需做类型提升
            // 直接将 INT32 转为 FP16（若硬件支持直接转换，此处模拟）
            if(m_enable)
                m_result = static_cast<FP16>(static_cast<float>(input));
        }
    }
private:
    float   m_scale;      // 缩放因子（浮点）
    INT32   m_zero_point; // 零点（整型）
    ConvertOp m_config;   // 当前配置
    FP16    m_result;     // 结果寄存器（输出）
    bool    m_enable;
};

//STAGE 2
class act_stage {
public:
    act_stage()
        : m_config(ActOp::BYPASS)
        , m_const_sqrt_2_pi(static_cast<FP16>(0.7978845608028654f)) // sqrt(2/π)
        , m_result(FP16{0.0f}) 
        , m_enable(false)   // 输出有效标志
    {
        for (int i = 0; i < 3; ++i) {
            m_reg[i] = FP16{0.0f};
            m_stage_enable[i] = false;   // 内部各级的有效标志
        }
    }

    // ---- 配置 ----
    void set_config(ActOp config) { m_config = config; }
    // ---- 读取结果寄存器 ----
    FP16 get_result() const { return m_result; }
    bool get_enable() const { return m_enable; } 
    // ---- 执行函数（流水线推进） ----
void execute(FP16 input,bool valid_in) {
        switch (m_config) {
            case ActOp::BYPASS:
                // 直通：输入直接输出
                m_enable = valid_in;
                if(m_enable)
                    m_result = input;
                // 可选：清空流水线寄存器（但非必须）
                // m_reg[0] = m_reg[1] = m_reg[2] = FP16{0.0f};
                break;
            case ActOp::RELU:
                // 单周期 ReLU
                m_enable = valid_in;
                if(m_enable)
                    m_result = compute_relu(input);
                break;
            case ActOp::GELU: {
                // ---- 三级流水线，倒序更新 ----
                // 寄存器分配：
                //   m_reg[0] : 原始 x（来自 Stage1）
                //   m_reg[1] : inner（来自 Stage1，供 Stage2 使用）
                //   m_reg[2] : tanh（来自 Stage2，供 Stage3 使用）
                //   m_result : 最终结果（Stage3 输出）
                m_enable = m_stage_enable[1];
                m_stage_enable[1] = m_stage_enable[0];
                m_stage_enable[0] = valid_in;
                // 1. Stage3（使用旧的 m_reg[0] 和 m_reg[2]）
                if(m_enable)
                {
                    FP16 old_x = m_reg[0];
                    FP16 old_tanh = m_reg[2];
                    m_result = compute_gelu_stage3(old_tanh, old_x);
                }
                // 2. Stage2（使用旧的 m_reg[1]）
                if(m_stage_enable[1])
                {
                    FP16 old_inner = m_reg[1];
                    FP16 new_tanh = compute_gelu_stage2(old_inner);
                    m_reg[2] = new_tanh;   // 写入 Stage2 结果
                }
                // 3. Stage1（使用当前输入）
                if(m_stage_enable[0])
                {
                    auto [inner, x] = compute_gelu_stage1(input);
                    m_reg[0] = x;          // 保存原始 x
                    m_reg[1] = inner;      // 保存 inner
                }
                break;
            }
        }
    }
    // ---- 纯计算函数（不修改成员） ----
    // 1. ReLU
    FP16 compute_relu(FP16 x) const {
        return (x > FP16{0.0f}) ? x : FP16{0.0f};
    }
    // 2. GELU 第一阶段：计算 inner = x + 0.044715 * x^3，同时返回原始 x
    std::pair<FP16, FP16> compute_gelu_stage1(FP16 x) const {
        float x_f = static_cast<float>(x);
        float x3 = x_f * x_f * x_f;
        float inner = x_f + 0.044715f * x3;
        return { static_cast<FP16>(inner), x };   // inner 和原始 x
    }
    // 3. GELU 第二阶段：计算 tanh( sqrt(2/π) * inner )
    FP16 compute_gelu_stage2(FP16 inner) const {
        float inner_f = static_cast<float>(inner);
        float scaled = static_cast<float>(m_const_sqrt_2_pi) * inner_f;
        float tanh_val = tanhf(scaled);
        return static_cast<FP16>(tanh_val);
    }
    // 4. GELU 第三阶段：计算 0.5 * x * (1 + tanh)
    FP16 compute_gelu_stage3(FP16 tanh_val, FP16 x) const {
        float tanh_f = static_cast<float>(tanh_val);
        float x_f = static_cast<float>(x);
        float result = 0.5f * x_f * (1.0f + tanh_f);
        // 饱和到 FP16 范围
        constexpr float FP16_MAX = 65504.0f;
        if (result > FP16_MAX) result = FP16_MAX;
        if (result < -FP16_MAX) result = -FP16_MAX;
        return static_cast<FP16>(result);
    }
private:
    ActOp m_config;
    FP16  m_const_sqrt_2_pi;   // sqrt(2/π) 以 FP16 存储
    FP16  m_result;            // 结果寄存器（输出）
    bool  m_enable;                // 最终输出有效标志

    FP16 m_reg[3];            // 中间结果寄存器
    bool  m_stage_enable[2];       // m_stage_enable[0]: Stage1 输出有效, m_stage_enable[1]: Stage2 输出有效
};