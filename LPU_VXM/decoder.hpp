#pragma once
#include "pipeline.hpp"

//define inst needs
using Opcode = uint8_t;
using UINT32 = uint32_t;

// instruction and data definition
struct VXMInstruction {
    Opcode opcode;          // Bias+GELU、Bias+ReLU等
    UINT32 vector_count;    // 连续处理多少个Vector
};


class Decoder
{
public:
    Decoder();
    // Instruction Decode
    void decoder_inst(const VXMInstruction& inst);
public:
    // Pipeline Configuration Registers
    BiasOp bias_reg;
    ConvertOp convert_reg;
    ActOp act_reg;
    // Number of vectors to execute
    UINT32 vector_num_reg;
};

Decoder::Decoder()
{
    bias_reg = BiasOp::BYPASS;
    convert_reg = ConvertOp::BYPASS;
    act_reg = ActOp::BYPASS;

    vector_num_reg = 0;
}

//from inst input to decoder 3 sets of regs
void Decoder::decoder_inst(const VXMInstruction& inst)
{
    // Save vector number
    vector_num_reg = inst.vector_count;

    // bit3 : Bias
    bias_reg =
        ((inst.opcode >> 3) & 0x1) ?
        BiasOp::ADD :
        BiasOp::BYPASS;

    // bit2 : Convert
    convert_reg =
        ((inst.opcode >> 2) & 0x1) ?
        ConvertOp::INT32_TO_FP16 :
        ConvertOp::BYPASS;

    // bit1~0 : Activation
    switch(inst.opcode & 0x3)
    {
        case 0:
            act_reg = ActOp::BYPASS;
            break;

        case 1:
            act_reg = ActOp::RELU;
            break;

        case 2:
            act_reg = ActOp::GELU;
            break;

        default:
            act_reg = ActOp::BYPASS;
            break;
    }
}
