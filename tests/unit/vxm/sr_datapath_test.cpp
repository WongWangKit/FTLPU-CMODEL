#include "ftlpu/vxm/sr_datapath.hpp"

#include <cassert>
#include <cstring>

namespace {

void check_swiglu_multiply_streams(bool westward)
{
    using namespace ftlpu;
    VxmSrDatapath datapath;
    auto& sr = datapath.fabric().registers();
    const auto local = [westward](std::size_t index) {
        return westward ? StreamId::West(index) : StreamId::East(index);
    };
    const auto opposite = [westward](std::size_t index) {
        return westward ? StreamId::East(index) : StreamId::West(index);
    };
    const std::size_t head = westward ? hw::kVxmAluCount - 1 : 0;
    const std::size_t tail = westward ? head - 1 : 1;
    const std::size_t input = westward ? hw::kVxmAluCount : 0;
    const std::size_t output = westward ? 0 : hw::kVxmAluCount;
    const std::size_t gateStream = westward ? 16 : 8;
    const std::size_t temporary = westward ? 12 : 16;
    const std::size_t result = westward ? 14 : 6;
    VxmSrDatapath::Instructions issued {};
    VxmSrInstruction multiply;
    multiply.operation = VxmAluOpcode::Multiply;
    multiply.precision = VxmAluPrecision::Float32;
    multiply.lhs = VxmSrOperand::StreamBFloat16(local(gateStream).packed());
    multiply.rhs = VxmSrOperand::StreamFloat32(opposite(20).packed());
    multiply.result_sr = local(temporary).packed();
    multiply.result_type = VxmCastTarget::Float32;
    issued[0][head] = multiply;
    multiply.lhs = VxmSrOperand::StreamFloat32(local(temporary).packed());
    multiply.rhs = VxmSrOperand::StreamBFloat16(local(gateStream + 2).packed());
    multiply.result_sr = local(result).packed();
    multiply.result_type = VxmCastTarget::BFloat16;
    issued[0][tail] = multiply;
    for (std::size_t cycle = 0; cycle <= 33; ++cycle) {
        datapath.begin_cycle();
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            if (cycle == 0) {
                float sigmoid = 0.75f;
                std::uint32_t bits;
                std::memcpy(&bits, &sigmoid, sizeof(bits));
                for (std::size_t byte = 0; byte < 4; ++byte)
                    sr.stage_write(output, 0, lane, opposite(20 + byte),
                        StreamCell::Valid(static_cast<std::uint8_t>(bits >> (8 * byte))));
            }
            if (cycle == 15 || cycle == 16) {
                const auto bits = VxmDataFormat::float_to_bf16_bits(
                    cycle == 15 ? 2.0f : 3.0f);
                const std::size_t stream = gateStream + (cycle == 15 ? 0 : 2);
                for (std::size_t byte = 0; byte < 2; ++byte)
                    sr.stage_write(input, 0, lane, local(stream + byte),
                        StreamCell::Valid(static_cast<std::uint8_t>(bits >> (8 * byte))));
            }
        }
        datapath.execute(issued);
        datapath.finish_cycle();
        if (sr.segment_valid(output, 0, local(result)) != (cycle == 33))
            throw std::logic_error("SwiGLU multiply result arrived at the wrong SR cycle");
    }
    const auto bits = static_cast<std::uint16_t>(sr.cell(output, 0, 0, local(result)).data)
        | (static_cast<std::uint16_t>(sr.cell(output, 0, 0, local(result + 1)).data) << 8);
    if (VxmDataFormat::bf16_bits_to_float(bits) != 4.5f)
        throw std::logic_error("SwiGLU multiply consumed misaligned operands");
}

} // namespace

int main()
{
    using namespace ftlpu;
    VxmSrDatapath datapath {};
    auto& sr = datapath.fabric().registers();
    const auto lhs = StreamId::East(0);
    const auto rhs = StreamId::East(2);
    const auto untouched = StreamId::East(8);
    const auto output = StreamId::East(4);
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        const auto one = VxmDataFormat::float_to_fp16_bits(1.0f + lane);
        const auto two = VxmDataFormat::float_to_fp16_bits(2.0f);
        sr.initialize_cell(0, 0, lane, lhs,
            StreamCell::Valid(static_cast<std::uint8_t>(one)));
        sr.initialize_cell(0, 0, lane, StreamId::East(1),
            StreamCell::Valid(static_cast<std::uint8_t>(one >> 8)));
        sr.initialize_cell(0, 0, lane, rhs,
            StreamCell::Valid(static_cast<std::uint8_t>(two)));
        sr.initialize_cell(0, 0, lane, StreamId::East(3),
            StreamCell::Valid(static_cast<std::uint8_t>(two >> 8)));
        sr.initialize_cell(0, 0, lane, untouched,
            StreamCell::Valid(static_cast<std::uint8_t>(40 + lane)));
    }

    VxmSrDatapath::Instructions issued {};
    VxmSrInstruction add {};
    add.operation = VxmAluOpcode::Add;
    add.lhs = VxmSrOperand::StreamFloat16(lhs.packed());
    add.rhs = VxmSrOperand::StreamFloat16(rhs.packed());
    add.result_sr = static_cast<std::uint8_t>(output.packed());
    add.result_type = VxmCastTarget::Float16;
    issued[0][0] = add;
    const auto executed = datapath.tick(issued);
    assert(executed[0][0]);
    assert(!sr.segment_valid(1, 0, lhs));
    assert(!sr.segment_valid(1, 0, rhs));
    assert(sr.segment_valid(1, 0, output));
    assert(sr.segment_valid(1, 0, untouched));
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        const auto low = sr.cell(1, 0, lane, output).data;
        const auto high = sr.cell(1, 0, lane, StreamId::East(5)).data;
        const auto bits = static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8);
        assert(VxmDataFormat::fp16_bits_to_float(bits)
            == 3.0f + static_cast<float>(lane));
    }
    check_swiglu_multiply_streams(false);
    check_swiglu_multiply_streams(true);
    return 0;
}
