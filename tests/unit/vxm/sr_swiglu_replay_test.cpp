#include "ftlpu/vxm/sr_datapath.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace ftlpu;

// Exercise both mirrored physical chains with changing operands, including
// the first and last rows. Constant data would hide an eight-row skew.
void check_replay(bool westward)
{
    auto owned = std::make_unique<VxmSrDatapath>();
    auto& datapath = *owned;
    VxmSpecialAlu reference;
    const auto configure = [&](VxmSpecialAluOpcode opcode, float lo,
                               float width, auto fn) {
        std::vector<VxmLutEntry> entries;
        for (int i = 0; i < 256; ++i) {
            const float x = lo + i * width;
            entries.push_back(VxmLutEntry::from_float(
                (fn(x + width) - fn(x)) / width, fn(x)));
        }
        datapath.configure_special_lut(opcode, {lo, width}, entries);
        reference.configure_lut(opcode, {lo, width}, entries);
    };
    constexpr float ln2 = 0.6931471805599453f;
    configure(VxmSpecialAluOpcode::Exp, -ln2 / 2, ln2 / 256,
        [](float x) { return std::exp(x); });
    configure(VxmSpecialAluOpcode::Reciprocal, 1, 1.0f / 256,
        [](float x) { return 1 / x; });
    const auto stream = [&](std::size_t s) {
        return westward ? StreamId::West(s) : StreamId::East(s);
    };
    const auto column = [&](std::size_t s) {
        return westward ? hw::kVxmAluCount - 1 - s : s;
    };
    const auto carrier = [&](std::size_t s) {
        const auto c = column(s);
        return stream((c % 16 < 8 ? 16 : 0) + 4 * (c % 4));
    };
    const std::size_t gateStream = westward ? 16 : 8;
    const std::size_t ingress = westward ? hw::kVxmAluCount : 0;
    const std::size_t egress = westward ? 0 : hw::kVxmAluCount;
    const std::size_t output = westward ? 14 : 6;
    VxmSrDatapath::Instructions instructions {};
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t stage = 0; stage < 6; ++stage) {
            VxmSrInstruction op;
            op.precision = VxmAluPrecision::Float32;
            op.result_type = stage == 5 ? VxmCastTarget::BFloat16 : VxmCastTarget::Float32;
            op.result_sr = stage == 5 ? stream(output).packed() : carrier(stage).packed();
            op.lhs = stage == 0
                ? VxmSrOperand::StreamBFloat16(stream(gateStream).packed())
                : VxmSrOperand::StreamFloat32(carrier(stage - 1).packed());
            op.rhs = VxmSrOperand::Immediate(stage == 2 ? 1 : 0);
            switch (stage) {
            case 0: op.operation = VxmAluOpcode::Negate; break;
            case 1: op.operation = VxmSpecialAluOpcode::Exp; break;
            case 2: op.operation = VxmAluOpcode::Add; break;
            case 3: op.operation = VxmSpecialAluOpcode::Reciprocal; break;
            default:
                op.operation = VxmAluOpcode::Multiply;
                op.rhs = VxmSrOperand::StreamBFloat16(
                    stream(gateStream + (stage == 4 ? 4 : 2)).packed());
            }
            instructions[tile][column(stage)] =
                VxmSrCompactInstructionCodec::decode(VxmSrCompactInstructionCodec::encode(op));
        }
    }
    constexpr std::size_t count = 257;
    const auto bf16 = [](float x) {
        return VxmDataFormat::bf16_bits_to_float(VxmDataFormat::float_to_bf16_bits(x));
    };
    const auto gate = [&](std::size_t row, std::size_t tile, std::size_t lane) {
        return bf16(-1.5f + 0.125f * ((row + tile * 3 + lane) % 25));
    };
    const auto up = [&](std::size_t row, std::size_t tile, std::size_t lane) {
        return bf16(0.25f + 0.125f * ((row * 7 + tile + lane * 3) % 19));
    };
    auto& sr = datapath.fabric().registers();
    std::size_t seen = 0, first = 0;
    for (std::size_t cycle = 0; cycle < count + 2 * hw::kVxmAluCount + 40; ++cycle) {
        datapath.begin_cycle();
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto put = [&](std::size_t s, float x) {
                    const auto bits = VxmDataFormat::float_to_bf16_bits(x);
                    for (std::size_t byte = 0; byte < 2; ++byte)
                        sr.stage_write(ingress, tile, lane, stream(s + byte),
                            StreamCell::Valid(static_cast<std::uint8_t>(bits >> (8 * byte))));
                };
                if (cycle < count) put(gateStream, gate(cycle, tile, lane));
                if (cycle >= 8 && cycle - 8 < count)
                    put(gateStream + 4, gate(cycle - 8, tile, lane));
                if (cycle >= 9 && cycle - 9 < count)
                    put(gateStream + 2, up(cycle - 9, tile, lane));
            }
        datapath.execute(instructions);
        datapath.finish_cycle();
        if (!sr.segment_valid(egress, 0, stream(output))) continue;
        if (seen == 0) first = cycle;
        if (seen >= count || cycle != first + seen)
            throw std::logic_error("SwiGLU replay output count or cadence mismatch");
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto bits = sr.cell(egress, tile, lane, stream(output)).data
                    | (sr.cell(egress, tile, lane, stream(output + 1)).data << 8);
                const float actual = VxmDataFormat::bf16_bits_to_float(bits);
                const float g = gate(seen, tile, lane);
                const float expected = bf16(g * reference.execute(
                    VxmSpecialAluOpcode::Reciprocal,
                    1 + reference.execute(VxmSpecialAluOpcode::Exp, -g))
                    * up(seen, tile, lane));
                if (actual != expected)
                    throw std::logic_error("SwiGLU replay numeric mismatch at row " + std::to_string(seen));
            }
        ++seen;
    }
    if (seen != count) throw std::logic_error("SwiGLU replay lost its tail");
    if (first != hw::kVxmAluCount + 10)
        throw std::logic_error("SwiGLU replay latency changed");
    std::cout << (westward ? "W" : "E") << " columns=" << hw::kVxmAluCount
              << " rows=" << seen << " first=" << first << " PASS\n";
}

int main() try {
    check_replay(false);
    check_replay(true);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
