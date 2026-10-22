#include "ftlpu/vxm/superlane.hpp"
#include "hardware_test_output.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
using namespace ftlpu;

template<typename Fn>
std::vector<VxmLutEntry> make_table(float start, float width, Fn fn)
{
    std::vector<VxmLutEntry> entries;
    for (std::size_t i = 0; i < 128; ++i) {
        const auto x = start + static_cast<float>(i) * width;
        entries.push_back(VxmLutEntry::from_float(
            (fn(x + width) - fn(x)) / width, fn(x)));
    }
    return entries;
}

void configure_log(VxmSpecialAlu& alu)
{
    alu.configure_lut(VxmSpecialAluOpcode::Log, {1.0f, 1.0f / 128.0f},
        make_table(1.0f, 1.0f / 128.0f, [](float x) { return std::log(x); }));
}

void verify_numerics()
{
    VxmSpecialAlu alu;
    configure_log(alu);
    assert(alu.execute(VxmSpecialAluOpcode::Log, 1.0f) == 0.0f);
    // Exhaust every positive normal FP16 input, including both sides of 1,
    // every LUT boundary, and the full exponent range. Error is measured
    // against the exact log of the represented input, not an unrounded float.
    for (std::uint32_t bits = 0x0400; bits < 0x7c00; ++bits) {
        const auto x = VxmDataFormat::fp16_bits_to_float(static_cast<std::uint16_t>(bits));
        const auto lookup = alu.make_lookup(VxmSpecialAluOpcode::Log, x);
        assert(lookup.index == ((bits & 0x03ffu) >> 3));
        assert(lookup.dx == static_cast<float>(bits & 7u) / 1024.0f);
        const auto result = alu.complete_lookup(lookup);
        assert(std::fabs(result - std::log(x)) <= 0.008f);
    }
    const auto inf = std::numeric_limits<float>::infinity();
    for (const auto x : {0.0f, -0.0f, 0x1p-20f, -0x1p-20f}) {
        assert(alu.execute(VxmSpecialAluOpcode::Log, x) == -inf);
    }
    assert(std::isnan(alu.execute(VxmSpecialAluOpcode::Log, -1.0f)));
    assert(std::isnan(alu.execute(VxmSpecialAluOpcode::Log, -inf)));
    assert(std::isnan(alu.execute(VxmSpecialAluOpcode::Log,
        std::numeric_limits<float>::quiet_NaN())));
    assert(alu.execute(VxmSpecialAluOpcode::Log, inf) == inf);

    // Deliberately incorrect coefficients prove execution uses the configured
    // SRAM contents and still applies the exponent compensation.
    alu.configure_lut(VxmSpecialAluOpcode::Log, {1.0f, 1.0f},
        {VxmLutEntry::from_float(0.0f, 0.25f)});
    assert(alu.execute(VxmSpecialAluOpcode::Log, 1.5f) == 0.25f);
    assert(alu.execute(VxmSpecialAluOpcode::Log, 6.0f)
        == VxmDataFormat::round_fp16_ftz(0.25f + 2.0f * std::log(2.0f)));
}

void verify_pipeline()
{
    VxmSpecialAlu alu;
    configure_log(alu);
    VxmSpecialAlu::Pipeline<std::size_t> pipeline;
    const std::vector<float> inputs{1.0f, 6.0f, 0.0f, -1.0f, 0.5f};
    const auto delay = VxmSpecialAlu::kPipelineLatency - 1;
    for (std::size_t cycle = 0; cycle < inputs.size() + delay; ++cycle) {
        using Request = VxmSpecialAlu::Pipeline<std::size_t>::Request;
        auto result = pipeline.tick(alu, cycle < inputs.size()
            ? std::optional<Request>{Request{VxmSpecialAluOpcode::Log, inputs[cycle], cycle}}
            : std::nullopt);
        if (cycle < delay) {
            assert(!result);
        } else {
            const auto index = cycle - delay;
            assert(result && result->metadata == index);
            const auto expected = alu.execute(VxmSpecialAluOpcode::Log, inputs[index]);
            assert(std::isnan(expected) ? std::isnan(result->value) : result->value == expected);
        }
    }
    assert(pipeline.empty());
}

void verify_placement_and_codec()
{
    for (const auto depth : {VxmChainDepth::Two, VxmChainDepth::Four, VxmChainDepth::Eight}) {
        for (std::size_t stage = 0; stage < 16; ++stage) {
            VxmLaneAluInstruction instruction{VxmSpecialAluOpcode::Log,
                VxmLaneOperand::Previous()};
            bool accepted = false;
            try {
                const auto packet = VxmCompactInstructionCodec::encode(stage, depth, instruction);
                const auto decoded = VxmCompactInstructionCodec::decode(stage, packet);
                assert(decoded.chain_depth == depth);
                assert(std::get<VxmSpecialAluOpcode>(decoded.instruction.operation)
                    == VxmSpecialAluOpcode::Log);
                accepted = true;
            } catch (const std::invalid_argument&) {}
            assert(accepted == (stage % 4 == 3));
        }
    }
}

void verify_softplus()
{
    VxmSuperlane superlane;
    configure_log(superlane.special_alu());
    constexpr float ln2 = 0.6931471805599453f;
    superlane.configure_special_lut(VxmSpecialAluOpcode::Exp,
        {-ln2 / 2.0f, ln2 / 128.0f},
        make_table(-ln2 / 2.0f, ln2 / 128.0f, [](float x) { return std::exp(x); }));
    superlane.set_chain_depth(VxmChainDepth::Four);
    constexpr std::size_t batches = 3;
    // Both logical four-stage groups are mirrored onto stages 8..15.
    for (std::size_t head : {0u, 4u}) {
        std::array<VxmLaneAluInstruction, 4> instructions{{
            {VxmAluOpcode::Bypass, VxmLaneOperand::StreamFloat16()},
            {VxmSpecialAluOpcode::Exp, VxmLaneOperand::Previous()},
            {VxmAluOpcode::Add, VxmLaneOperand::Previous(), VxmLaneOperand::Imm(1.0f)},
            {VxmSpecialAluOpcode::Log, VxmLaneOperand::Previous()}}};
        for (std::size_t offset = 0; offset < 4; ++offset) {
            auto& instruction = instructions[offset];
            instruction.repeat_count = batches;
            if (offset == 3) {
                instruction.output_stream = VxmLane::fixed_output_stream_for_block((head + 3) / 2);
            }
            superlane.enqueue_compact_instruction(head + offset,
                VxmCompactInstructionCodec::encode(head + offset, VxmChainDepth::Four, instruction));
        }
    }
    superlane.tick(); // decode before data arrives
    std::array<std::size_t, 4> output_counts{};
    for (std::size_t cycle = 0; cycle < 24; ++cycle) {
        if (cycle < batches) {
            VxmSuperlane::StreamMatrix streams{};
            for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
                for (std::size_t group = 0; group < 4; ++group) {
                    const auto x = -2.0f + static_cast<float>(lane + group + cycle) * 0.25f;
                    const auto bytes = VxmLane::pack_float16(x);
                    const auto base = VxmLane::fixed_input_group_for_stage(group * 4, false) * 2;
                    streams[lane][base] = bytes[0];
                    streams[lane][base + 1] = bytes[1];
                }
            }
            superlane.set_stream_inputs(streams);
        }
        superlane.tick();
        for (const auto& output : superlane.outputs()) {
            const auto group = (output.stream - 2) / 4;
            assert(group < 4 && output.stream == 2 + group * 4);
            const auto batch = output_counts[group]++;
            assert(batch < batches);
            for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
                const auto x = -2.0f + static_cast<float>(lane + group + batch) * 0.25f;
                const auto value = VxmLane::unpack_float16(
                    {output.byte_values[lane][0], output.byte_values[lane][1]});
                assert(std::fabs(value - std::log1p(std::exp(x))) < 0.004f);
            }
        }
    }
    for (const auto count : output_counts) assert(count == batches);
    assert(superlane.idle());
}
} // namespace

int main()
{
    verify_numerics();
    verify_pipeline();
    verify_placement_and_codec();
    verify_softplus();
    vxm_hardware_test::write_pass_result("log_test_results.txt", "log_test");
}
