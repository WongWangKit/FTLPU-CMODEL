#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

namespace {

constexpr std::size_t kColumns = ftlpu::hw::kMxmColumns;

std::size_t index(std::size_t row, std::size_t column)
{
    return row * kColumns + column;
}

std::int32_t input_value(std::size_t row, std::size_t column)
{
    return static_cast<std::int32_t>((row * 3 + column * 5) % 257) - 128;
}

ftlpu::VxmSlice::StreamMatrix stream_matrix_for_column(std::size_t tile, std::size_t column)
{
    auto streams = ftlpu::VxmSlice::StreamMatrix {};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        const auto row = tile * ftlpu::hw::kLanesPerTile + lane;
        const auto bytes = ftlpu::VxmLane::pack_int32(input_value(row, column));
        for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
            streams[lane][byte] = bytes[byte];
        }
    }
    return streams;
}

} // namespace

int main()
{
    auto slice = std::make_unique<ftlpu::VxmSlice>();
    auto icu = ftlpu::InstructionControlUnit {};
    std::vector<std::int8_t> output_matrix(ftlpu::VxmSlice::kRows * kColumns);
    std::vector<bool> output_valid(ftlpu::VxmSlice::kRows * kColumns, false);

    for (std::size_t token = 0; token < kColumns; ++token) {
        icu.enqueue_vxm(0, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(0),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Int8,
            0,
        });
    }

    std::ostringstream log;
    const auto total_cycles = kColumns;
    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        for (std::size_t tile = 0; tile < ftlpu::VxmSlice::kTileCount; ++tile) {
            slice->set_stream_inputs(tile, stream_matrix_for_column(tile, cycle));
        }

        icu.dispatch_vxm(*slice, &log);
        slice->tick(&log);

        for (std::size_t tile = 0; tile < ftlpu::VxmSlice::kTileCount; ++tile) {
            const auto& output = slice->output_at(tile);
            if (!output.has_value()) {
                continue;
            }

            const auto column = cycle;
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto row = tile * ftlpu::hw::kLanesPerTile + lane;
                output_matrix[index(row, column)] = output->values[lane];
                output_valid[index(row, column)] = true;
            }
        }
    }

    for (std::size_t row = 0; row < ftlpu::VxmSlice::kRows; ++row) {
        for (std::size_t column = 0; column < kColumns; ++column) {
            assert(output_valid[index(row, column)]);
            assert(output_matrix[index(row, column)]
                == ftlpu::VxmAlu::cast_scalar_to_int8(static_cast<float>(input_value(row, column))));
        }
    }

    const auto text = log.str();
    assert(text.find("ICU -> VXM alu0") != std::string::npos);
    assert(text.find("vxm_slice cycle 0") != std::string::npos);
    assert(text.find("tile 0 alu0 cast") != std::string::npos);
    assert(text.find("tile 19 alu0 cast") != std::string::npos);
    assert(text.find("tile 0 output") != std::string::npos);
    assert(text.find("tile 19 output") != std::string::npos);

    return 0;
}
