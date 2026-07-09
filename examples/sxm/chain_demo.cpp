#include "ftlpu/sxm/slice.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Word = std::int32_t;
using Sxm = ftlpu::SxmUnitGroup<Word>;
using TileVector = ftlpu::SxmSlice::TileVector<Word>;
using Matrix16 = ftlpu::SxmSlice::Matrix16<Word>;

ftlpu::SxmInstruction::StreamList stream_range(std::size_t first, std::size_t count)
{
    ftlpu::SxmInstruction::StreamList streams;
    streams.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        streams.push_back(ftlpu::SxmStreamId {first + index});
    }
    return streams;
}

TileVector input_vector(std::size_t stream)
{
    TileVector vector{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        vector[lane] = static_cast<Word>(stream * 100 + lane);
    }
    return vector;
}

Matrix16 golden_transpose(const Matrix16& input)
{
    Matrix16 output{};
    for (std::size_t out_stream = 0; out_stream < ftlpu::hw::kLanesPerTile; ++out_stream) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            output[out_stream][lane] = input[lane][out_stream];
        }
    }
    return output;
}

TileVector golden_distribute(
    const TileVector& input,
    const ftlpu::Distribute16::Map& map,
    Word zero = 0)
{
    TileVector output{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        const auto source = map[lane];
        output[lane] = source == ftlpu::Distribute16::kZeroFill ? zero : input[source];
    }
    return output;
}

TileVector read_stream_data(const Sxm& sxm, std::size_t stream)
{
    const ftlpu::SxmStreamId id {stream};
    if (!sxm.stream_available(id)) {
        throw std::logic_error("SXM demo expected an available output stream");
    }

    TileVector data{};
    const auto& slot = sxm.stream(id);
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        data[lane] = slot[lane]->data;
    }
    return data;
}

bool equal_vector(const TileVector& lhs, const TileVector& rhs)
{
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        if (lhs[lane] != rhs[lane]) {
            return false;
        }
    }
    return true;
}

void print_vector(std::ostream& os, const TileVector& vector)
{
    os << '[';
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        if (lane != 0) {
            os << ' ';
        }
        os << std::setw(4) << vector[lane];
    }
    os << ']';
}

ftlpu::Distribute16::Map rotate_left_map(std::size_t distance)
{
    ftlpu::Distribute16::Map map{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        map[lane] = (lane + distance) % ftlpu::hw::kLanesPerTile;
    }
    return map;
}

ftlpu::Distribute16::Map reverse_map()
{
    ftlpu::Distribute16::Map map{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        map[lane] = ftlpu::hw::kLanesPerTile - 1 - lane;
    }
    return map;
}

ftlpu::Distribute16::Map stride_map(std::size_t stride)
{
    ftlpu::Distribute16::Map map{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        map[lane] = (lane * stride) % ftlpu::hw::kLanesPerTile;
    }
    return map;
}

bool check_equal(
    std::ostream& log,
    const char* label,
    const TileVector& actual,
    const TileVector& expected)
{
    log << label << '\n';
    log << "  actual   ";
    print_vector(log, actual);
    log << '\n';
    log << "  expected ";
    print_vector(log, expected);
    log << '\n';

    const auto ok = equal_vector(actual, expected);
    log << "  result   " << (ok ? "PASS" : "FAIL") << "\n\n";
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string log_path = argc > 1 ? argv[1] : "sxm_chain_demo.log";
    std::ofstream log(log_path);
    if (!log) {
        std::cerr << "failed to open log file: " << log_path << '\n';
        return 1;
    }

    Sxm sxm;
    Matrix16 input{};
    for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
        input[stream] = input_vector(stream);
        sxm.set_stream_input(ftlpu::SxmStreamId {stream}, input[stream]);
    }

    log << "SXM chain demo\n";
    log << "cycle 0: issue Transpose streams 0..15 -> 16..31\n";
    sxm.issue(ftlpu::SxmInstruction::Transpose(stream_range(0, 16), stream_range(16, 16)));
    sxm.tick();

    const auto transposed = golden_transpose(input);
    bool ok = true;
    ok = check_equal(log, "transpose output stream 16", read_stream_data(sxm, 16), transposed[0]) && ok;
    ok = check_equal(log, "transpose output stream 23", read_stream_data(sxm, 23), transposed[7]) && ok;

    const auto rotate = rotate_left_map(3);
    const auto reverse = reverse_map();
    const auto stride = stride_map(5);

    log << "pipeline chain: one SXM stage per cycle\n";
    log << "cycle 1: stream 16 -> 40 rotate-left-by-3\n";
    sxm.issue(ftlpu::SxmInstruction::Distribute(ftlpu::SxmStreamId {16}, ftlpu::SxmStreamId {40}, rotate));
    sxm.tick();
    const auto golden_after_rotate = golden_distribute(transposed[0], rotate);
    ok = check_equal(log, "stage 0 output stream 40", read_stream_data(sxm, 40), golden_after_rotate) && ok;

    log << "cycle 2: stream 40 -> 41 reverse\n";
    sxm.issue(ftlpu::SxmInstruction::Distribute(ftlpu::SxmStreamId {40}, ftlpu::SxmStreamId {41}, reverse));
    sxm.tick();
    const auto golden_after_reverse = golden_distribute(golden_after_rotate, reverse);
    ok = check_equal(log, "stage 1 output stream 41", read_stream_data(sxm, 41), golden_after_reverse) && ok;

    log << "cycle 3: stream 41 -> 42 stride-5\n";
    sxm.issue(ftlpu::SxmInstruction::Distribute(ftlpu::SxmStreamId {41}, ftlpu::SxmStreamId {42}, stride));
    sxm.tick();
    const auto golden_after_stride = golden_distribute(golden_after_reverse, stride);

    ok = check_equal(log, "final chained output stream 42", read_stream_data(sxm, 42), golden_after_stride) && ok;
    log << "intermediate stream 40 occupied after pipeline chain: "
        << (sxm.stream_occupied(ftlpu::SxmStreamId {40}) ? "yes" : "no") << '\n';
    log << "intermediate stream 41 occupied after pipeline chain: "
        << (sxm.stream_occupied(ftlpu::SxmStreamId {41}) ? "yes" : "no") << '\n';

    if (!ok) {
        std::cerr << "SXM chain demo failed golden comparison; see " << log_path << '\n';
        return 1;
    }

    std::cout << "SXM chain demo passed; wrote log: " << log_path << '\n';
    return 0;
}
