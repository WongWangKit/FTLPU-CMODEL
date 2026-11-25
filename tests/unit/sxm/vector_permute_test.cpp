#include "../../support/sxm_macro_fixture.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/sxm/slice.hpp"

#include <algorithm>
#include <iostream>
#include <random>

using namespace ftlpu;

namespace {
constexpr auto rows = hw::kTileRows;
constexpr auto lanes = hw::kLanesPerTile;
constexpr auto width = rows * lanes;

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

StreamCell input_cell(std::size_t vector, std::size_t lane)
{
    const auto id = vector * width + lane;
    return StreamCell::Valid(static_cast<std::uint8_t>(3 + id * 7), lane % 5 == 0, id + 100);
}

void run(const SxmInstruction::PermuteMap& map, StreamId src, StreamId dst, std::size_t interval)
{
    constexpr std::size_t count = 5;
    const auto instruction = SxmInstruction::Permute({{src.packed()}}, {{dst.packed()}}, map);
    auto second = instruction;
    std::rotate(second.permute_map.begin(), second.permute_map.begin()+1, second.permute_map.end());
    test::SxmMacroFixture program;
    program.add(0, 3, interval, instruction);
    program.add(3*interval, count-3, interval, second);
    InstructionControlUnit::SxmPermuteIcu queue;
    program.load(queue, false);
    while (!queue.raw_macro_ready()) queue.prefetch_raw_macro();
    const auto ports = SxmStreamPortMap::BetweenColumns(0, 1, 1, 0);
    SxmSlice slice(ports, SxmTransposeArchitecture::WideStreaming);
    StreamRegisterFabric fabric(2);
    auto inject = [&](std::size_t t) {
        const auto frame = t / interval, sl = t % interval;
        if (frame >= count || sl >= rows) return;
        StreamTileSegment segment{};
        for (std::size_t lane = 0; lane < lanes; ++lane)
            segment[lane] = input_cell(frame, sl * lanes + lane);
        StreamOutputPort(fabric, ports.input_column(src.direction()), src.direction(), "source")
            .write_segment(sl, src.index(), segment);
    };
    fabric.begin_cycle();
    inject(0);
    fabric.commit_cycle();
    std::size_t checked = 0, issues = 0;
    for (std::size_t t = 0; t <= (count-1)*interval + 3*rows; ++t) {
        fabric.begin_cycle();
        if (auto issued = queue.tick()) {
            const auto& expected = issues < 3 ? instruction : second;
            require(t == issues * interval && issued->is_vector_permute() &&
                    issued->permute_map == expected.permute_map, "incorrect raw macro issue");
            require(slice.can_issue(*issued), "vector unexpectedly blocked");
            slice.issue(*issued);
            require(!slice.can_issue(*issued), "accepted two inputs in one cycle");
            ++issues;
        }
        slice.evaluate(fabric);
        inject(t+1);
        fabric.commit_cycle();
        for (std::size_t sl = 0; sl < rows; ++sl) {
            const auto first = 2*rows + sl;
            const bool active = t >= first && (t-first)%interval == 0 && (t-first)/interval < count;
            for (std::size_t l = 0; l < lanes; ++l) {
                const auto cell = fabric.cell(ports.output_column(dst.direction()), sl, l, dst);
                require(cell.valid == active, "unexpected output cycle or skew");
                if (!active) continue;
                const auto frame = (t-first)/interval;
                const auto& mapping = frame < 3 ? map : second.permute_map;
                const auto expected = input_cell(frame, mapping[sl*lanes+l]);
                require(cell.data == expected.data && cell.last == expected.last &&
                        cell.vector_tag == expected.vector_tag, "wrong source lane or metadata");
                ++checked;
            }
        }
    }
    require(issues == count && checked == count * width && slice.idle(), "incomplete drain");
}
}

int main()
{
    auto map = Permute320::identity_map();
    SxmSlice local(SxmStreamPortMap::SameDirection(0, 1));
    for (std::size_t row = 0; row < lanes; ++row) {
        for (std::size_t plane = 0; plane < 2; ++plane) {
            auto bank = SxmInstruction::PermuteFromBank(row, plane, {3}, map);
            require(!local.can_issue(bank), "unfilled transpose bank admitted");
            auto raw = isa::decode_sxm_instruction(isa::encode_sxm_instruction(bank));
            require(raw.reads_transpose_bank() && raw.input_row == 2 * row + plane,
                    "bank byte row lost in native SXM encoding");
            test::SxmMacroFixture program;
            program.add(0, 1, 1, bank);
            InstructionControlUnit::SxmPermuteIcu queue;
            program.load(queue, false);
            while (!queue.raw_macro_ready()) queue.prefetch_raw_macro();
            auto issued = queue.tick();
            require(issued && issued->reads_transpose_bank() && issued->input_row == raw.input_row
                    && issued->permute_map == map, "bank selector lost in raw macro");
        }
    }
    run(map, StreamId::East(0), StreamId::West(1), rows);
    std::reverse(map.begin(), map.end());
    run(map, StreamId::West(hw::kStreamsPerDirection-1), StreamId::East(2), rows);
    std::mt19937 rng(2026);
    for (int i = 0; i < 8; ++i) {
        std::shuffle(map.begin(), map.end(), rng);
        run(map, StreamId::East(1), StreamId::West(2), rows + i%3);
    }
    SxmSlice slice(SxmStreamPortMap::SameDirection(0, 1));
    auto ins = SxmInstruction::Permute({{0}}, {{1}}, map);
    slice.issue(ins);
    StreamRegisterFabric empty(2);
    empty.begin_cycle();
    bool missing = false;
    try { slice.evaluate(empty); } catch (const std::logic_error&) { missing = true; }
    require(missing, "missing scheduled input accepted");
    slice.reset();
    require(slice.idle(), "reset did not release vector stages");
    ins.permute_map[0] = ins.permute_map[1];
    bool rejected = false;
    try { isa::write_sxm_macro_native(ins, false, [](auto, auto) {}); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "duplicate lane accepted");
    std::cout << "PASS: 50 raw-macro vectors, " << 50*width
              << " lane values/tags, issue interval " << rows << ", output at issue+"
              << 2*rows << "+SL; invalid map, missing input and reset checked\n";
}
