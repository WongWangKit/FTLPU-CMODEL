#include "ftlpu/sxm/slice.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

ftlpu::SxmInstruction::StreamList streams(std::size_t base, std::size_t count)
{
    ftlpu::SxmInstruction::StreamList result;
    for (std::size_t i = 0; i < count; ++i) result.push_back({base + i});
    return result;
}

struct Fixture {
    ftlpu::StreamRegisterFabric fabric{2};
    ftlpu::SxmSlice sxm{ftlpu::SxmStreamPortMap::SameDirection(0, 1)};

    ftlpu::SxmInstruction transpose(bool row_input = false, std::size_t row = 0)
    {
        auto instruction = ftlpu::SxmInstruction::Transpose(
            streams(0, row_input ? 2 : 16), streams(16, 16));
        if (row_input) instruction.input_row = row;
        return instruction;
    }

    ftlpu::SxmInstruction permute()
    {
        return ftlpu::SxmInstruction::Permute(streams(16, 16),
            streams(ftlpu::StreamId::West(0).packed(), 16),
            ftlpu::Permute320::identity_map());
    }

    static std::uint8_t value(std::size_t tag, std::size_t row,
        std::size_t lane, std::size_t plane)
    {
        return static_cast<std::uint8_t>(tag * 43 + row * 9 + lane + plane * 101);
    }

    void input(std::size_t tag, bool row_input = false, std::size_t row = 0)
    {
        // Isolate source tile 0. Other northbound T positions see no input.
        for (std::size_t q = 0; q < (row_input ? 1u : 8u); ++q)
            for (std::size_t b = 0; b < 2; ++b)
                for (std::size_t l = 0; l < 8; ++l)
                    fabric.initialize_cell(0, 0, l, ftlpu::StreamId::East(q * 2 + b),
                        ftlpu::StreamCell::Valid(value(tag, row_input ? row : q, l, b), l == 7));
    }

    void tick()
    {
        fabric.begin_cycle();
        sxm.evaluate(fabric);
        fabric.commit_cycle();
    }

    void output(std::size_t tag, std::size_t selected = 8)
    {
        for (std::size_t q = 0; q < 8; ++q)
            for (std::size_t b = 0; b < 2; ++b)
                for (std::size_t l = 0; l < 8; ++l) {
                    const auto cell = fabric.cell(1, 0, l, ftlpu::StreamId::West(q * 2 + b));
                    require(cell.valid == (selected == 8 || selected == q), "incorrect P row valid");
                    if (cell.valid) require(cell.data == value(tag, l, q, b), "P data mismatch");
                }
    }

    void no_output()
    {
        for (std::size_t q = 0; q < 16; ++q)
            require(!fabric.segment_valid(1, 0, ftlpu::StreamId::West(q)),
                "same-cycle T-to-P bypass is forbidden");
    }
};

void partial_capture_is_busy()
{
    Fixture f;
    f.input(0, true, 0);
    f.sxm.issue(f.transpose(true, 0));
    f.tick();
    for (std::size_t n = 0; n < ftlpu::hw::kTileRows; ++n) f.tick();
    // VModel compat busy includes any occupied row, not only a complete block.
    require(!f.sxm.idle(), "partial transpose was incorrectly reported drained");
    f.sxm.reset();
    require(f.sxm.idle(), "reset did not clear partial ownership");
}

void release_then_capture()
{
    Fixture f;
    f.input(1);
    f.sxm.issue(f.transpose());
    f.sxm.issue(f.permute());
    f.tick();
    f.no_output();

    f.input(2);
    f.sxm.issue(f.transpose());
    f.sxm.issue(f.permute());
    f.tick();
    f.output(1); // P reads the old slot before T replaces it at the same edge.
    f.sxm.issue(f.permute());
    f.tick();
    f.output(2);
    for (std::size_t n = 0; n < ftlpu::hw::kTileRows; ++n) f.tick();
    require(f.sxm.idle(), "released source slot did not drain");
}

void missing_p_is_not_buffered()
{
    Fixture f;
    f.input(1);
    f.sxm.issue(f.transpose());
    f.tick();
    f.input(2);
    f.sxm.issue(f.transpose());
    bool full = false;
    try { f.tick(); }
    catch (const std::logic_error& e) {
        full = std::string(e.what()).find("buffer is full at tile 0") != std::string::npos;
    }
    require(full, "a missing P must not silently overwrite or queue the old block");
}

void row_capture_and_drain()
{
    Fixture f;
    for (std::size_t r = 0; r < 8; ++r) {
        f.input(3, true, r);
        f.sxm.issue(f.transpose(true, r));
        f.sxm.issue(f.permute());
        f.tick();
        f.no_output();
    }
    for (std::size_t r = 0; r < 8; ++r) {
        auto p = f.permute();
        p.output_row = r;
        f.sxm.issue(p);
        f.tick();
        f.output(3, r);
        if (r != 7) require(!f.sxm.idle(), "row P released the source before row 7");
    }
    require(f.sxm.idle(), "last P row did not release the source");
}

} // namespace

int main() try
{
    partial_capture_is_busy();
    release_then_capture();
    missing_p_is_not_buffered();
    row_capture_and_drain();
    std::cout << "SXM local-block timing passed: partial busy, no bypass, same-edge reuse, missing P, row drain\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
