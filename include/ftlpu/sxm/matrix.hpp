#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ftlpu {

enum class SxmMatrixDtype { BFloat16, Float32 };

// Typed module command, NOT an ICU packet/opcode assignment. Only matrix
// ROW -> BLOCK8 is exposed; legacy SxmSlice remains an independent reference.
struct SxmMatrixStart {
    static constexpr std::size_t T = hw::kTileRows;
    std::uint32_t matrices{1};
    SxmMatrixDtype dtype{SxmMatrixDtype::BFloat16};
    std::uint32_t mem_base{0};
    std::array<std::size_t, T> tile_map{}; // input block-row -> destination
    std::array<std::array<std::size_t, 8>, T> lane_map{}; // per destination

    SxmMatrixStart()
    {
        for (std::size_t t = 0; t < T; ++t) {
            tile_map[t] = t;
            for (std::size_t l = 0; l < 8; ++l) lane_map[t][l] = l;
        }
    }
    std::size_t bytes() const { return dtype == SxmMatrixDtype::Float32 ? 4 : 2; }
    void validate() const
    {
        if (!matrices || matrices > 65535 ||
            (dtype != SxmMatrixDtype::BFloat16 && dtype != SxmMatrixDtype::Float32))
            throw std::invalid_argument("SXM matrix count/dtype outside RTL contract");
        std::array<bool, T> seen{};
        for (std::size_t t = 0; t < T; ++t) {
            if (tile_map[t] >= T || seen[tile_map[t]])
                throw std::invalid_argument("SXM tile_map must be a bijection");
            seen[tile_map[t]] = true;
            for (auto l : lane_map[t])
                if (l >= 8) throw std::invalid_argument("SXM lane_map outside 0..7");
        }
    }
};

class SxmMatrixCore {
public:
    static constexpr std::size_t T = hw::kTileRows;
    static constexpr std::size_t kSlots = T * (T + 1) / 2;
    static constexpr std::size_t kPayloadBytes = kSlots * 8 * 8 * 4;
    static_assert(T == 2 || T == 4 || T == 8);
    static_assert(hw::kLanesPerTile == 8);
    struct Row {
        std::array<std::uint32_t, 8> words{};
        std::uint8_t lane_valid{0xff};
    };
    struct Block {
        std::array<std::array<std::uint32_t, 8>, 8> words{};
        std::size_t matrix{}, source{};
    };
    using Inputs = std::array<std::optional<Row>, T>;
    using Outputs = std::array<std::optional<Block>, T>;
    using Ready = std::array<bool, T>;
    static Ready all_ready() { Ready r{}; r.fill(true); return r; }

    void reset(std::uint64_t cycle = 0) { *this = SxmMatrixCore{}; cycle_ = cycle; }
    void synchronize_idle(std::uint64_t cycle)
    {
        if (busy() || fault_ || cycle < cycle_)
            throw std::logic_error("SXM cannot synchronize active/faulted time");
        cycle_ = cycle;
    }
    void start(const SxmMatrixStart& job)
    {
        if (busy() || fault_) fail("SXM start while busy/faulted");
        try { job.validate(); } catch (...) { fault_ = true; throw; }
        job_ = job; // Private snapshot: subsequent caller changes cannot retarget P.
        received_.fill(0); rd_.fill(0); wr_.fill(0); peak_.fill(0);
        sent_ = 0; active_ = true; launch_ = cycle_;
    }
    Outputs tick(const Inputs& inputs = {}, const Ready& ready = all_ready())
    {
        if (fault_) throw std::logic_error("SXM sticky fault; reset required");
        Outputs out{};
        // Observe old payload and validate every due output BEFORE any release
        // or capture. This is the RTL's same-edge read/reuse, not a row-7 bypass.
        for (std::size_t s = 0; s < T; ++s) {
            auto& slot = slots_[base(s) + rd_[s]];
            if (!active_ || !slot.complete || slot.due != timer()) continue;
            if (!ready[slot.dest] || out[slot.dest]) fail("SXM due P blocked/colliding");
            Block b{}; b.matrix = slot.matrix; b.source = s;
            for (std::size_t q = 0; q < 8; ++q)
                for (std::size_t l = 0; l < 8; ++l)
                    b.words[q][l] = slot.words[job_.lane_map[slot.dest][l]][q];
            out[slot.dest] = b;
        }
        // Release old heads. A source's row 0 can now reuse that very slot.
        for (std::size_t s = 0; s < T; ++s) {
            auto& slot = slots_[base(s) + rd_[s]];
            if (active_ && slot.complete && slot.due == timer()) {
                slot.occupied = slot.complete = false;
                rd_[s] = (rd_[s] + 1) % (s + 1); ++sent_;
            }
        }
        for (std::size_t s = 0; s < T; ++s) {
            if (!inputs[s]) continue;
            const auto row = received_[s] % 8;
            auto& slot = slots_[base(s) + wr_[s]];
            if (!active_ || cycle_ == launch_ || inputs[s]->lane_valid != 0xff ||
                received_[s] >= job_.matrices * 8 * T ||
                (row == 0 ? slot.occupied : (!slot.occupied || slot.complete)))
                fail("SXM illegal CAP / source ring full");
            if (row == 0) {
                slot.occupied = true;
                slot.dest = job_.tile_map[(received_[s] / 8) % T];
                slot.matrix = received_[s] / (8 * T);
            }
            for (std::size_t l = 0; l < 8; ++l)
                slot.words[row][l] = inputs[s]->words[l] &
                    (job_.dtype == SxmMatrixDtype::Float32 ? 0xffffffffu : 0xffffu);
            ++received_[s];
            if (row == 7) {
                slot.complete = true;
                slot.due = static_cast<std::uint16_t>(timer() + 1 + 7 * s);
                wr_[s] = (wr_[s] + 1) % (s + 1);
            }
            if (occupancy(s) > peak_[s]) peak_[s] = occupancy(s);
        }
        if (active_ && sent_ == job_.matrices * T * T) active_ = false;
        ++cycle_;
        return out;
    }
    bool busy() const
    {
        if (active_) return true;
        for (const auto& slot : slots_) if (slot.occupied) return true;
        return false;
    }
    bool fault() const { return fault_; }
    std::uint64_t cycle() const { return cycle_; }
    std::size_t sent() const { return sent_; }
    const auto& peaks() const { return peak_; }
    const auto& received() const { return received_; }
    std::size_t occupancy(std::size_t s) const
    {
        if (s >= T) throw std::out_of_range("SXM source");
        std::size_t n = 0;
        for (std::size_t i = 0; i <= s; ++i) n += slots_[base(s) + i].occupied;
        return n;
    }
private:
    struct Slot {
        std::array<std::array<std::uint32_t, 8>, 8> words{};
        std::uint16_t due{};
        std::size_t dest{}, matrix{};
        bool occupied{}, complete{};
    };
    static constexpr std::size_t base(std::size_t s) { return s * (s + 1) / 2; }
    std::uint16_t timer() const { return static_cast<std::uint16_t>(cycle_); }
    [[noreturn]] void fail(const char* msg) { fault_ = true; throw std::runtime_error(msg); }
    std::array<Slot, kSlots> slots_{};
    std::array<std::size_t, T> received_{}, rd_{}, wr_{}, peak_{};
    SxmMatrixStart job_{};
    std::uint64_t cycle_{}, launch_{};
    std::size_t sent_{};
    bool active_{}, fault_{};
};

} // namespace ftlpu
