#pragma once

#include "ftlpu/sxm/matrix.hpp"
#include "ftlpu/mem/mem_array.hpp"

#include <algorithm>
#include <vector>

namespace ftlpu {

// Explicit logical-bank binding for the existing CModel geometry (4 slices x
// 2 banks/group). No change to MEM, its Main read latency, or its instruction ABI.
struct SxmMatrixMemLayout {
    static_assert(hw::kMemSlicesPerGroup * hw::kMemBanksPerSlice == 8);
    static_assert(hw::kMemGroups >= 4 && hw::kStreamsPerDirection >= 32);
    static constexpr std::size_t inject_column = hw::kMemGroups + 1;
    static constexpr std::size_t read_column = hw::kMemGroups - 4;
    static std::size_t group(std::size_t byte_plane) { return byte_plane / 8; }
    static std::size_t slice(std::size_t j)
    {
        return (hw::kMemGroups - 1 - group(j)) * hw::kMemSlicesPerGroup
            + (j % 8) / hw::kMemBanksPerSlice;
    }
    static std::size_t bank(std::size_t j) { return j % hw::kMemBanksPerSlice; }
    static std::size_t input_column(std::size_t j) { return hw::kMemGroups - group(j); }
    static std::size_t write_hops(std::size_t j) { return 2 + group(j); }
    static std::size_t read_hops(std::size_t j, std::size_t consumer_column = read_column)
    {
        if (consumer_column > read_column) throw std::invalid_argument("SXM Read consumer lies before a bank output");
        return hw::kMemGroups - group(j) - consumer_column;
    }
    static void validate(const MemArrayModel& mem, const StreamRegisterFabric& fabric)
    {
        if (mem.read_timing() != MemArrayModel::ReadTiming::Main)
            throw std::invalid_argument("SXM matrix adapter requires unchanged MEM Main timing");
        if (fabric.column_count() <= inject_column)
            throw std::invalid_argument("SXM matrix fabric lacks injection column");
        for (std::size_t j = 0; j < 32; ++j) {
            if (mem.ports().input_column(slice(j), StreamDirection::West) != input_column(j) ||
                mem.ports().output_column(slice(j), StreamDirection::West) + 1 != input_column(j))
                throw std::invalid_argument("SXM matrix adapter needs BetweenBoundaries MEM map");
        }
    }
};

// Exclusive module-level harness: caller reserves the bound MEM banks and SR
// route before start(). tick() owns one MEM/fabric cycle, not an ICU macro.
// The only persistent payload is the triangular core. P goes straight to SR;
// there is no output-block FIFO, reverse queue, or software pending-P queue.
class SxmMatrixMemReference {
public:
    static constexpr std::size_t T = hw::kTileRows;
    using Layout = SxmMatrixMemLayout;
    struct Statistics {
        std::size_t commits{}, tap_issues{}, blocks{};
        std::uint64_t first_p{}, last_p{}, last_commit{}, done_cycle{};
    };
    SxmMatrixMemReference(MemArrayModel& mem, StreamRegisterFabric& fabric,
                         std::size_t reserved_depth)
        : mem_(mem), fabric_(fabric), depth_(reserved_depth)
    {
        Layout::validate(mem_, fabric_);
        if (!depth_ || depth_ > 32768 || depth_ > hw::kSramDepthRows)
            throw std::invalid_argument("SXM reserved depth");
    }
    void start(const SxmMatrixStart& job, bool resources_reserved)
    {
        // Admission checks have no side effects. The owner must reserve the
        // actual active SRAM depth as well as banks/streams for the whole job.
        job.validate();
        Layout::validate(mem_, fabric_);
        if (!resources_reserved || active_ || fault() || !mem_.idle() || !fabric_.empty() ||
            fabric_.cycle_open() || mem_.cycle() != fabric_.cycle())
            throw std::logic_error("SXM start requires idle, exclusively reserved resources");
        if (std::uint64_t(job.mem_base) + std::uint64_t(job.matrices) * T > depth_)
            throw std::invalid_argument("SXM matrix address exceeds reserved SRAM");
        core_.synchronize_idle(mem_.cycle());
        core_.start(job); job_ = job; launch_ = mem_.cycle();
        for (std::size_t a = 0; a < T; ++a) inverse_[job_.tile_map[a]] = a;
        stats_ = {}; active_ = true; done_ = false;
    }
    void tick(const SxmMatrixCore::Inputs& inputs = {})
    {
        if (fault()) throw std::logic_error("SXM matrix adapter fault; reset required");
        if (fabric_.cycle_open() || mem_.cycle() != fabric_.cycle() || core_.cycle() != mem_.cycle())
            throw std::logic_error("SXM matrix cycle ownership mismatch");
        const auto c = mem_.cycle();
        done_ = false; // RTL done_o is a one-cycle pulse; statistics retain the event.
        try {
            fabric_.begin_cycle();
            const auto outputs = core_.tick(inputs);
            if (active_) {
                const auto age = c - launch_;
                // Exact P calendar; delayed/missing CAP is diagnosed here, not
                // hidden by extra buffering or re-timed MEM instructions.
                for (std::size_t d = 0; d < T; ++d) {
                    const auto n = target(age, 9, d);
                    if (bool(outputs[d]) != bool(n) || (n &&
                        (outputs[d]->matrix != *n / T || outputs[d]->source != *n % T)))
                        fail("SXM missing/unexpected P on fixed WriteTap calendar");
                }
                for (std::size_t d = 0; d < T; ++d) if (outputs[d]) {
                    if (!stats_.blocks) stats_.first_p = c;
                    stats_.last_p = c; ++stats_.blocks;
                    for (std::size_t j = 0; j < job_.bytes() * 8; ++j) {
                        StreamPayloadTileSegment bytes{};
                        for (std::size_t l = 0; l < 8; ++l)
                            bytes[l] = static_cast<std::uint8_t>(outputs[d]->words[j/job_.bytes()][l] >> (8*(j%job_.bytes())));
                        fabric_.stage_payload_segment(Layout::inject_column, d, StreamId::West(j), bytes,
                            outputs[d]->matrix * T + outputs[d]->source, "SXM matrix P");
                    }
                }
                for (std::size_t j = 0; j < job_.bytes() * 8; ++j)
                    for (std::size_t d = 0; d < T; ++d)
                        if (auto n = target(age, 9 + Layout::write_hops(j) - d, d)) {
                            mem_.enqueue_instruction(Layout::slice(j), Layout::bank(j),
                                MemInstruction::WriteTap(job_.mem_base + *n, StreamId::West(j)));
                            ++stats_.tap_issues;
                        }
            }
            mem_.evaluate(fabric_); // keep real transfer traces for commit validation
            if (active_) check_commits(c);
            fabric_.stage_linear_links();
            fabric_.commit_cycle();
            if (active_) {
                const auto last_p = 9 + 8 * (std::uint64_t(job_.matrices) * T + T - 2);
                // writer.active drops at last_P + max(H) + T. Completion sees
                // that OLD registered busy one edge later, as in the RTL monitor.
                const auto horizon = last_p + (job_.bytes() + 1) + T + 1;
                if (c - launch_ >= horizon) {
                    if (core_.busy() || !mem_.idle() || !fabric_.empty() ||
                        stats_.commits != job_.matrices * 8 * T * T * job_.bytes())
                        fail("SXM incomplete commit/drain at writer completion");
                    active_ = false; done_ = true; stats_.done_cycle = c;
                }
            }
        } catch (...) { fault_ = true; throw; }
    }
    // Explicit coordinated abort; SRAM contents are intentionally retained.
    void reset_execution_state()
    {
        mem_.reset_execution_state(); fabric_.reset(); core_.reset();
        active_ = done_ = fault_ = false; stats_ = {};
    }
    bool busy() const { return active_; }
    bool done() const { return done_; }
    bool fault() const { return fault_ || core_.fault(); }
    const Statistics& statistics() const { return stats_; }
    const SxmMatrixCore& core() const { return core_; }
private:
    std::optional<std::size_t> target(std::uint64_t age, std::size_t first, std::size_t d) const
    {
        if (age < first || (age - first) % 8) return {};
        const auto k = (age - first) / 8;
        if (k < inverse_[d]) return {};
        const auto n = k - inverse_[d];
        if (n >= job_.matrices * T) return {};
        return static_cast<std::size_t>(n);
    }
    void check_commits(std::uint64_t c)
    {
        std::array<std::array<bool, T>, 32> seen{};
        const auto age = c - launch_;
        for (const auto& x : mem_.executed_transfers()) {
            if (x.kind != MemArrayModel::MemTransfer::Kind::StoreStreamToSram ||
                x.stream.direction() != StreamDirection::West || x.stream.index() >= job_.bytes() * 8)
                fail("SXM unexpected transfer on reserved MEM path");
            const auto j = x.stream.index(); const auto d = x.tile;
            const auto n = target(age, 9 + Layout::write_hops(j), d);
            if (!n || seen[j][d] || x.mem_slice != Layout::slice(j) || x.bank != Layout::bank(j) ||
                x.sr_column != Layout::input_column(j) || x.address != job_.mem_base + *n)
                fail("SXM wrong/duplicate WriteTap commit");
            seen[j][d] = true; ++stats_.commits; stats_.last_commit = c;
            // Native WriteTap is non-consuming. Consume at its actual final use
            // only after MEM evaluated; no copies survive to a second group.
            fabric_.consume_segment(x.sr_column, d, x.stream, "SXM WriteTap final use");
        }
        for (std::size_t j = 0; j < job_.bytes() * 8; ++j)
            for (std::size_t d = 0; d < T; ++d)
                if (bool(target(age, 9 + Layout::write_hops(j), d)) != seen[j][d])
                    fail("SXM scheduled WriteTap did not commit");
    }
    [[noreturn]] void fail(const char* msg) { fault_ = true; throw std::runtime_error(msg); }
    MemArrayModel& mem_;
    StreamRegisterFabric& fabric_;
    std::size_t depth_;
    SxmMatrixCore core_{};
    SxmMatrixStart job_{};
    std::array<std::size_t, T> inverse_{};
    Statistics stats_{};
    std::uint64_t launch_{};
    bool active_{}, done_{}, fault_{};
};

// Metadata-only normal MEM Read plan for one transposed row / operand.
// FP32 reuses one two-stream pair at E and E+1; it is not a new MEM opcode.
struct SxmMatrixRowRead {
    struct Command { std::uint64_t cycle; std::size_t slice, bank; MemInstruction instruction; };
    static std::vector<Command> plan(const SxmMatrixStart& job, std::size_t matrix,
        std::size_t row, std::size_t stream_pair_base, std::uint64_t arrival_tile0,
        std::size_t consumer_column = SxmMatrixMemLayout::read_column)
    {
        job.validate();
        if (matrix >= job.matrices || row >= 8 * hw::kTileRows || stream_pair_base % 2 ||
            stream_pair_base + 1 >= hw::kStreamsPerDirection ||
            arrival_tile0 < SxmMatrixMemLayout::read_hops(0, consumer_column) ||
            std::uint64_t(job.mem_base) + std::uint64_t(job.matrices) * hw::kTileRows > 32768)
            throw std::invalid_argument("SXM matrix row Read range/calendar");
        std::vector<Command> commands;
        for (std::size_t b = 0; b < job.bytes(); ++b) {
            const auto j = (row % 8) * job.bytes() + b;
            commands.push_back({arrival_tile0 + b/2 - SxmMatrixMemLayout::read_hops(j, consumer_column),
                SxmMatrixMemLayout::slice(j), SxmMatrixMemLayout::bank(j),
                MemInstruction::Read(job.mem_base + matrix * hw::kTileRows + row/8,
                                     StreamId::West(stream_pair_base + b%2))});
        }
        return commands;
    }
};

} // namespace ftlpu
