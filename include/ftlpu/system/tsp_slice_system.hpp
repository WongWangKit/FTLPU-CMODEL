#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>

namespace ftlpu {

class TspSliceSystem {
public:
    static constexpr std::size_t kMxmCount = 2;

    struct LogSinks {
        std::ostream* icu{nullptr};
        std::ostream* mem{nullptr};
        std::ostream* mxm{nullptr};
        std::ostream* vxm{nullptr};
        std::ostream* system{nullptr};
        std::optional<std::size_t> mem_log_tile{};
        std::optional<std::size_t> mxm_log_tile{};
        std::optional<std::size_t> vxm_log_tile{};
    };

    TileArrayModel& mem()
    {
        return mem_;
    }

    const TileArrayModel& mem() const
    {
        return mem_;
    }

    VxmSlice& vxm()
    {
        return vxm_;
    }

    const VxmSlice& vxm() const
    {
        return vxm_;
    }

    MxmArray& mxm()
    {
        return mxms_[0].array();
    }

    const MxmArray& mxm() const
    {
        return mxms_[0].array();
    }

    MxmControlSlice& mxm_control()
    {
        return mxms_[0].control();
    }

    Mxm& mxm_unit(std::size_t index)
    {
        check_mxm_index(index);
        return mxms_[index];
    }

    const Mxm& mxm_unit(std::size_t index) const
    {
        check_mxm_index(index);
        return mxms_[index];
    }

    InstructionControlUnit& icu()
    {
        return icu_;
    }

    const InstructionControlUnit& icu() const
    {
        return icu_;
    }

    void tick(std::ostream& os)
    {
        LogSinks sinks {&os, &os, &os, &os, &os};
        tick(sinks);
    }

    void tick(LogSinks sinks)
    {
        if (sinks.system != nullptr) {
            *sinks.system << "system cycle " << cycle_ << '\n';
        }
        icu_.dispatch(mem_, vxm_, mxms_, sinks.icu);
        tick_mxm_controls(sinks);
        tick_mxm_datapaths(sinks);
        vxm_.prepare_cycle();
        transfer_mem_west_to_vxm(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        transfer_vxm_to_mem_east(sinks);
        if (sinks.mem != nullptr) {
            mem_.tick(*sinks.mem, sinks.mem_log_tile);
        }
        ++cycle_;
    }

    void dispatch_icu_only(std::ostream* os = nullptr)
    {
        icu_.dispatch(mem_, vxm_, mxms_, os);
    }

    void tick_mxm_controls_only(LogSinks sinks)
    {
        tick_mxm_controls(sinks);
    }

    void tick_mxm_datapaths_only(LogSinks sinks)
    {
        tick_mxm_datapaths(sinks);
    }

    void tick_vxm_stream_bridge(LogSinks sinks, std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (log_tile.has_value()) {
            sinks.vxm_log_tile = log_tile;
        }
        vxm_.prepare_cycle();
        transfer_mem_west_to_vxm(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        transfer_vxm_to_mem_east(sinks);
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

private:
    static void check_mxm_index(std::size_t index)
    {
        if (index >= kMxmCount) {
            throw std::out_of_range("MXM index is outside the two east-side MXM units");
        }
    }

    static const TileArrayModel::StreamSlot& mem_edge_stream(
        const TileArrayModel& mem,
        std::size_t tile,
        std::size_t lane,
        std::size_t stream)
    {
        if (stream < hw::kEastStreams) {
            return mem.east_register(tile, lane, 0, stream);
        }
        return mem.west_register(tile, lane, 0, stream - hw::kEastStreams);
    }

    void tick_mxm_controls(LogSinks sinks)
    {
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            if (sinks.mxm != nullptr) {
                *sinks.mxm << "mxm" << mxm << " cycle " << cycle_ << '\n';
            }
            auto provider = [this, mxm, sinks](std::size_t tile) {
                if (sinks.mxm != nullptr && (!sinks.mxm_log_tile.has_value() || tile == *sinks.mxm_log_tile)) {
                    *sinks.mxm << "  MEM.sreg11 -> MXM" << mxm << " tile " << tile << '\n';
                }
                return collect_mxm_weight_input_from_streams(mxm, tile);
            };
            if (sinks.mxm != nullptr) {
                mxms_[mxm].control().tick(*sinks.mxm, provider, false, sinks.mxm_log_tile);
            } else {
                static NullStream null_stream;
                mxms_[mxm].control().tick(null_stream.stream(), provider, false);
            }
        }
    }

    void tick_mxm_datapaths(LogSinks sinks)
    {
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            mxms_[mxm].tick_datapath(mem_, mxm, sinks.mxm, sinks.mxm_log_tile);
        }
    }

    MxmControlSlice::WeightInput collect_mxm_weight_input_from_streams(std::size_t mxm, std::size_t tile)
    {
        constexpr auto kTargetSreg = hw::kStreamRegisterColumns - 1;
        auto input = MxmControlSlice::WeightInput {};
        const auto stream_base = mxm * hw::kMxmLoadStreamsPerCycle;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
                const auto slot = mem_.consume_east_register(tile, lane, kTargetSreg, stream_base + stream);
                if (!slot.has_value()) {
                    throw std::logic_error("MXM IW reached tile before MEM east stream arrived at sreg11");
                }
                input[lane][stream] = MxmArray::Supercell::InputWord {
                    static_cast<std::int8_t>(slot->data),
                    stream + 1 == hw::kMxmLoadStreamsPerCycle,
                };
            }
        }
        return input;
    }

    bool has_complete_vxm_input(std::size_t tile) const
    {
        const auto& required_streams = vxm_.required_streams_at(tile);
        if (!required_streams.has_value()) {
            return false;
        }

        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kStreams; ++stream) {
                if (!(*required_streams)[stream]) {
                    continue;
                }
                if (!mem_edge_stream(mem_, tile, lane, stream).has_value()) {
                    return false;
                }
            }
        }
        return true;
    }

    void transfer_mem_west_to_vxm(LogSinks sinks)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!has_complete_vxm_input(tile)) {
                continue;
            }

            auto streams = VxmSlice::StreamMatrix {};
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                for (std::size_t stream = 0; stream < hw::kStreams; ++stream) {
                    const auto& slot = mem_edge_stream(mem_, tile, lane, stream);
                    streams[lane][stream] = slot.has_value() ? slot->data : 0;
                }
            }
            vxm_.set_stream_inputs(tile, streams);
            if (sinks.vxm != nullptr && (!sinks.vxm_log_tile.has_value() || tile == *sinks.vxm_log_tile)) {
                *sinks.vxm << "  MEM.edge -> VXM tile " << tile << '\n';
            }
        }
    }

    void transfer_vxm_to_mem_east(LogSinks sinks)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& output = vxm_.output_at(tile);
            if (!output.has_value()) {
                continue;
            }

            if (output->stream >= hw::kStreams) {
                throw std::out_of_range("VXM output stream is outside the 64-stream lane");
            }
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto word = TileArrayModel::DataWord {
                    static_cast<std::uint8_t>(output->values[lane]),
                    lane + 1 == hw::kLanesPerTile,
                };
                if (output->stream < hw::kEastStreams) {
                    mem_.set_east_stream_input(tile, lane, output->stream, word);
                } else {
                    mem_.set_west_stream_input(tile, lane, output->stream - hw::kEastStreams, word);
                }
            }
            if (sinks.mem != nullptr && (!sinks.mem_log_tile.has_value() || tile == *sinks.mem_log_tile)) {
                *sinks.mem << "  VXM -> MEM tile " << tile << " stream " << output->stream << '\n';
            }
        }
    }

    class NullStream {
    public:
        std::ostream& stream()
        {
            return stream_;
        }

    private:
        class Buffer : public std::streambuf {
        public:
            int overflow(int c) override
            {
                return c;
            }
        };

        Buffer buffer_{};
        std::ostream stream_{&buffer_};
    };

    TileArrayModel mem_{};
    VxmSlice vxm_{};
    std::array<Mxm, kMxmCount> mxms_{};
    InstructionControlUnit icu_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
