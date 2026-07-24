#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
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
    static constexpr std::size_t kMxmCountPerHemisphere = 2;
    static constexpr std::size_t kMxmCount = hw::kHemispheres * kMxmCountPerHemisphere;

    struct LogSinks {
        std::ostream* icu{nullptr};
        std::ostream* mem{nullptr};
        std::ostream* mxm{nullptr};
        std::ostream* vxm{nullptr};
        std::ostream* system{nullptr};
        std::optional<std::size_t> mem_log_tile{};
        std::optional<std::size_t> mxm_log_tile{};
        std::optional<std::size_t> vxm_log_tile{};
        std::ostream* sxm{nullptr};
    };

    TspSliceSystem()
        : sxms_ {
            SxmSlice(make_sxm_port_map()),
            SxmSlice(make_sxm_port_map()),
        }
    {
    }

    void initialize_mem_sram_lane_byte(
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        std::uint8_t value)
    {
        initialize_mem_sram_lane_byte(Hemisphere::East, column, tile, row, lane, value);
    }

    void initialize_mem_sram_lane_byte(
        Hemisphere hemisphere,
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        std::uint8_t value)
    {
        mems_[hemisphere_index(hemisphere)].set_sram_lane_byte(column, tile, row, lane, value);
    }

    std::uint8_t read_mem_sram_lane_byte(
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        return read_mem_sram_lane_byte(Hemisphere::East, column, tile, row, lane);
    }

    std::uint8_t read_mem_sram_lane_byte(
        Hemisphere hemisphere,
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        return mems_[hemisphere_index(hemisphere)].sram_lane_byte(column, tile, row, lane);
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
        icu_.dispatch(mems_, vxm_, sxms_, mxms_, sinks.icu);
        tick_mxm_controls(sinks);
        tick_mxm_datapaths(sinks);
        vxm_.prepare_cycle();
        transfer_mem_edges_to_vxm(sinks);
        transfer_unconsumed_streams_across_vxm(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        transfer_vxm_to_mem_edges(sinks);
        for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
            if (sinks.mem != nullptr) {
                *sinks.mem << "mem." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                           << " cycle " << cycle_ << '\n';
                mems_[hemisphere].tick(sxms_[hemisphere], *sinks.mem, sinks.mem_log_tile);
            } else {
                mems_[hemisphere].tick(sxms_[hemisphere]);
            }
            if (sinks.sxm != nullptr) {
                *sinks.sxm << "sxm."
                            << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                            << " system_cycle " << cycle_ << '\n';
                sxms_[hemisphere].log_cycle(*sinks.sxm);
            }
        }
        ++cycle_;
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

private:
    static SxmStreamPortMap make_sxm_port_map()
    {
        return SxmStreamPortMap::BetweenColumns(
            hw::kMemEastBoundaryStreamRegisterColumn,
            hw::kMxmBoundaryStreamRegisterColumn,
            hw::kMxmBoundaryStreamRegisterColumn,
            hw::kMemEastBoundaryStreamRegisterColumn);
    }

    static Hemisphere mxm_hemisphere(std::size_t mxm)
    {
        return static_cast<Hemisphere>(mxm / kMxmCountPerHemisphere);
    }

    static std::size_t local_mxm_index(std::size_t mxm)
    {
        return mxm % kMxmCountPerHemisphere;
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
                    *sinks.mxm << "  SXM.sreg12 -> MXM" << mxm << " tile " << tile << '\n';
                }
                try {
                    return collect_mxm_weight_input_from_streams(mxm, tile);
                } catch (const std::exception& ex) {
                    throw std::logic_error(
                        "MXM" + std::to_string(mxm)
                        + " IW tile " + std::to_string(tile) + ": " + ex.what());
                }
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
            const auto hemisphere = hemisphere_index(mxm_hemisphere(mxm));
            mxms_[mxm].tick_datapath(
                mems_[hemisphere], local_mxm_index(mxm), sinks.mxm, sinks.mxm_log_tile);
        }
    }

    MxmControlSlice::WeightInput collect_mxm_weight_input_from_streams(std::size_t mxm, std::size_t tile)
    {
        constexpr auto kTargetSreg = hw::kMxmBoundaryStreamRegisterColumn;
        auto input = MxmControlSlice::WeightInput {};
        const auto hemisphere = hemisphere_index(mxm_hemisphere(mxm));
        const auto stream_base = local_mxm_index(mxm) * hw::kMxmLoadStreamsPerCycle;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t column = 0; column < hw::kMxmSupercellColumns; ++column) {
                const auto low_stream = stream_base + column * hw::kMxmWeightBytesPerValue;
                const auto low = mems_[hemisphere].consume_east_register(tile, lane, kTargetSreg, low_stream);
                const auto high = mems_[hemisphere].consume_east_register(tile, lane, kTargetSreg, low_stream + 1);
                if (!low.has_value() || !high.has_value()) {
                    throw std::logic_error("MXM IW reached tile before both FP16 weight streams arrived at sreg12");
                }
                const auto bits = static_cast<std::uint16_t>(low->data)
                    | (static_cast<std::uint16_t>(high->data) << 8);
                input[lane][column] = MxmArray::Supercell::InputWord {
                    Fp16::from_bits(bits).to_float(),
                    column + 1 == hw::kMxmSupercellColumns,
                };
            }
        }
        return input;
    }

    bool has_complete_vxm_input(Hemisphere hemisphere, std::size_t tile) const
    {
        const auto& required_streams = vxm_.required_streams_at(hemisphere, tile);
        if (!required_streams.has_value()) {
            return false;
        }

        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kStreams; ++stream) {
                if (!(*required_streams)[stream]) {
                    continue;
                }
                if (!mem_edge_stream(mems_[hemisphere_index(hemisphere)], tile, lane, stream).has_value()) {
                    return false;
                }
            }
        }
        return true;
    }

    void transfer_mem_edges_to_vxm(LogSinks sinks)
    {
        for (std::size_t hemisphere_index_value = 0; hemisphere_index_value < hw::kHemispheres;
             ++hemisphere_index_value) {
            const auto hemisphere = static_cast<Hemisphere>(hemisphere_index_value);
            const auto& mem = mems_[hemisphere_index_value];
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                if (!has_complete_vxm_input(hemisphere, tile)) {
                    continue;
                }

                auto streams = VxmSlice::StreamMatrix {};
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    for (std::size_t stream = 0; stream < hw::kStreams; ++stream) {
                        const auto& slot = mem_edge_stream(mem, tile, lane, stream);
                        streams[lane][stream] = slot.has_value() ? slot->data : 0;
                    }
                }
                vxm_.set_stream_inputs(hemisphere, tile, streams);
                if (sinks.vxm != nullptr
                    && (!sinks.vxm_log_tile.has_value() || tile == *sinks.vxm_log_tile)) {
                    *sinks.vxm << "  MEM." << hemisphere_short_name(hemisphere)
                               << ".edge -> VXM tile " << tile << '\n';
                }
            }
        }
    }

    void transfer_unconsumed_streams_across_vxm(LogSinks sinks)
    {
        for (std::size_t source_index = 0; source_index < hw::kHemispheres; ++source_index) {
            const auto source = static_cast<Hemisphere>(source_index);
            const auto destination_index = source_index ^ 1;
            auto& destination = mems_[destination_index];
            const auto& source_mem = mems_[source_index];
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                const auto& required = vxm_.required_streams_at(source, tile);
                for (std::size_t stream = 0; stream < hw::kWestStreams; ++stream) {
                    const auto packed = hw::kEastStreams + stream;
                    if (required.has_value() && (*required)[packed]) continue;

                    auto complete = true;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        complete = complete
                            && mem_edge_stream(source_mem, tile, lane, packed).has_value();
                    }
                    if (!complete) continue;

                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        const auto& cell = mem_edge_stream(source_mem, tile, lane, packed);
                        destination.set_east_stream_input(
                            tile,
                            lane,
                            stream,
                            TileArrayModel::DataWord {cell->data, cell->last});
                    }
                    if (sinks.system != nullptr) {
                        *sinks.system << "  passive VXM bridge "
                                      << hemisphere_short_name(source) << ".W" << stream
                                      << " -> "
                                      << hemisphere_short_name(
                                             static_cast<Hemisphere>(destination_index))
                                      << ".E" << stream << " tile " << tile << '\n';
                    }
                }
            }
        }
    }

    void transfer_vxm_to_mem_edges(LogSinks sinks)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (const auto& output : vxm_.outputs_at(tile)) {
                if (output.stream + output.byte_count > hw::kStreams) {
                    throw std::out_of_range("VXM output stream is outside the 64-stream lane");
                }
                auto& mem = mems_[hemisphere_index(output.hemisphere)];
                for (std::size_t byte = 0; byte < output.byte_count; ++byte) {
                    const auto stream = output.stream + byte;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        const auto word = TileArrayModel::DataWord {
                            output.byte_values[lane][byte],
                            lane + 1 == hw::kLanesPerTile,
                        };
                        if (stream < hw::kEastStreams) {
                            mem.set_east_stream_input(tile, lane, stream, word);
                        } else {
                            mem.set_west_stream_input(tile, lane, stream - hw::kEastStreams, word);
                        }
                    }
                }
                if (sinks.mem != nullptr && (!sinks.mem_log_tile.has_value() || tile == *sinks.mem_log_tile)) {
                    *sinks.mem << "  VXM -> MEM." << hemisphere_short_name(output.hemisphere)
                               << " tile " << tile << " stream " << output.stream
                               << " bytes=" << output.byte_count << '\n';
                }
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

    std::array<TileArrayModel, hw::kHemispheres> mems_{};
    VxmSlice vxm_{};
    std::array<SxmSlice, hw::kHemispheres> sxms_;
    std::array<Mxm, kMxmCount> mxms_{};
    InstructionControlUnit icu_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
