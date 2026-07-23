#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/instruction_packet.hpp"
#include "ftlpu/core/stream.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ftlpu {

enum class IcuLocationKind : std::uint8_t {
    Mem,
    Vxm,
    MxmLoad,
    MxmCompute,
};

struct IcuLocation {
    IcuLocationKind kind{IcuLocationKind::Mem};
    std::size_t unit{0};
    std::size_t index{0};

    static IcuLocation Mem(std::size_t slice) { return {IcuLocationKind::Mem, 0, slice}; }
    static IcuLocation Vxm(std::size_t alu) { return {IcuLocationKind::Vxm, 0, alu}; }
    static IcuLocation MxmLoad(std::size_t mxm) { return {IcuLocationKind::MxmLoad, mxm, 0}; }
    static IcuLocation MxmCompute(std::size_t mxm) { return {IcuLocationKind::MxmCompute, mxm, 0}; }

    friend bool operator==(const IcuLocation&, const IcuLocation&) = default;
};

// Physical SR attachment is topology state, not part of Fetch encoding.
class IcuFetchPortMap {
public:
    explicit IcuFetchPortMap(std::size_t default_column = 0)
    {
        mem_.fill(default_column);
        vxm_.fill(default_column);
        mxm_load_.fill(default_column);
        mxm_compute_.fill(default_column);
    }

    void map(IcuLocation location, std::size_t column)
    {
        slot(location) = column;
    }

    std::size_t column(IcuLocation location) const
    {
        return slot(location);
    }

private:
    std::size_t& slot(IcuLocation location)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem: return mem_.at(location.index);
        case IcuLocationKind::Vxm: return vxm_.at(location.index);
        case IcuLocationKind::MxmLoad: return mxm_load_.at(location.unit);
        case IcuLocationKind::MxmCompute: return mxm_compute_.at(location.unit);
        }
        throw std::logic_error("unknown ICU location kind");
    }

    const std::size_t& slot(IcuLocation location) const
    {
        return const_cast<IcuFetchPortMap*>(this)->slot(location);
    }

    std::array<std::size_t, hw::kMemSliceColumns> mem_{};
    std::array<std::size_t, 16> vxm_{};
    std::array<std::size_t, 2> mxm_load_{};
    std::array<std::size_t, 2> mxm_compute_{};
};

// Per-ICU transient storage for exactly one 640-byte Ifetch.
class IcuFetchBuffer {
public:
    using PacketArray = std::array<isa::EncodedInstructionPacket, hw::kIcuFetchPackets>;

    void reset()
    {
        bytes_.fill(0);
        for (auto& mask : received_) mask.reset();
        for (auto& tag : vector_tags_) tag.reset();
        next_vector_index_ = 0;
    }

    std::size_t current_vector_index() const noexcept { return next_vector_index_; }
    const std::bitset<hw::kTileRows>& received_mask(std::size_t vector) const
    {
        return received_.at(vector);
    }

    void accept_segment(std::size_t tile, const StreamSegment16& segment)
    {
        if (tile >= hw::kTileRows) {
            throw std::out_of_range("ICU fetch tile is outside the 20-tile vector");
        }
        const auto tag = validate_segment(segment);
        const auto vector = find_or_allocate_vector(tag);
        if (received_[vector].test(tile)) {
            throw std::logic_error("ICU fetch received a duplicate tile segment");
        }
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            bytes_[vector * hw::kPhysicalVectorBytes
                + tile * hw::kLanesPerTile + lane] = segment[lane].data;
        }
        received_[vector].set(tile);
        while (next_vector_index_ < hw::kIcuFetchVectorCount
            && received_[next_vector_index_].all()) {
            ++next_vector_index_;
        }
    }

    bool complete() const noexcept
    {
        for (const auto& mask : received_) {
            if (!mask.all()) return false;
        }
        return true;
    }

    PacketArray packets() const
    {
        if (!complete()) {
            throw std::logic_error("cannot decode an incomplete ICU fetch buffer");
        }
        PacketArray packets{};
        for (std::size_t packet = 0; packet < packets.size(); ++packet) {
            for (std::size_t byte = 0; byte < hw::kEncodedInstructionPacketBytes; ++byte) {
                packets[packet].bytes[byte] =
                    bytes_[packet * hw::kEncodedInstructionPacketBytes + byte];
            }
        }
        return packets;
    }

private:
    static std::uint64_t validate_segment(const StreamSegment16& segment)
    {
        if (!segment[0].valid) {
            throw std::logic_error("ICU fetch received an invalid segment");
        }
        const auto tag = segment[0].vector_tag;
        for (std::size_t lane = 0; lane < segment.size(); ++lane) {
            if (!segment[lane].valid || segment[lane].vector_tag != tag) {
                throw std::logic_error("ICU fetch segment mixes validity or vector tags across lanes");
            }
        }
        return tag;
    }

    std::size_t find_or_allocate_vector(std::uint64_t tag)
    {
        for (std::size_t vector = 0; vector < vector_tags_.size(); ++vector) {
            if (vector_tags_[vector].has_value() && *vector_tags_[vector] == tag) {
                return vector;
            }
        }
        for (std::size_t vector = 0; vector < vector_tags_.size(); ++vector) {
            if (!vector_tags_[vector].has_value()) {
                vector_tags_[vector] = tag;
                return vector;
            }
        }
        throw std::logic_error("ICU fetch observed a third vector tag in one 640-byte fetch");
    }

    std::array<std::uint8_t, hw::kIcuFetchBufferBytes> bytes_{};
    std::array<std::bitset<hw::kTileRows>, hw::kIcuFetchVectorCount> received_{};
    std::array<std::optional<std::uint64_t>, hw::kIcuFetchVectorCount> vector_tags_{};
    std::size_t next_vector_index_{0};
};

struct IcuFetchState {
    StreamId source_stream{};
    IcuFetchBuffer buffer{};
    bool iq_reserved{false};
};

} // namespace ftlpu
