#pragma once

#include "ftlpu/core/timing.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu::timing {

// Cycle-accurate model of rtl/common/lpu_sram.sv.  The single port accepts
// either one write or one read at a rising edge.  Read data/valid are
// registered; reset clears only those output registers and deliberately keeps
// the storage array intact.
template <typename T, std::size_t Depth>
class Sram1Rw final : public ClockedModule {
public:
    static_assert(Depth > 0);

    struct Inputs {
        bool rst_ni{true};
        bool cen_i{false};
        bool wen_i{false};
        std::size_t address_i{0};
        T wdata_i{};
    };

    struct Outputs {
        bool rvalid_o{false};
        T rdata_o{};
    };

    struct State {
        bool rvalid{false};
        T rdata{};
    };

    Sram1Rw()
        : storage_(std::make_shared<Storage>())
    {}

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }

    void low() override
    {
        if (inputs_.cen_i && inputs_.address_i >= Depth)
            throw std::out_of_range("1RW SRAM address exceeds configured depth");

        state_.prepare();
        pending_write_.reset();
        if (!inputs_.rst_ni) {
            state_.next_value() = {};
            return;
        }

        auto& next = state_.next_value();
        next.rvalid = inputs_.cen_i && !inputs_.wen_i;
        if (inputs_.cen_i && inputs_.wen_i) {
            pending_write_ = Write{inputs_.address_i, inputs_.wdata_i};
        } else if (inputs_.cen_i) {
            next.rdata = read_storage(inputs_.address_i);
        }
    }

    void validate_posedge() const override { state_.validate_commit(); }

    void posedge() override
    {
        state_.commit();
        if (pending_write_) write_storage(pending_write_->address,
                                          pending_write_->data);
        pending_write_.reset();
    }

    void high() override {}

    [[nodiscard]] Outputs outputs() const
    {
        return Outputs{state_.current().rvalid, state_.current().rdata};
    }
    [[nodiscard]] bool rvalid() const noexcept
    { return state_.current().rvalid; }
    [[nodiscard]] const T& rdata() const noexcept
    { return state_.current().rdata; }
    [[nodiscard]] const State& current() const noexcept
    { return state_.current(); }
    [[nodiscard]] const State& next() const noexcept
    { return state_.next(); }

    // Program loading is an initialization/debug convenience, not an extra
    // modeled SRAM port.  Runtime writes must use drive()/LOW/POSEDGE.
    void seed(std::size_t address, const T& value)
    {
        if (address >= Depth)
            throw std::out_of_range("1RW SRAM seed address exceeds configured depth");
        write_storage(address, value);
    }

    void seed_image(std::shared_ptr<const std::vector<T>> image)
    {
        if (!image || image->empty() || image->size() > Depth)
            throw std::invalid_argument("invalid 1RW SRAM image");
        auto replacement = std::make_shared<Storage>();
        replacement->data = *image;
        replacement->valid.assign(image->size(), true);
        storage_ = std::move(replacement);
    }

    [[nodiscard]] bool initialized(std::size_t address) const
    {
        if (address >= Depth)
            throw std::out_of_range("1RW SRAM probe address exceeds configured depth");
        return address < storage_->valid.size() && storage_->valid[address];
    }

    [[nodiscard]] T inspect(std::size_t address) const
    { return read_storage(address); }

private:
    struct Storage {
        std::vector<T> data{};
        std::vector<bool> valid{};
    };
    struct Write {
        std::size_t address{};
        T data{};
    };

    [[nodiscard]] T read_storage(std::size_t address) const
    {
        if (address >= storage_->valid.size() || !storage_->valid[address])
            throw std::logic_error("1RW SRAM read from an uninitialized location");
        return storage_->data[address];
    }

    void write_storage(std::size_t address, const T& value)
    {
        if (storage_.use_count() != 1)
            storage_ = std::make_shared<Storage>(*storage_);
        if (storage_->data.size() <= address) storage_->data.resize(address + 1);
        if (storage_->valid.size() <= address)
            storage_->valid.resize(address + 1, false);
        storage_->data[address] = value;
        storage_->valid[address] = true;
    }

    Inputs inputs_{};
    ClockedState<State> state_{};
    std::shared_ptr<Storage> storage_{};
    std::optional<Write> pending_write_{};
};

} // namespace ftlpu::timing
