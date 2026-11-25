#pragma once

#include "ftlpu/core/sram_1rw.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ftlpu::timing {

// Cycle-accurate model of rtl/common/lpu_sram_fifo.sv.  This is an explicit
// SRAM-backed FIFO with one read in flight and a held output register.  It is
// intentionally separate from the ordinary register FIFO model.
template <typename T, std::size_t Depth>
class SramFifo final : public ClockedModule {
public:
    static_assert(Depth > 0);

    struct Inputs {
        bool rst_ni{true};
        bool enqueue_valid_i{false};
        T enqueue_data_i{};
        bool dequeue_ready_i{false};
    };

    struct Outputs {
        bool enqueue_ready_o{false};
        bool dequeue_valid_o{false};
        T dequeue_data_o{};
        std::size_t level_o{0};
        bool enqueue_fire{false};
        bool dequeue_fire{false};
        bool dequeue_request{false};
        bool read_pending{false};
    };

    struct State {
        std::size_t write_ptr{0};
        std::size_t read_ptr{0};
        std::size_t level{0};
        bool read_pending{false};
        bool output_valid{false};
        T output_data{};
    };

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }

    void low() override
    {
        ports_ = evaluate();
        state_.prepare();
        auto& next = state_.next_value();

        typename Sram1Rw<T, Depth>::Inputs storage_inputs{};
        storage_inputs.rst_ni = inputs_.rst_ni;

        if (!inputs_.rst_ni) {
            next = {};
        } else {
            const auto& current = state_.current();
            if (ports_.enqueue_fire)
                next.write_ptr = increment(current.write_ptr);
            if (ports_.dequeue_request)
                next.read_ptr = increment(current.read_ptr);

            if (ports_.enqueue_fire && !ports_.dequeue_fire) ++next.level;
            else if (!ports_.enqueue_fire && ports_.dequeue_fire) --next.level;

            const auto sram = storage_.outputs();
            if (ports_.dequeue_request && !sram.rvalid_o)
                next.read_pending = true;
            else if (!ports_.dequeue_request && sram.rvalid_o)
                next.read_pending = false;

            if (ports_.dequeue_fire) next.output_valid = false;
            if (sram.rvalid_o) {
                next.output_valid = true;
                next.output_data = sram.rdata_o;
            }

            storage_inputs.cen_i = ports_.dequeue_request || ports_.enqueue_fire;
            storage_inputs.wen_i = ports_.enqueue_fire;
            storage_inputs.address_i = ports_.enqueue_fire
                ? current.write_ptr : current.read_ptr;
            storage_inputs.wdata_i = inputs_.enqueue_data_i;
        }

        if (next.level > Depth)
            throw std::logic_error("SRAM FIFO occupancy exceeds configured depth");
        storage_.drive(std::move(storage_inputs));
        storage_.low();
    }

    void validate_posedge() const override
    {
        state_.validate_commit();
        storage_.validate_posedge();
    }

    void posedge() override
    {
        storage_.posedge();
        state_.commit();
    }

    void high() override
    {
        storage_.high();
        ports_ = evaluate();
    }

    [[nodiscard]] const Outputs& outputs() const noexcept { return ports_; }
    [[nodiscard]] const State& current() const noexcept
    { return state_.current(); }
    [[nodiscard]] const Sram1Rw<T, Depth>& storage() const noexcept
    { return storage_; }

private:
    [[nodiscard]] static std::size_t increment(std::size_t pointer) noexcept
    { return pointer + 1 == Depth ? 0 : pointer + 1; }

    [[nodiscard]] Outputs evaluate() const
    {
        const auto& current = state_.current();
        Outputs result{};
        result.dequeue_valid_o = current.output_valid;
        result.dequeue_data_o = current.output_data;
        result.level_o = current.level;
        result.dequeue_fire = result.dequeue_valid_o
            && inputs_.dequeue_ready_i;
        const std::size_t reserved = (current.output_valid ? 1u : 0u)
            + (current.read_pending ? 1u : 0u);
        const bool unread = current.level > reserved;
        const bool output_slot = !current.output_valid || result.dequeue_fire;
        result.dequeue_request = !current.read_pending && output_slot && unread;
        result.enqueue_ready_o = !result.dequeue_request
            && (current.level < Depth || result.dequeue_fire);
        result.enqueue_fire = inputs_.enqueue_valid_i
            && result.enqueue_ready_o;
        result.read_pending = current.read_pending;
        return result;
    }

    Inputs inputs_{};
    Outputs ports_{};
    ClockedState<State> state_{};
    Sram1Rw<T, Depth> storage_{};
};

} // namespace ftlpu::timing
