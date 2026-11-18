#pragma once

#include "ftlpu/core/timing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ftlpu {

inline constexpr std::uint8_t kInstructionDmaFaultId = 2;
inline constexpr std::uint8_t kInstructionDmaFaultResponse = 3;

struct InstructionDmaProtocolConfig {
    std::size_t data_width_bits{256};
    std::size_t id_width_bits{8};
    std::size_t outstanding_depth{16};
    std::size_t response_fifo_depth{16};

    [[nodiscard]] std::size_t data_bytes() const noexcept
    {
        return data_width_bits / 8;
    }

    void validate() const
    {
        if (data_width_bits != 256)
            throw std::invalid_argument("instruction DMA protocol width must be 256 bits");
        if (id_width_bits == 0 || id_width_bits > 8)
            throw std::invalid_argument("instruction DMA protocol ID width must be in [1, 8]");
        if (outstanding_depth == 0
            || outstanding_depth > (std::size_t{1} << id_width_bits))
            throw std::invalid_argument("instruction DMA outstanding depth exceeds the ID space");
        if (response_fifo_depth == 0)
            throw std::invalid_argument("instruction DMA response FIFO depth must be non-zero");
    }
};

struct InstructionDmaResponseBeat {
    static constexpr std::size_t kDataBytes = 32;
    std::array<std::uint8_t, kDataBytes> data{};
    std::uint32_t keep{0};
    bool last{false};
    std::uint8_t id{0};
    bool error{false};
};

struct InstructionDmaProtocolInputs {
    bool reset_n{true};
    bool command_valid{false};
    std::uint64_t command_address{0};
    std::uint32_t command_length{0};
    std::uint8_t command_id{0};
    bool request_ready{false};
    bool response_valid{false};
    InstructionDmaResponseBeat response{};
    bool reassembler_ready{false};
};

// Pin-level behavioral counterpart of rtl/program_load/lpu_instruction_dma.sv.
// It is intentionally independent of Ddr4Model. The integrated
// InstructionFetchDma owns the DDR adapter and uses this class as its only
// outstanding-ID, response-FIFO and sticky-fault authority.
class InstructionDmaProtocolCore final : public timing::ClockedModule {
public:
    explicit InstructionDmaProtocolCore(
        InstructionDmaProtocolConfig config = {})
        : config_(config), state_(make_state(config)), next_(std::nullopt)
    {
        config_.validate();
    }

    void drive(InstructionDmaProtocolInputs inputs)
    {
        if (next_)
            throw std::logic_error("instruction DMA protocol inputs arrived after LOW");
        if (!inputs.reset_n)
            reset(); // Models the RTL asynchronous active-low reset assertion.
        inputs_ = std::move(inputs);
        driven_ = true;
    }

    void low() override
    {
        if (next_)
            throw std::logic_error("instruction DMA protocol LOW executed more than once");
        if (!driven_)
            throw std::logic_error("instruction DMA protocol LOW requires driven inputs");

        if (!inputs_.reset_n) {
            next_ = make_state(config_);
            return;
        }

        next_ = state_;
        const bool outEnqueue = command_fire();
        const bool outDequeue = outstanding_dequeue_fire();
        const bool rxEnqueue = response_input_fire();
        const bool rxDequeue = response_fire();

        if (outEnqueue) {
            next_->outstanding_storage[state_.outstanding_write_pointer]
                = inputs_.command_id;
            next_->outstanding_write_pointer = increment(
                state_.outstanding_write_pointer,
                config_.outstanding_depth);
        }
        if (outDequeue) {
            next_->outstanding_read_pointer = increment(
                state_.outstanding_read_pointer,
                config_.outstanding_depth);
        }
        update_level(next_->outstanding_level, outEnqueue, outDequeue);

        if (rxEnqueue) {
            next_->response_storage[state_.response_write_pointer]
                = inputs_.response;
            next_->response_write_pointer = increment(
                state_.response_write_pointer,
                config_.response_fifo_depth);
        }
        if (rxDequeue) {
            next_->response_read_pointer = increment(
                state_.response_read_pointer,
                config_.response_fifo_depth);
        }
        update_level(next_->response_level, rxEnqueue, rxDequeue);

        if (!state_.fault) {
            if (response_id_invalid()) {
                next_->fault = true;
                next_->fault_code = kInstructionDmaFaultId;
            } else if (response_error()) {
                next_->fault = true;
                next_->fault_code = kInstructionDmaFaultResponse;
            }
        }
    }

    void validate_posedge() const override
    {
        if (!next_)
            throw std::logic_error("instruction DMA protocol posedge requires LOW");
        if (next_->outstanding_level > config_.outstanding_depth
            || next_->response_level > config_.response_fifo_depth)
            throw std::logic_error("instruction DMA protocol FIFO level is invalid");
    }

    void posedge() override
    {
        validate_posedge();
        state_ = std::move(*next_);
        next_.reset();
        driven_ = false;
        inputs_ = {};
    }

    void high() override {}

    void reset()
    {
        if (next_)
            throw std::logic_error("instruction DMA protocol reset cannot interrupt LOW");
        state_ = make_state(config_);
        driven_ = false;
        inputs_ = {};
    }

    [[nodiscard]] bool command_ready() const noexcept
    {
        return driven_ && inputs_.request_ready && outstanding_enqueue_ready()
            && !state_.fault;
    }
    [[nodiscard]] bool command_fire() const noexcept
    {
        return driven_ && inputs_.command_valid && command_ready();
    }
    [[nodiscard]] bool request_valid() const noexcept
    {
        return driven_ && inputs_.command_valid && outstanding_enqueue_ready()
            && !state_.fault;
    }
    [[nodiscard]] bool request_fire() const noexcept
    {
        return request_valid() && inputs_.request_ready;
    }
    [[nodiscard]] std::uint64_t request_address() const noexcept
    {
        return inputs_.command_address;
    }
    [[nodiscard]] std::uint32_t request_length() const noexcept
    {
        return inputs_.command_length;
    }
    [[nodiscard]] std::uint8_t request_id() const noexcept
    {
        return inputs_.command_id;
    }

    [[nodiscard]] bool response_ready() const noexcept
    {
        return driven_ && !state_.fault && response_enqueue_ready();
    }
    [[nodiscard]] bool response_input_fire() const noexcept
    {
        return driven_ && inputs_.response_valid && response_ready();
    }
    [[nodiscard]] bool response_valid() const noexcept
    {
        return state_.response_level != 0;
    }
    [[nodiscard]] const InstructionDmaResponseBeat& response_front() const
    {
        if (!response_valid())
            throw std::logic_error("instruction DMA response FIFO is empty");
        return state_.response_storage[state_.response_read_pointer];
    }
    [[nodiscard]] bool response_fire() const noexcept
    {
        return driven_ && response_valid() && inputs_.reassembler_ready;
    }

    [[nodiscard]] bool response_id_invalid() const noexcept
    {
        if (!response_input_fire()) return false;
        return state_.outstanding_level == 0
            || inputs_.response.id
                != state_.outstanding_storage[state_.outstanding_read_pointer];
    }
    [[nodiscard]] bool response_error() const noexcept
    {
        return response_input_fire() && inputs_.response.error;
    }
    [[nodiscard]] bool fault() const noexcept
    {
        return state_.fault || response_id_invalid() || response_error();
    }
    [[nodiscard]] std::uint8_t fault_code() const noexcept
    {
        if (state_.fault) return state_.fault_code;
        if (response_id_invalid()) return kInstructionDmaFaultId;
        if (response_error()) return kInstructionDmaFaultResponse;
        return 0;
    }

    [[nodiscard]] std::size_t outstanding() const noexcept
    {
        return state_.outstanding_level;
    }
    [[nodiscard]] std::size_t response_fifo_occupancy() const noexcept
    {
        return state_.response_level;
    }
    [[nodiscard]] bool idle() const noexcept
    {
        return state_.outstanding_level == 0 && state_.response_level == 0;
    }
    [[nodiscard]] std::size_t next_outstanding() const noexcept
    {
        return next_ ? next_->outstanding_level : state_.outstanding_level;
    }
    [[nodiscard]] std::size_t next_response_fifo_occupancy() const noexcept
    {
        return next_ ? next_->response_level : state_.response_level;
    }
    [[nodiscard]] bool next_idle() const noexcept
    {
        const auto& state = next_ ? *next_ : state_;
        return state.outstanding_level == 0 && state.response_level == 0;
    }
    [[nodiscard]] const InstructionDmaProtocolConfig& config() const noexcept
    {
        return config_;
    }

private:
    struct State {
        std::vector<std::uint8_t> outstanding_storage{};
        std::size_t outstanding_write_pointer{0};
        std::size_t outstanding_read_pointer{0};
        std::size_t outstanding_level{0};
        std::vector<InstructionDmaResponseBeat> response_storage{};
        std::size_t response_write_pointer{0};
        std::size_t response_read_pointer{0};
        std::size_t response_level{0};
        bool fault{false};
        std::uint8_t fault_code{0};
    };

    static State make_state(const InstructionDmaProtocolConfig& config)
    {
        State state{};
        state.outstanding_storage.resize(config.outstanding_depth);
        state.response_storage.resize(config.response_fifo_depth);
        return state;
    }

    static std::size_t increment(std::size_t pointer, std::size_t depth) noexcept
    {
        return pointer + 1 == depth ? 0 : pointer + 1;
    }

    static void update_level(std::size_t& level, bool enqueue, bool dequeue)
    {
        if (enqueue && !dequeue) ++level;
        else if (!enqueue && dequeue) --level;
    }

    [[nodiscard]] bool outstanding_dequeue_fire() const noexcept
    {
        return state_.outstanding_level != 0 && response_input_fire()
            && inputs_.response.last;
    }

    [[nodiscard]] bool outstanding_enqueue_ready() const noexcept
    {
        return state_.outstanding_level < config_.outstanding_depth
            || outstanding_dequeue_fire();
    }

    [[nodiscard]] bool response_enqueue_ready() const noexcept
    {
        return state_.response_level < config_.response_fifo_depth
            || response_fire();
    }

    InstructionDmaProtocolConfig config_{};
    State state_{};
    std::optional<State> next_{};
    InstructionDmaProtocolInputs inputs_{};
    bool driven_{false};
};

} // namespace ftlpu
