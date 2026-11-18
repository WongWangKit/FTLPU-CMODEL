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
    bool rst_ni{true};
    bool ctrl_dma_cmd_valid_i{false};
    std::uint64_t ctrl_dma_cmd_address_i{0};
    std::uint32_t ctrl_dma_cmd_length_i{0};
    std::uint8_t ctrl_dma_cmd_id_i{0};
    bool ddr_read_req_ready_i{false};
    bool ddr_read_rsp_valid_i{false};
    std::array<std::uint8_t, InstructionDmaResponseBeat::kDataBytes>
        ddr_read_rsp_data_i{};
    std::uint32_t ddr_read_rsp_keep_i{0};
    bool ddr_read_rsp_last_i{false};
    std::uint8_t ddr_read_rsp_id_i{0};
    bool ddr_read_rsp_error_i{false};
    bool reasm_rsp_ready_i{false};
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
        if (!inputs.rst_ni)
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

        if (!inputs_.rst_ni) {
            next_ = make_state(config_);
            return;
        }

        next_ = state_;
        const bool outEnqueue = ctrl_dma_cmd_fire();
        const bool outDequeue = outstanding_dequeue_fire();
        const bool rxEnqueue = ddr_read_rsp_fire();
        const bool rxDequeue = reasm_rsp_fire();

        if (outEnqueue) {
            next_->outstanding_storage[state_.outstanding_write_pointer]
                = inputs_.ctrl_dma_cmd_id_i;
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
                = ddr_read_rsp_beat_i();
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
            if (ddr_read_rsp_id_invalid()) {
                next_->fault = true;
                next_->fault_code = kInstructionDmaFaultId;
            } else if (ddr_read_rsp_error()) {
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

    [[nodiscard]] bool ctrl_dma_cmd_ready_o() const noexcept
    {
        return driven_ && inputs_.ddr_read_req_ready_i
            && outstanding_enqueue_ready()
            && !state_.fault;
    }
    [[nodiscard]] bool ctrl_dma_cmd_fire() const noexcept
    {
        return driven_ && inputs_.ctrl_dma_cmd_valid_i
            && ctrl_dma_cmd_ready_o();
    }
    [[nodiscard]] bool ddr_read_req_valid_o() const noexcept
    {
        return driven_ && inputs_.ctrl_dma_cmd_valid_i
            && outstanding_enqueue_ready()
            && !state_.fault;
    }
    [[nodiscard]] bool ddr_read_req_fire() const noexcept
    {
        return ddr_read_req_valid_o() && inputs_.ddr_read_req_ready_i;
    }
    [[nodiscard]] std::uint64_t ddr_read_req_address_o() const noexcept
    {
        return inputs_.ctrl_dma_cmd_address_i;
    }
    [[nodiscard]] std::uint32_t ddr_read_req_length_o() const noexcept
    {
        return inputs_.ctrl_dma_cmd_length_i;
    }
    [[nodiscard]] std::uint8_t ddr_read_req_id_o() const noexcept
    {
        return inputs_.ctrl_dma_cmd_id_i;
    }

    [[nodiscard]] bool ddr_read_rsp_ready_o() const noexcept
    {
        return driven_ && !state_.fault && response_enqueue_ready();
    }
    [[nodiscard]] bool ddr_read_rsp_fire() const noexcept
    {
        return driven_ && inputs_.ddr_read_rsp_valid_i
            && ddr_read_rsp_ready_o();
    }
    [[nodiscard]] bool reasm_rsp_valid_o() const noexcept
    {
        return state_.response_level != 0;
    }
    [[nodiscard]] const InstructionDmaResponseBeat& reasm_rsp_o() const
    {
        if (!reasm_rsp_valid_o())
            throw std::logic_error("instruction DMA response FIFO is empty");
        return state_.response_storage[state_.response_read_pointer];
    }
    [[nodiscard]] const auto& reasm_rsp_data_o() const
    {
        return reasm_rsp_o().data;
    }
    [[nodiscard]] std::uint32_t reasm_rsp_keep_o() const
    {
        return reasm_rsp_o().keep;
    }
    [[nodiscard]] bool reasm_rsp_last_o() const
    {
        return reasm_rsp_o().last;
    }
    [[nodiscard]] std::uint8_t reasm_rsp_id_o() const
    {
        return reasm_rsp_o().id;
    }
    [[nodiscard]] bool reasm_rsp_error_o() const
    {
        return reasm_rsp_o().error;
    }
    [[nodiscard]] bool reasm_rsp_fire() const noexcept
    {
        return driven_ && reasm_rsp_valid_o() && inputs_.reasm_rsp_ready_i;
    }

    [[nodiscard]] bool ddr_read_rsp_id_invalid() const noexcept
    {
        if (!ddr_read_rsp_fire()) return false;
        return state_.outstanding_level == 0
            || inputs_.ddr_read_rsp_id_i
                != state_.outstanding_storage[state_.outstanding_read_pointer];
    }
    [[nodiscard]] bool ddr_read_rsp_error() const noexcept
    {
        return ddr_read_rsp_fire() && inputs_.ddr_read_rsp_error_i;
    }
    [[nodiscard]] bool fault() const noexcept
    {
        return state_.fault || ddr_read_rsp_id_invalid()
            || ddr_read_rsp_error();
    }
    [[nodiscard]] std::uint8_t fault_code() const noexcept
    {
        if (state_.fault) return state_.fault_code;
        if (ddr_read_rsp_id_invalid()) return kInstructionDmaFaultId;
        if (ddr_read_rsp_error()) return kInstructionDmaFaultResponse;
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
        return state_.outstanding_level != 0 && ddr_read_rsp_fire()
            && inputs_.ddr_read_rsp_last_i;
    }

    [[nodiscard]] bool outstanding_enqueue_ready() const noexcept
    {
        return state_.outstanding_level < config_.outstanding_depth
            || outstanding_dequeue_fire();
    }

    [[nodiscard]] bool response_enqueue_ready() const noexcept
    {
        return state_.response_level < config_.response_fifo_depth
            || reasm_rsp_fire();
    }

    [[nodiscard]] InstructionDmaResponseBeat ddr_read_rsp_beat_i() const
    {
        InstructionDmaResponseBeat beat{};
        beat.data = inputs_.ddr_read_rsp_data_i;
        beat.keep = inputs_.ddr_read_rsp_keep_i;
        beat.last = inputs_.ddr_read_rsp_last_i;
        beat.id = inputs_.ddr_read_rsp_id_i;
        beat.error = inputs_.ddr_read_rsp_error_i;
        return beat;
    }

    InstructionDmaProtocolConfig config_{};
    State state_{};
    std::optional<State> next_{};
    InstructionDmaProtocolInputs inputs_{};
    bool driven_{false};
};

} // namespace ftlpu
