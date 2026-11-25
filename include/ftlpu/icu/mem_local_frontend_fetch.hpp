#pragma once

#include "ftlpu/core/sram_1rw.hpp"
#include "ftlpu/icu/raw_imem_word.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// Cycle model for the frozen P2.0 Local-iMEM -> Fetch -> Macro Word FIFO
// boundary.  The public Ports object is the LOW/PRE_POSEDGE view sampled by
// the active edge; accessors after cycle() are the POSEDGE/POST_COMB state.
//
// This component is intentionally independent from the Macro parser.  It is
// used both by the production MEM/MXM Raw-Macro frontends and by directed
// tests that drive request/response/word-boundary backpressure and protocol
// faults.  MEM instantiates a 96-bit local iMEM and MXM instantiates a 128-bit
// local iMEM; VXM/SXM do not use this component until their RTL contract is
// defined.
template <typename RawWordT,
          std::size_t ImemDepth,
          std::size_t ResponseDepth = 4,
          std::size_t WordFifoDepth = 4,
          std::size_t RequestInitiationInterval = 1>
class LocalImemFrontendFetchModel {
public:
    static_assert(ImemDepth > 0);
    static_assert(ResponseDepth > 0);
    static_assert(WordFifoDepth > 0);
    static_assert(RequestInitiationInterval > 0);

    using RawWord = RawWordT;

    enum class FaultCode : std::uint8_t {
        None = 0,
        Descriptor = 1,
        ResponseProtocol = 2,
        LocalImemProtocol = 3,
    };

    struct Inputs {
        bool rst_ni{true};
        bool clear_i{false};

        bool start_valid_i{false};
        std::size_t start_base_address_i{0};
        std::size_t start_word_count_i{0};

        // Canonical boundary controls used by the directed testbench.
        bool imem_request_accept_enable_i{true};
        bool word_accept_enable_i{true};
        bool decoder_word_ready_i{true};

        // Testbench-only corruption at the Fetch response boundary.  It does
        // not alter SRAM contents and is used only by P2.1-F10.
        std::optional<bool> response_last_override_i{};
    };

    struct Ports {
        bool start_valid_i{false};
        bool start_ready_o{false};
        std::size_t start_base_address_i{0};
        std::size_t start_word_count_i{0};
        bool start_fire{false};

        bool imem_read_req_valid_o{false};
        bool imem_read_req_ready_i{false};
        std::size_t imem_read_req_address_o{0};
        bool imem_read_req_last_o{false};
        bool imem_read_req_fire{false};

        bool imem_read_rsp_valid_i{false};
        bool imem_read_rsp_ready_o{false};
        RawWord imem_read_rsp_data_i{};
        bool imem_read_rsp_last_i{false};
        bool imem_read_rsp_fire{false};

        bool word_valid_o{false};
        bool word_ready_i{false};
        RawWord word_o{};
        bool word_last_o{false};
        bool word_fire{false};

        bool decoder_word_valid_i{false};
        bool decoder_word_ready_o{false};
        RawWord decoder_word_i{};
        bool decoder_word_last_i{false};
        bool decoder_word_fire{false};

        bool busy_o{false};
        bool done_o{false};
        bool fault_o{false};
        FaultCode fault_code_o{FaultCode::None};
    };

    struct FifoEntry {
        RawWord data{};
        bool last{false};
        std::size_t address{0};
    };

    void load_word(std::size_t address, const RawWord& word)
    {
        if (address >= ImemDepth)
            throw std::out_of_range("P2.1 local iMEM load address");
        sram_.seed(address, word);
    }

    // Production decoders share their immutable, actual-size local-iMEM
    // image with this finite transport state. Current/Next copies therefore
    // clone only pipeline/FIFO state, never the configured maximum SRAM.
    void bind_image(std::shared_ptr<const std::vector<RawWord>> image)
    {
        if (!image || image->empty() || image->size() > ImemDepth)
            throw std::invalid_argument("invalid MEM local iMEM image view");
        if (descriptor_started_ || busy_ || outstanding_ != 0
            || request_stage_ || response_stage_
            || !response_fifo_.empty() || !word_fifo_.empty())
            throw std::logic_error("cannot rebind active MEM local frontend");
        sram_.seed_image(std::move(image));
    }

    bool memory_valid(std::size_t address) const
    {
        if (address >= ImemDepth)
            throw std::out_of_range("P2.1 local iMEM read address");
        return sram_.initialized(address);
    }

    // Evaluate canonical pre-edge ports and commit the sampled edge.
    void cycle(const Inputs& inputs)
    {
        ports_ = {};
        if (!inputs.rst_ni || inputs.clear_i) {
            typename Sram::Inputs sram_inputs{};
            sram_inputs.rst_ni = false;
            sram_.drive(std::move(sram_inputs));
            sram_.low();
            sram_.validate_posedge();
            sram_.posedge();
            sram_.high();
            reset_pipeline();
            ++physical_cycle_;
            return;
        }

        const bool decoder_pop = !word_fifo_.empty()
            && inputs.decoder_word_ready_i;
        const bool word_fifo_ready = word_fifo_.size() < WordFifoDepth
            || decoder_pop;

        const bool response_valid = !response_fifo_.empty();
        const auto response_entry = response_valid
            ? response_fifo_.front() : FifoEntry{};

        ports_.start_valid_i = inputs.start_valid_i;
        ports_.start_ready_o = !busy_ && !fault_;
        ports_.start_base_address_i = inputs.start_base_address_i;
        ports_.start_word_count_i = inputs.start_word_count_i;
        ports_.start_fire = ports_.start_valid_i && ports_.start_ready_o;

        ports_.imem_read_rsp_valid_i = response_valid;
        ports_.imem_read_rsp_data_i = response_entry.data;
        ports_.imem_read_rsp_last_i = inputs.response_last_override_i
            .value_or(response_entry.last);
        ports_.word_valid_o = response_valid && !fault_;
        ports_.word_ready_i = inputs.word_accept_enable_i
            && word_fifo_ready && !fault_;
        ports_.word_o = response_entry.data;
        ports_.word_last_o = ports_.imem_read_rsp_last_i;
        ports_.imem_read_rsp_ready_o = ports_.word_ready_i && !fault_;
        ports_.imem_read_rsp_fire = ports_.imem_read_rsp_valid_i
            && ports_.imem_read_rsp_ready_o;
        ports_.word_fire = ports_.word_valid_o && ports_.word_ready_i;

        const bool capacity_ready = outstanding_ < ResponseDepth
            || ports_.imem_read_rsp_fire;
        ports_.imem_read_req_valid_o = busy_
            && requests_remaining_ != 0 && request_cooldown_ == 0 && !fault_;
        ports_.imem_read_req_ready_i = inputs.imem_request_accept_enable_i
            && capacity_ready && !fault_;
        ports_.imem_read_req_address_o = next_address_;
        ports_.imem_read_req_last_o = ports_.imem_read_req_valid_o
            && requests_remaining_ == 1;
        ports_.imem_read_req_fire = ports_.imem_read_req_valid_o
            && ports_.imem_read_req_ready_i;

        ports_.decoder_word_valid_i = !word_fifo_.empty();
        ports_.decoder_word_ready_o = inputs.decoder_word_ready_i;
        if (!word_fifo_.empty()) {
            ports_.decoder_word_i = word_fifo_.front().data;
            ports_.decoder_word_last_i = word_fifo_.front().last;
        }
        ports_.decoder_word_fire = ports_.decoder_word_valid_i
            && ports_.decoder_word_ready_o;

        const bool descriptor_error = ports_.start_fire
            && (inputs.start_word_count_i == 0
                || inputs.start_base_address_i >= ImemDepth
                || inputs.start_word_count_i
                    > ImemDepth - inputs.start_base_address_i);
        const bool expected_last = responses_remaining_ == 1;
        const bool response_error = ports_.imem_read_rsp_fire
            && (!busy_ || ports_.imem_read_rsp_last_i != expected_last);

        const bool response_fifo_pop = ports_.imem_read_rsp_fire;
        const bool response_fifo_ready = response_fifo_.size() < ResponseDepth
            || response_fifo_pop;
        std::optional<FifoEntry> arriving_response;
        arriving_response = response_stage_;
        const bool response_fifo_push = arriving_response.has_value()
            && response_fifo_ready;
        const bool local_imem_error = arriving_response.has_value()
            && !response_fifo_ready;

        ports_.busy_o = busy_;
        ports_.done_o = done_;
        ports_.fault_o = fault_ || descriptor_error
            || response_error || local_imem_error;
        ports_.fault_code_o = fault_code_;
        if (descriptor_error)
            ports_.fault_code_o = FaultCode::Descriptor;
        else if (response_error)
            ports_.fault_code_o = FaultCode::ResponseProtocol;
        else if (local_imem_error)
            ports_.fault_code_o = FaultCode::LocalImemProtocol;

        done_ = false;

        if (ports_.start_fire && !descriptor_error) {
            descriptor_started_ = true;
            busy_ = true;
            next_address_ = inputs.start_base_address_i;
            requests_remaining_ = inputs.start_word_count_i;
            responses_remaining_ = inputs.start_word_count_i;
        }

        if (ports_.imem_read_req_fire) {
            ++next_address_;
            --requests_remaining_;
            request_cooldown_ = RequestInitiationInterval;
        }

        if (ports_.imem_read_rsp_fire && responses_remaining_ != 0) {
            --responses_remaining_;
            // RTL terminates the descriptor on the accepted response Last,
            // independently from whether Last matches the remaining count.
            // A mismatch is still reported as a sticky protocol fault below.
            if (ports_.imem_read_rsp_last_i) {
                busy_ = false;
                done_ = true;
            }
        }

        if (descriptor_error) {
            fault_ = true;
            fault_code_ = FaultCode::Descriptor;
        } else if (response_error) {
            fault_ = true;
            fault_code_ = FaultCode::ResponseProtocol;
        } else if (local_imem_error) {
            fault_ = true;
            fault_code_ = FaultCode::LocalImemProtocol;
        }

        if (ports_.decoder_word_fire)
            word_fifo_.pop_front();
        if (ports_.word_fire)
            word_fifo_.push_back(response_entry);

        if (response_fifo_pop)
            response_fifo_.pop_front();
        if (response_fifo_push)
            response_fifo_.push_back(*arriving_response);

        std::optional<FifoEntry> next_response_stage;
        const auto sram_response = sram_.outputs();
        if (sram_response.rvalid_o) {
            if (!request_stage_)
                throw std::logic_error(
                    "local iMEM SRAM response has no request metadata");
            next_response_stage = FifoEntry{
                sram_response.rdata_o, request_stage_->last,
                request_stage_->address};
        }
        std::optional<FifoEntry> next_request_stage;
        if (ports_.imem_read_req_fire) {
            next_request_stage = FifoEntry{
                {}, ports_.imem_read_req_last_o,
                ports_.imem_read_req_address_o};
        }
        request_stage_ = std::move(next_request_stage);
        response_stage_ = std::move(next_response_stage);

        typename Sram::Inputs sram_inputs{};
        sram_inputs.rst_ni = true;
        sram_inputs.cen_i = ports_.imem_read_req_fire;
        sram_inputs.wen_i = false;
        sram_inputs.address_i = ports_.imem_read_req_address_o;
        sram_.drive(std::move(sram_inputs));
        sram_.low();
        sram_.validate_posedge();
        sram_.posedge();
        sram_.high();

        if (ports_.imem_read_req_fire && !ports_.imem_read_rsp_fire)
            ++outstanding_;
        else if (!ports_.imem_read_req_fire && ports_.imem_read_rsp_fire) {
            if (outstanding_ == 0)
                throw std::logic_error("P2.1 outstanding underflow");
            --outstanding_;
        }

        if (request_cooldown_ != 0) --request_cooldown_;

        ++physical_cycle_;
    }

    const Ports& ports() const noexcept { return ports_; }
    std::size_t physical_cycle() const noexcept { return physical_cycle_; }
    bool busy() const noexcept { return busy_; }
    bool done() const noexcept { return done_; }
    bool descriptor_started() const noexcept { return descriptor_started_; }
    bool fault() const noexcept { return fault_; }
    FaultCode fault_code() const noexcept { return fault_code_; }
    std::size_t next_address() const noexcept { return next_address_; }
    std::size_t requests_remaining() const noexcept
    { return requests_remaining_; }
    std::size_t responses_remaining() const noexcept
    { return responses_remaining_; }
    std::size_t outstanding() const noexcept { return outstanding_; }
    bool request_stage_valid() const noexcept
    { return request_stage_.has_value(); }
    bool response_stage_valid() const noexcept
    { return response_stage_.has_value(); }
    std::optional<FifoEntry> request_stage() const noexcept
    { return request_stage_; }
    std::optional<FifoEntry> response_stage() const noexcept
    { return response_stage_; }
    std::size_t response_fifo_level() const noexcept
    { return response_fifo_.size(); }
    std::size_t word_fifo_level() const noexcept
    { return word_fifo_.size(); }
    std::optional<FifoEntry> response_fifo_head() const noexcept
    {
        return response_fifo_.empty()
            ? std::nullopt : std::optional<FifoEntry>(response_fifo_.front());
    }
    std::optional<FifoEntry> word_fifo_head() const noexcept
    {
        return word_fifo_.empty()
            ? std::nullopt : std::optional<FifoEntry>(word_fifo_.front());
    }
    std::size_t payload_words_in_flight() const noexcept
    {
        std::size_t result = 0;
        const auto count = [&](const auto& entry) {
            if (entry.has_value() && entry->address != 0) ++result;
        };
        count(request_stage_);
        count(response_stage_);
        for (const auto& entry : response_fifo_)
            result += entry.address != 0;
        for (const auto& entry : word_fifo_)
            result += entry.address != 0;
        return result;
    }

private:
    void reset_pipeline()
    {
        request_stage_.reset();
        response_stage_.reset();
        response_fifo_.clear();
        word_fifo_.clear();
        busy_ = false;
        done_ = false;
        fault_ = false;
        fault_code_ = FaultCode::None;
        next_address_ = 0;
        requests_remaining_ = 0;
        responses_remaining_ = 0;
        outstanding_ = 0;
        request_cooldown_ = 0;
        descriptor_started_ = false;
        ports_ = {};
    }

    using Sram = timing::Sram1Rw<RawWord, ImemDepth>;
    Sram sram_{};
    std::optional<FifoEntry> request_stage_{};
    std::optional<FifoEntry> response_stage_{};
    std::deque<FifoEntry> response_fifo_{};
    std::deque<FifoEntry> word_fifo_{};

    bool busy_{false};
    bool done_{false};
    bool fault_{false};
    FaultCode fault_code_{FaultCode::None};
    std::size_t next_address_{0};
    std::size_t requests_remaining_{0};
    std::size_t responses_remaining_{0};
    std::size_t outstanding_{0};
    std::size_t request_cooldown_{0};
    bool descriptor_started_{false};
    std::size_t physical_cycle_{0};
    Ports ports_{};
};

template <std::size_t ImemDepth,
          std::size_t ResponseDepth = 4,
          std::size_t WordFifoDepth = 4,
          std::size_t RequestInitiationInterval = 1>
using MemLocalFrontendFetchModel = LocalImemFrontendFetchModel<
    IcuRawImemWord96, ImemDepth, ResponseDepth, WordFifoDepth,
    RequestInitiationInterval>;

template <std::size_t ImemDepth,
          std::size_t ResponseDepth = 4,
          std::size_t WordFifoDepth = 4,
          std::size_t RequestInitiationInterval = 1>
using MxmLocalFrontendFetchModel = LocalImemFrontendFetchModel<
    IcuRawImemWord128, ImemDepth, ResponseDepth, WordFifoDepth,
    RequestInitiationInterval>;

} // namespace ftlpu
