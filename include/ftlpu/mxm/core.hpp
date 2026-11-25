#pragma once

#include "ftlpu/core/timing.hpp"
#include "ftlpu/mxm/bf16_arithmetic.hpp"
#include "ftlpu/mxm/datapath_clocked.hpp"
#include "ftlpu/mxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ftlpu {

// Cycle model for the standalone RTL lpu_mxm_core.  This is intentionally a
// core-boundary model: Macro decode, compiler scheduling and final stream
// formatting are outside this class.  Public signal names mirror the RTL
// ports; *_fire members are derived observability points, not extra RTL pins.
class MxmCore final : public timing::ClockedModule {
public:
    static constexpr std::size_t kRows = 4;
    static constexpr std::size_t kColumns = 4;
    static constexpr std::size_t kTerms = 8;
    static constexpr std::size_t kOutputs = 8;
    static constexpr std::size_t kWeightStreamCount = 32;
    static constexpr std::size_t kQueueDepth = 16;
    static constexpr std::size_t kInstructionInterval = 3;
    static constexpr std::size_t kDotPipelineCycles = 9;
    static constexpr std::size_t kPartialAddCycles = 3;
    static constexpr std::size_t kAccumulatorDepth = 1024;
    static constexpr std::size_t kAccumulatorBankDepth = kAccumulatorDepth / 2;
    static constexpr std::size_t kMatrixN = kColumns * kOutputs;

    static constexpr std::size_t kWeightQStream = 0;
    static constexpr std::size_t kWeightScaleLoStream = 1;
    static constexpr std::size_t kWeightScaleHiStream = 2;
    static constexpr std::size_t kWeightZeroStream = 3;

    using ComputeInstruction = MxmComputeCommand;
    using LoadInstruction = MxmLoadCommand;
    using ParamLoadInstruction = MxmParamLoadCommand;
    using DequantInstruction = MxmDequantCommand;
    using BackendOperation = MxmBackendOperation;
    using BackendInstruction = MxmBackendCommand;

    using Bf16Vector = std::array<std::uint16_t, kTerms>;
    using Fp32Vector = std::array<std::uint32_t, kOutputs>;
    using ActivationRows = std::array<Bf16Vector, kRows>;
    using RowValid = std::array<bool, kRows>;
    using ColumnValid = std::array<bool, kColumns>;
    using RowFp32 = std::array<Fp32Vector, kRows>;
    using ColumnFp32 = std::array<Fp32Vector, kColumns>;
    using WeightStreamValid = std::array<
        std::array<std::array<bool, kTerms>, kWeightStreamCount>, kRows>;
    using WeightStreamData = std::array<
        std::array<std::array<std::uint8_t, kTerms>, kWeightStreamCount>, kRows>;

    struct Inputs {
        bool rst_ni{true};
        bool run_i{false};

        bool compute_enqueue_valid_i{};
        bool compute_launch_i{};
        ComputeInstruction compute_instruction_i{};
        bool load_enqueue_valid_i{};
        bool load_launch_i{};
        LoadInstruction load_instruction_i{};
        bool param_load_enqueue_valid_i{};
        bool param_load_launch_i{};
        ParamLoadInstruction param_load_instruction_i{};
        bool dequant_enqueue_valid_i{};
        bool dequant_launch_i{};
        DequantInstruction dequant_instruction_i{};
        bool backend_enqueue_valid_i{};
        bool backend_launch_i{};
        BackendInstruction backend_instruction_i{};

        RowValid activation_valid_i{};
        ActivationRows activation_i{};
        WeightStreamValid weight_stream_valid_i{};
        WeightStreamData weight_stream_i{};
    };

    struct Outputs {
        bool compute_enqueue_ready_o{};
        bool compute_launch_ready_o{};
        bool load_enqueue_ready_o{};
        bool load_launch_ready_o{};
        bool param_load_enqueue_ready_o{};
        bool param_load_launch_ready_o{};
        bool dequant_enqueue_ready_o{};
        bool dequant_launch_ready_o{};
        bool backend_enqueue_ready_o{};
        bool backend_launch_ready_o{};

        bool compute_enqueue_fire{};
        bool compute_launch_fire{};
        bool load_enqueue_fire{};
        bool load_launch_fire{};
        bool param_load_enqueue_fire{};
        bool param_load_launch_fire{};
        bool dequant_enqueue_fire{};
        bool dequant_launch_fire{};
        bool backend_enqueue_fire{};
        bool backend_launch_fire{};

        RowValid compute_row_valid_o{};
        RowValid load_row_valid_o{};
        RowValid param_load_row_valid_o{};
        RowValid dequant_row_valid_o{};
        std::array<ComputeInstruction, kRows> compute_row_instruction_o{};
        std::array<LoadInstruction, kRows> load_row_instruction_o{};
        std::array<ParamLoadInstruction, kRows> param_load_row_instruction_o{};
        std::array<DequantInstruction, kRows> dequant_row_instruction_o{};
        std::array<std::uint16_t, kRows> compute_row_tag_o{};
        std::array<std::uint16_t, kRows> load_row_index_o{};

        WeightStreamValid weight_stream_consume_o{};
        std::array<std::array<bool, kColumns>, kRows> cell_weight_valid_o{};
        std::array<std::array<bool, kColumns>, kRows> cell_weight_buffer_o{};
        std::array<std::array<bool, kColumns>, kRows> cell_activation_valid_o{};
        std::array<std::array<bool, kColumns>, kRows> cell_activation_buffer_o{};

        ColumnValid partial_valid_o{};
        ColumnFp32 partial_o{};
        std::array<std::uint16_t, kColumns> partial_tag_o{};
        RowValid returned_valid_o{};
        RowFp32 returned_o{};
        std::array<std::uint16_t, kRows> returned_tag_o{};

        ColumnValid backend_issue_valid_o{};
        ColumnValid backend_issue_bf16_o{};
        std::array<BackendOperation, kColumns> backend_issue_operation_o{};
        std::array<std::uint16_t, kColumns> backend_issue_address_o{};
        ColumnValid backend_slot_valid_o{};
        ColumnValid backend_slot_bf16_o{};
        std::array<BackendOperation, kColumns> backend_slot_operation_o{};
        std::array<std::uint16_t, kColumns> backend_slot_address_o{};
        ColumnValid alu_valid_o{};
        ColumnFp32 operand_a_o{};
        ColumnFp32 operand_b_o{};
        std::array<std::uint16_t, kColumns> alu_address_o{};
        std::array<std::uint16_t, kColumns> alu_tag_o{};
        std::array<std::array<bool, 2>, kColumns> bank_request_valid_o{};
        std::array<std::array<bool, 2>, kColumns> bank_write_o{};
        std::array<std::array<std::uint16_t, 2>, kColumns> bank_address_o{};

        ColumnValid result_valid_o{};
        ColumnValid result_bf16_o{};
        ColumnFp32 result_o{};
        std::array<std::uint16_t, kColumns> result_address_o{};
        std::array<std::uint16_t, kColumns> result_tag_o{};

        bool frontend_fault_o{};
        bool backend_fault_o{};
        bool fault_o{};
    };

    MxmCore();
    ~MxmCore() override;

    MxmCore(const MxmCore&) = delete;
    MxmCore& operator=(const MxmCore&) = delete;

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }
    [[nodiscard]] const Outputs& outputs() const noexcept { return observed_; }
    [[nodiscard]] const Outputs& pre_posedge_outputs() const noexcept
    { return pre_posedge_observed_; }

    // LOW samples public inputs and prepares every registered Next value.
    void low() override
    {
        state_->prepare();
        pending_weight_write_count_ = 0;
        pending_acc_write_count_ = 0;
        clear_weight_valid_ = false;
        clear_accumulator_valid_ = false;

        const auto& current = state_->current();
        observed_ = evaluate(current);
        pre_posedge_observed_ = observed_;
        auto& next = state_->next_value();

        if (!inputs_.rst_ni) {
            next = State{};
            clear_weight_valid_ = true;
            clear_accumulator_valid_ = true;
            return;
        }

        const bool compute_direct = current.compute_queue.count == 0
            && observed_.compute_enqueue_fire
            && observed_.compute_launch_fire;
        const bool load_direct = current.load_queue.count == 0
            && observed_.load_enqueue_fire && observed_.load_launch_fire;
        const bool param_direct = current.param_queue.count == 0
            && observed_.param_load_enqueue_fire
            && observed_.param_load_launch_fire;
        const bool dequant_direct = current.dequant_queue.count == 0
            && observed_.dequant_enqueue_fire
            && observed_.dequant_launch_fire;
        const bool backend_direct = current.backend_queue.count == 0
            && observed_.backend_enqueue_fire
            && observed_.backend_launch_fire;

        update_queue(next.compute_queue, current.compute_queue,
            observed_.compute_enqueue_fire && !compute_direct,
            inputs_.compute_instruction_i,
            observed_.compute_launch_fire && current.compute_queue.count != 0);
        update_queue(next.load_queue, current.load_queue,
            observed_.load_enqueue_fire && !load_direct,
            inputs_.load_instruction_i,
            observed_.load_launch_fire && current.load_queue.count != 0);
        update_queue(next.param_queue, current.param_queue,
            observed_.param_load_enqueue_fire && !param_direct,
            inputs_.param_load_instruction_i,
            observed_.param_load_launch_fire && current.param_queue.count != 0);
        update_queue(next.dequant_queue, current.dequant_queue,
            observed_.dequant_enqueue_fire && !dequant_direct,
            inputs_.dequant_instruction_i,
            observed_.dequant_launch_fire && current.dequant_queue.count != 0);
        update_queue(next.backend_queue, current.backend_queue,
            observed_.backend_enqueue_fire && !backend_direct,
            inputs_.backend_instruction_i,
            observed_.backend_launch_fire && current.backend_queue.count != 0);

        if (!inputs_.run_i) {
            clear_runtime_state(next);
            clear_weight_valid_ = true;
            return;
        }

        const auto compute_launch_instruction = current.compute_queue.count != 0
            ? queue_head(current.compute_queue) : inputs_.compute_instruction_i;
        const auto load_launch_instruction = current.load_queue.count != 0
            ? queue_head(current.load_queue) : inputs_.load_instruction_i;
        const auto param_launch_instruction = current.param_queue.count != 0
            ? queue_head(current.param_queue) : inputs_.param_load_instruction_i;
        const auto dequant_launch_instruction = current.dequant_queue.count != 0
            ? queue_head(current.dequant_queue) : inputs_.dequant_instruction_i;
        const auto backend_head = current.backend_queue.count != 0
            ? queue_head(current.backend_queue) : inputs_.backend_instruction_i;
        advance_wave(next.compute_wave, current.compute_wave,
            observed_.compute_launch_fire,
            ComputeWavePayload{compute_launch_instruction, current.next_tag});
        advance_wave(next.load_wave, current.load_wave,
            observed_.load_launch_fire, load_launch_instruction);
        advance_wave(next.param_wave, current.param_wave,
            observed_.param_load_launch_fire, param_launch_instruction);
        advance_wave(next.dequant_wave, current.dequant_wave,
            observed_.dequant_launch_fire, dequant_launch_instruction);
        advance_wave(next.backend_wave, current.backend_wave,
            observed_.backend_launch_fire && backend_legal(backend_head),
            backend_head);

        if (observed_.compute_launch_fire) ++next.next_tag;
        update_load_counters(next, current);
        update_dequant(next, current);
        update_cells(next, current);
        update_result_return(next, current);
        update_backend(next, current);

        const bool frontend_bad_launch =
            (inputs_.compute_launch_i && !observed_.compute_launch_ready_o)
            || (inputs_.load_launch_i && !observed_.load_launch_ready_o)
            || (inputs_.param_load_launch_i && !observed_.param_load_launch_ready_o)
            || (inputs_.dequant_launch_i && !observed_.dequant_launch_ready_o);
        next.frontend_fault = current.frontend_fault || frontend_bad_launch
            || cycle_frontend_fault_;
        const bool backend_bad_launch = inputs_.backend_launch_i
            && (!observed_.backend_launch_ready_o || !backend_legal(backend_head));
        next.backend_fault = current.backend_fault || backend_bad_launch
            || cycle_backend_fault_;
    }

    void validate_posedge() const override
    {
        state_->validate_commit();
        if (pending_weight_write_count_ > pending_weight_writes_.size()
            || pending_acc_write_count_ > pending_acc_writes_.size()) {
            throw std::logic_error("MXM sparse commit buffer overflow");
        }
    }

    // POSEDGE performs only the common state commit plus already prepared
    // sparse memory writes.  No combinational decision is made here.
    void posedge() override
    {
        state_->commit();
        if (clear_weight_valid_) {
            for (auto& row : storage_->weight_valid)
                for (auto& column : row)
                    for (auto& buffer : column) buffer.fill(false);
        }
        if (clear_accumulator_valid_) {
            for (auto& column : storage_->accumulator_valid)
                for (auto& bank : column) bank.fill(false);
        }
        for (std::size_t index = 0; index < pending_weight_write_count_; ++index) {
            const auto& write = pending_weight_writes_[index];
            storage_->weights[write.row][write.column][write.buffer][write.group]
                = write.data;
            storage_->weight_valid[write.row][write.column][write.buffer][write.group]
                = write.valid;
        }
        for (std::size_t index = 0; index < pending_acc_write_count_; ++index) {
            const auto& write = pending_acc_writes_[index];
            storage_->accumulator[write.column][write.bank][write.row] = write.data;
            storage_->accumulator_valid[write.column][write.bank][write.row] = true;
        }
        pending_weight_write_count_ = 0;
        pending_acc_write_count_ = 0;
        clear_weight_valid_ = false;
        clear_accumulator_valid_ = false;
    }

    // HIGH is observation-only.  It makes the post-commit Current visible at
    // the canonical interface without advancing any state.
    void high() override { observed_ = evaluate(state_->current()); }
    void negedge() override {}

    [[nodiscard]] std::uint64_t cycle() const noexcept
    { return state_->commit_count(); }
    void reset()
    {
        state_ = std::make_unique<timing::ClockedState<State>>();
        storage_ = std::make_unique<BulkStorage>();
        inputs_ = {};
        observed_ = {};
        pre_posedge_observed_ = {};
        pending_weight_write_count_ = 0;
        pending_acc_write_count_ = 0;
        clear_weight_valid_ = false;
        clear_accumulator_valid_ = false;
        cycle_frontend_fault_ = false;
        cycle_backend_fault_ = false;
    }
    [[nodiscard]] std::size_t compute_queue_level() const noexcept
    { return state_->current().compute_queue.count; }
    [[nodiscard]] std::size_t load_queue_level() const noexcept
    { return state_->current().load_queue.count; }
    [[nodiscard]] std::size_t param_load_queue_level() const noexcept
    { return state_->current().param_queue.count; }
    [[nodiscard]] std::size_t dequant_queue_level() const noexcept
    { return state_->current().dequant_queue.count; }
    [[nodiscard]] std::size_t backend_queue_level() const noexcept
    { return state_->current().backend_queue.count; }
    [[nodiscard]] bool idle() const noexcept
    { return state_is_idle(state_->current()); }
    [[nodiscard]] bool accumulator_initialized(
        std::size_t column, std::size_t address) const
    {
        check_column_address(column, address);
        return storage_->accumulator_valid[column][address & 1u][address >> 1u];
    }

private:
    template <typename T>
    struct QueueState {
        std::array<T, kQueueDepth> memory{};
        std::size_t read{};
        std::size_t write{};
        std::size_t count{};
    };

    template <typename T, std::size_t Stages>
    struct WaveState {
        std::array<bool, Stages> valid{};
        std::array<T, Stages> payload{};
    };

    struct ComputeWavePayload {
        ComputeInstruction instruction{};
        std::uint16_t tag{};
    };
    static constexpr std::size_t kFrontendWaveStages =
        1 + (kRows - 1) * kInstructionInterval;
    static constexpr std::size_t kBackendWaveStages = kColumns;

    struct DequantToken {
        bool valid{};
        bool buffer{};
        Bf16Vector data{};
    };
    struct ActivationToken {
        bool valid{};
        bool buffer{};
        std::uint16_t tag{};
        Bf16Vector data{};
    };
    struct Fp32Token {
        bool valid{};
        std::uint16_t tag{};
        Fp32Vector data{};
    };
    struct BackendToken {
        bool valid{};
        bool ok{};
        bool bf16{};
        BackendOperation operation{BackendOperation::DirectOut};
        std::uint16_t address{};
        std::uint16_t tag{};
        Fp32Vector data{};
    };
    struct BankState {
        bool capture_valid{};
        Fp32Vector capture_data{};
        std::uint16_t capture_address{};
        bool read_valid{};
        Fp32Vector read_data{};
        std::uint16_t read_address{};
    };
    struct BackendColumnState {
        std::array<BackendToken, 2> context{};
        std::array<BackendToken, 3> bypass{};
        std::array<BackendToken, 3> sum{};
        std::array<bool, 5> pending_valid{};
        std::array<std::uint16_t, 5> pending_address{};
        std::array<BankState, 2> banks{};
        BackendToken output{};
    };
    struct ParameterState {
        std::array<bool, 2> ready{};
        std::array<std::array<std::uint16_t, kTerms>, 2> scale{};
        std::array<std::array<std::int8_t, kTerms>, 2> zero{};
    };
    struct CellState {
        std::array<Fp32Token, kDotPipelineCycles> dot{};
        std::array<Fp32Token, kPartialAddCycles> partial{};
    };
    struct ReturnState {
        std::array<std::array<Fp32Token, kColumns>, kRows> down{};
        std::array<std::array<Fp32Token, kColumns>, kRows> left{};
    };

    struct State {
        QueueState<ComputeInstruction> compute_queue{};
        QueueState<LoadInstruction> load_queue{};
        QueueState<ParamLoadInstruction> param_queue{};
        QueueState<DequantInstruction> dequant_queue{};
        QueueState<BackendInstruction> backend_queue{};
        WaveState<ComputeWavePayload, kFrontendWaveStages> compute_wave{};
        WaveState<LoadInstruction, kFrontendWaveStages> load_wave{};
        WaveState<ParamLoadInstruction, kFrontendWaveStages> param_wave{};
        WaveState<DequantInstruction, kFrontendWaveStages> dequant_wave{};
        WaveState<BackendInstruction, kBackendWaveStages> backend_wave{};
        std::uint16_t next_tag{};
        std::array<std::uint8_t, kRows> load_remaining{};
        std::array<std::uint16_t, kRows> load_index{};
        std::array<ParameterState, kRows> parameters{};
        std::array<std::array<DequantToken, 2>, kRows> dequant{};
        std::array<std::array<ActivationToken, kColumns>, kRows> activation_link{};
        std::array<std::array<DequantToken, kColumns>, kRows> weight_link{};
        std::array<std::array<CellState, kColumns>, kRows> cells{};
        ReturnState result_return{};
        std::array<BackendColumnState, kColumns> backend{};
        bool frontend_fault{};
        bool backend_fault{};
    };

public:
    // Combinational Scheduler -> native command-port contract.  These helpers
    // inspect only committed Current state; they do not advance the Core or
    // bypass the common LOW/POSEDGE barrier.
    [[nodiscard]] bool command_ready(
        bool run_i, const ComputeInstruction&) const noexcept
    {
        return run_i;
    }

    [[nodiscard]] bool command_ready(
        bool run_i, const ParamLoadInstruction&) const noexcept
    {
        return run_i;
    }

    [[nodiscard]] bool command_ready(
        bool run_i, const DequantInstruction&) const noexcept
    {
        return run_i;
    }

    [[nodiscard]] bool command_ready(
        bool run_i, const LoadInstruction&) const noexcept
    {
        const auto& state = state_->current();
        const bool load_arrival0 = state.load_wave.valid[0];
        const bool load_busy = load_arrival0
            ? (!state.load_wave.payload[0].column_mode && kMatrixN > 1)
            : state.load_remaining[0] > 1;
        return run_i && !load_busy;
    }

    [[nodiscard]] bool command_ready(
        bool run_i, const BackendInstruction& candidate) const noexcept
    {
        const auto& queue = state_->current().backend_queue;
        const auto& launch = queue.count != 0 ? queue_head(queue) : candidate;
        return run_i && backend_legal(launch);
    }

private:

    template <typename Payload, std::size_t Stages>
    static bool wave_idle(const WaveState<Payload, Stages>& wave) noexcept
    {
        for (const bool valid : wave.valid)
            if (valid) return false;
        return true;
    }

    static bool state_is_idle(const State& state) noexcept
    {
        if (state.compute_queue.count != 0 || state.load_queue.count != 0
            || state.param_queue.count != 0 || state.dequant_queue.count != 0
            || state.backend_queue.count != 0
            || !wave_idle(state.compute_wave) || !wave_idle(state.load_wave)
            || !wave_idle(state.param_wave) || !wave_idle(state.dequant_wave)
            || !wave_idle(state.backend_wave))
            return false;
        for (const auto remaining : state.load_remaining)
            if (remaining != 0) return false;
        for (const auto& row : state.dequant)
            for (const auto& token : row)
                if (token.valid) return false;
        for (const auto& row : state.activation_link)
            for (const auto& token : row)
                if (token.valid) return false;
        for (const auto& row : state.weight_link)
            for (const auto& token : row)
                if (token.valid) return false;
        for (const auto& row : state.cells)
            for (const auto& cell : row) {
                for (const auto& token : cell.dot)
                    if (token.valid) return false;
                for (const auto& token : cell.partial)
                    if (token.valid) return false;
            }
        for (const auto& row : state.result_return.down)
            for (const auto& token : row)
                if (token.valid) return false;
        for (const auto& row : state.result_return.left)
            for (const auto& token : row)
                if (token.valid) return false;
        for (const auto& column : state.backend) {
            for (const auto& token : column.context)
                if (token.valid) return false;
            for (const auto& token : column.bypass)
                if (token.valid) return false;
            for (const auto& token : column.sum)
                if (token.valid) return false;
            for (const auto valid : column.pending_valid)
                if (valid) return false;
            for (const auto& bank : column.banks)
                if (bank.capture_valid || bank.read_valid) return false;
            if (column.output.valid) return false;
        }
        return true;
    }

    using WeightGroup = Bf16Vector;
    using WeightGroups = std::array<WeightGroup, kOutputs>;
    using WeightBuffers = std::array<WeightGroups, 2>;
    using WeightColumns = std::array<WeightBuffers, kColumns>;
    using WeightRows = std::array<WeightColumns, kRows>;
    using WeightValidBuffers = std::array<std::array<bool, kOutputs>, 2>;
    using WeightValidColumns = std::array<WeightValidBuffers, kColumns>;
    using WeightValidRows = std::array<WeightValidColumns, kRows>;
    using AccumulatorRows = std::array<Fp32Vector, kAccumulatorBankDepth>;
    using AccumulatorBanks = std::array<AccumulatorRows, 2>;
    using Accumulator = std::array<AccumulatorBanks, kColumns>;
    using AccumulatorValidRows = std::array<bool, kAccumulatorBankDepth>;
    using AccumulatorValidBanks = std::array<AccumulatorValidRows, 2>;
    using AccumulatorValid = std::array<AccumulatorValidBanks, kColumns>;
    struct BulkStorage {
        WeightRows weights{};
        WeightValidRows weight_valid{};
        Accumulator accumulator{};
        AccumulatorValid accumulator_valid{};
    };

    struct WeightWrite {
        std::size_t row{};
        std::size_t column{};
        std::size_t buffer{};
        std::size_t group{};
        Bf16Vector data{};
        bool valid{};
    };
    struct AccumulatorWrite {
        std::size_t column{};
        std::size_t bank{};
        std::size_t row{};
        Fp32Vector data{};
    };

    template <typename T>
    static const T& queue_head(const QueueState<T>& queue)
    {
        return queue.memory[queue.read];
    }

    template <typename T>
    static void update_queue(QueueState<T>& next, const QueueState<T>& current,
        bool push, const T& data, bool pop)
    {
        next = current;
        if (push) {
            next.memory[current.write] = data;
            next.write = (current.write + 1) % kQueueDepth;
        }
        if (pop) next.read = (current.read + 1) % kQueueDepth;
        if (push && !pop) ++next.count;
        if (pop && !push) --next.count;
    }

    template <typename T, std::size_t Stages>
    static void advance_wave(WaveState<T, Stages>& next,
        const WaveState<T, Stages>& current, bool launch, const T& payload)
    {
        next.valid[0] = launch;
        if (launch) next.payload[0] = payload;
        for (std::size_t stage = 1; stage < Stages; ++stage) {
            next.valid[stage] = current.valid[stage - 1];
            if (current.valid[stage - 1])
                next.payload[stage] = current.payload[stage - 1];
        }
    }

    static bool backend_legal(const BackendInstruction& instruction) noexcept
    {
        const auto opcode = static_cast<unsigned>(instruction.opcode);
        return instruction.address < kAccumulatorDepth && opcode < 4
            && (!instruction.out_bf16
                || instruction.opcode == BackendOperation::DirectOut
                || instruction.opcode == BackendOperation::AccOut);
    }

    Outputs evaluate(const State& state) const
    {
        Outputs out{};
        const bool compute_direct = inputs_.compute_launch_i
            && inputs_.compute_enqueue_valid_i
            && state.compute_queue.count == 0;
        const auto load_arrival0 = state.load_wave.valid[0];
        const auto load_busy = load_arrival0
            ? (!state.load_wave.payload[0].column_mode && kMatrixN > 1)
            : state.load_remaining[0] > 1;
        const bool load_direct = inputs_.load_launch_i
            && inputs_.load_enqueue_valid_i && state.load_queue.count == 0;
        const bool param_direct = inputs_.param_load_launch_i
            && inputs_.param_load_enqueue_valid_i
            && state.param_queue.count == 0;
        const bool dequant_direct = inputs_.dequant_launch_i
            && inputs_.dequant_enqueue_valid_i
            && state.dequant_queue.count == 0;
        const bool backend_direct = inputs_.backend_launch_i
            && inputs_.backend_enqueue_valid_i
            && state.backend_queue.count == 0;

        const auto compute_launch_instruction = state.compute_queue.count != 0
            ? queue_head(state.compute_queue) : inputs_.compute_instruction_i;
        const auto load_launch_instruction = state.load_queue.count != 0
            ? queue_head(state.load_queue) : inputs_.load_instruction_i;
        const auto param_launch_instruction = state.param_queue.count != 0
            ? queue_head(state.param_queue) : inputs_.param_load_instruction_i;
        const auto dequant_launch_instruction = state.dequant_queue.count != 0
            ? queue_head(state.dequant_queue) : inputs_.dequant_instruction_i;
        const auto backend_launch_instruction = state.backend_queue.count != 0
            ? queue_head(state.backend_queue) : inputs_.backend_instruction_i;
        static_cast<void>(compute_launch_instruction);
        static_cast<void>(load_launch_instruction);
        static_cast<void>(param_launch_instruction);
        static_cast<void>(dequant_launch_instruction);

        out.compute_launch_ready_o = inputs_.run_i
            && (state.compute_queue.count != 0
                || inputs_.compute_enqueue_valid_i);
        out.load_launch_ready_o = inputs_.run_i && !load_busy
            && (state.load_queue.count != 0 || inputs_.load_enqueue_valid_i);
        out.param_load_launch_ready_o = inputs_.run_i
            && (state.param_queue.count != 0
                || inputs_.param_load_enqueue_valid_i);
        out.dequant_launch_ready_o = inputs_.run_i
            && (state.dequant_queue.count != 0
                || inputs_.dequant_enqueue_valid_i);
        out.backend_launch_ready_o = inputs_.run_i
            && (state.backend_queue.count != 0
                || inputs_.backend_enqueue_valid_i)
            && backend_legal(backend_launch_instruction);
        const bool compute_pop = inputs_.compute_launch_i
            && out.compute_launch_ready_o && state.compute_queue.count != 0;
        const bool load_pop = inputs_.load_launch_i
            && out.load_launch_ready_o && state.load_queue.count != 0;
        const bool param_pop = inputs_.param_load_launch_i
            && out.param_load_launch_ready_o && state.param_queue.count != 0;
        const bool dequant_pop = inputs_.dequant_launch_i
            && out.dequant_launch_ready_o && state.dequant_queue.count != 0;
        const bool backend_pop = inputs_.backend_launch_i
            && out.backend_launch_ready_o && state.backend_queue.count != 0;
        const bool compute_queue_ready = state.compute_queue.count < kQueueDepth
            || compute_pop;
        const bool load_queue_ready = state.load_queue.count < kQueueDepth
            || load_pop;
        const bool param_queue_ready = state.param_queue.count < kQueueDepth
            || param_pop;
        const bool dequant_queue_ready = state.dequant_queue.count < kQueueDepth
            || dequant_pop;
        const bool backend_queue_ready = state.backend_queue.count < kQueueDepth
            || backend_pop;
        out.compute_enqueue_ready_o = compute_direct
            ? out.compute_launch_ready_o : compute_queue_ready;
        out.load_enqueue_ready_o = load_direct
            ? out.load_launch_ready_o : load_queue_ready;
        out.param_load_enqueue_ready_o = param_direct
            ? out.param_load_launch_ready_o : param_queue_ready;
        out.dequant_enqueue_ready_o = dequant_direct
            ? out.dequant_launch_ready_o : dequant_queue_ready;
        out.backend_enqueue_ready_o = backend_direct
            ? out.backend_launch_ready_o : backend_queue_ready;
        out.compute_enqueue_fire = inputs_.compute_enqueue_valid_i && out.compute_enqueue_ready_o;
        out.compute_launch_fire = inputs_.compute_launch_i && out.compute_launch_ready_o;
        out.load_enqueue_fire = inputs_.load_enqueue_valid_i && out.load_enqueue_ready_o;
        out.load_launch_fire = inputs_.load_launch_i && out.load_launch_ready_o;
        out.param_load_enqueue_fire = inputs_.param_load_enqueue_valid_i && out.param_load_enqueue_ready_o;
        out.param_load_launch_fire = inputs_.param_load_launch_i && out.param_load_launch_ready_o;
        out.dequant_enqueue_fire = inputs_.dequant_enqueue_valid_i && out.dequant_enqueue_ready_o;
        out.dequant_launch_fire = inputs_.dequant_launch_i && out.dequant_launch_ready_o;
        out.backend_enqueue_fire = inputs_.backend_enqueue_valid_i && out.backend_enqueue_ready_o;
        out.backend_launch_fire = inputs_.backend_launch_i && out.backend_launch_ready_o;

        for (std::size_t row = 0; row < kRows; ++row) {
            const auto wave_stage = row * kInstructionInterval;
            out.compute_row_valid_o[row] = inputs_.run_i
                && state.compute_wave.valid[wave_stage];
            out.compute_row_instruction_o[row]
                = state.compute_wave.payload[wave_stage].instruction;
            out.compute_row_tag_o[row] = state.compute_wave.payload[wave_stage].tag;
            const bool load_arrival = inputs_.run_i && state.load_wave.valid[wave_stage];
            out.load_row_valid_o[row] = inputs_.run_i
                && (load_arrival || state.load_remaining[row] != 0);
            out.load_row_instruction_o[row] = state.load_wave.payload[wave_stage];
            out.load_row_index_o[row] = load_arrival ? 0 : state.load_index[row];
            out.param_load_row_valid_o[row] = inputs_.run_i
                && state.param_wave.valid[wave_stage];
            out.param_load_row_instruction_o[row] = state.param_wave.payload[wave_stage];
            out.dequant_row_valid_o[row] = inputs_.run_i
                && state.dequant_wave.valid[wave_stage];
            out.dequant_row_instruction_o[row] = state.dequant_wave.payload[wave_stage];

            bool param_available = true;
            bool weight_available = true;
            for (std::size_t lane = 0; lane < kTerms; ++lane) {
                weight_available = weight_available
                    && inputs_.weight_stream_valid_i[row][kWeightQStream][lane];
                param_available = param_available
                    && inputs_.weight_stream_valid_i[row][kWeightScaleLoStream][lane]
                    && inputs_.weight_stream_valid_i[row][kWeightScaleHiStream][lane]
                    && inputs_.weight_stream_valid_i[row][kWeightZeroStream][lane];
            }
            const auto parameter_buffer = out.param_load_row_instruction_o[row].param_buffer;
            const auto dequant_buffer = out.dequant_row_instruction_o[row].param_buffer;
            const bool collision = out.load_row_valid_o[row]
                && out.dequant_row_valid_o[row]
                && out.param_load_row_valid_o[row]
                && dequant_buffer == parameter_buffer;
            const bool param_accepted = out.param_load_row_valid_o[row]
                && param_available && !collision;
            const bool weight_accepted = out.load_row_valid_o[row]
                && out.dequant_row_valid_o[row] && weight_available
                && state.parameters[row].ready[dequant_buffer] && !collision;
            if (param_accepted) {
                for (std::size_t lane = 0; lane < kTerms; ++lane) {
                    out.weight_stream_consume_o[row][kWeightScaleLoStream][lane] = true;
                    out.weight_stream_consume_o[row][kWeightScaleHiStream][lane] = true;
                    out.weight_stream_consume_o[row][kWeightZeroStream][lane] = true;
                }
            }
            if (weight_accepted) {
                for (std::size_t lane = 0; lane < kTerms; ++lane)
                    out.weight_stream_consume_o[row][kWeightQStream][lane] = true;
            }

            ActivationToken activation = activation_input(state, out, row, 0);
            DequantToken weight = weight_input(state, row, 0);
            for (std::size_t column = 0; column < kColumns; ++column) {
                if (column != 0) {
                    activation = state.activation_link[row][column - 1];
                    weight = state.weight_link[row][column - 1];
                }
                out.cell_activation_valid_o[row][column] = activation.valid;
                out.cell_activation_buffer_o[row][column] = activation.buffer;
                const bool weight_busy = activation.valid
                    && activation.buffer == weight.buffer;
                out.cell_weight_valid_o[row][column] = weight.valid && !weight_busy;
                out.cell_weight_buffer_o[row][column] = weight.buffer;
            }
        }

        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto& partial = state.cells[kRows - 1][column].partial.back();
            out.partial_valid_o[column] = inputs_.run_i && partial.valid;
            out.partial_o[column] = partial.data;
            out.partial_tag_o[column] = partial.tag;
        }
        for (std::size_t row = 0; row < kRows; ++row) {
            const auto& returned = state.result_return.left[row][0];
            out.returned_valid_o[row] = inputs_.run_i && returned.valid;
            out.returned_o[row] = returned.data;
            out.returned_tag_o[row] = returned.tag;
        }
        evaluate_backend_outputs(state, out);
        out.frontend_fault_o = state.frontend_fault;
        out.backend_fault_o = state.backend_fault;
        out.fault_o = state.frontend_fault || state.backend_fault;
        return out;
    }

    ActivationToken activation_input(const State&, const Outputs& out,
        std::size_t row, std::size_t column) const
    {
        if (column != 0) return {};
        return ActivationToken{
            out.compute_row_valid_o[row] && inputs_.activation_valid_i[row],
            out.compute_row_instruction_o[row].weight_buffer,
            out.compute_row_tag_o[row], inputs_.activation_i[row]};
    }

    static DequantToken weight_input(const State& state,
        std::size_t row, std::size_t column)
    {
        return column == 0 ? state.dequant[row].back() : DequantToken{};
    }

    void update_load_counters(State& next, const State& current)
    {
        for (std::size_t row = 0; row < kRows; ++row) {
            const auto stage = row * kInstructionInterval;
            if (current.load_wave.valid[stage]) {
                const auto& instruction = current.load_wave.payload[stage];
                next.load_remaining[row] = instruction.column_mode
                    ? 0 : static_cast<std::uint8_t>(kMatrixN - 1);
                next.load_index[row] = 1;
            } else if (current.load_remaining[row] != 0) {
                next.load_remaining[row] = current.load_remaining[row] - 1;
                next.load_index[row] = current.load_index[row] + 1;
            }
        }
    }

    void update_dequant(State& next, const State& current)
    {
        cycle_frontend_fault_ = false;
        for (std::size_t row = 0; row < kRows; ++row) {
            const auto& out = observed_;
            bool parameter_available = true;
            bool weight_available = true;
            for (std::size_t lane = 0; lane < kTerms; ++lane) {
                weight_available = weight_available
                    && inputs_.weight_stream_valid_i[row][kWeightQStream][lane];
                parameter_available = parameter_available
                    && inputs_.weight_stream_valid_i[row][kWeightScaleLoStream][lane]
                    && inputs_.weight_stream_valid_i[row][kWeightScaleHiStream][lane]
                    && inputs_.weight_stream_valid_i[row][kWeightZeroStream][lane];
            }
            const bool parameter_request = out.param_load_row_valid_o[row];
            const bool weight_request = out.load_row_valid_o[row]
                && out.dequant_row_valid_o[row];
            const auto parameter_buffer = out.param_load_row_instruction_o[row].param_buffer;
            const auto dequant_buffer = out.dequant_row_instruction_o[row].param_buffer;
            const bool collision = parameter_request && weight_request
                && parameter_buffer == dequant_buffer;
            const bool parameter_accepted = parameter_request
                && parameter_available && !collision;
            const bool weight_accepted = weight_request && weight_available
                && current.parameters[row].ready[dequant_buffer] && !collision;

            if (parameter_accepted) {
                next.parameters[row].ready[parameter_buffer] = true;
                for (std::size_t lane = 0; lane < kTerms; ++lane) {
                    next.parameters[row].scale[parameter_buffer][lane]
                        = static_cast<std::uint16_t>(
                            (std::uint16_t(inputs_.weight_stream_i[row][kWeightScaleHiStream][lane]) << 8)
                            | inputs_.weight_stream_i[row][kWeightScaleLoStream][lane]);
                    next.parameters[row].zero[parameter_buffer][lane]
                        = static_cast<std::int8_t>(
                            inputs_.weight_stream_i[row][kWeightZeroStream][lane]);
                }
            }
            DequantToken d1{};
            d1.valid = weight_accepted;
            d1.buffer = out.load_row_instruction_o[row].weight_buffer;
            if (weight_accepted) {
                for (std::size_t lane = 0; lane < kTerms; ++lane) {
                    d1.data[lane] = mxm_numeric::int8_dequant_bf16(
                        static_cast<std::int8_t>(
                            inputs_.weight_stream_i[row][kWeightQStream][lane]),
                        current.parameters[row].zero[dequant_buffer][lane],
                        current.parameters[row].scale[dequant_buffer][lane]);
                }
            }
            mxm_timing::shift_pipeline(next.dequant[row], current.dequant[row], d1);
            cycle_frontend_fault_ = cycle_frontend_fault_
                || (parameter_request && !parameter_accepted)
                || (weight_request && !weight_accepted);
        }
    }

    void update_cells(State& next, const State& current)
    {
        for (std::size_t row = 0; row < kRows; ++row) {
            ActivationToken activation = activation_input(current, observed_, row, 0);
            DequantToken weight = weight_input(current, row, 0);
            for (std::size_t column = 0; column < kColumns; ++column) {
                if (column != 0) {
                    activation = current.activation_link[row][column - 1];
                    weight = current.weight_link[row][column - 1];
                }
                next.activation_link[row][column] = activation;
                next.weight_link[row][column] = {};

                const bool busy = activation.valid && activation.buffer == weight.buffer;
                const bool weight_accept = weight.valid && !busy;
                if (weight.valid && !weight_accept) cycle_frontend_fault_ = true;
                if (weight_accept) {
                    const auto buffer = static_cast<std::size_t>(weight.buffer);
                    const auto shifted = mxm_timing::shift_weight_buffer(
                        storage_->weights[row][column][buffer],
                        storage_->weight_valid[row][column][buffer], weight.data);
                    if (column + 1 < kColumns && shifted.forward_valid) {
                        next.weight_link[row][column] = DequantToken{
                            true, weight.buffer, shifted.forward_data};
                    }
                    for (std::size_t group = 0; group < kOutputs; ++group)
                        stage_weight_write(row, column, buffer, group,
                            shifted.data[group], shifted.valid[group]);
                }

                Fp32Token dot_input{};
                dot_input.valid = activation.valid;
                dot_input.tag = activation.tag;
                if (activation.valid) {
                    const auto buffer = static_cast<std::size_t>(activation.buffer);
                    for (std::size_t group = 0; group < kOutputs; ++group) {
                        if (!storage_->weight_valid[row][column][buffer][group]) {
                            dot_input.valid = false;
                            cycle_frontend_fault_ = true;
                            break;
                        }
                        dot_input.data[group] = mxm_numeric::bf16_dot(
                            activation.data,
                            storage_->weights[row][column][buffer][group]).bits;
                    }
                }
                mxm_timing::shift_pipeline(next.cells[row][column].dot,
                    current.cells[row][column].dot, dot_input);

                const auto& dot = current.cells[row][column].dot.back();
                Fp32Token add_input{};
                if (row == 0) {
                    add_input = dot;
                } else {
                    const auto& below = current.cells[row - 1][column].partial.back();
                    if (dot.valid != below.valid
                        || (dot.valid && dot.tag != below.tag)) {
                        if (dot.valid || below.valid) cycle_frontend_fault_ = true;
                    } else if (dot.valid) {
                        add_input.valid = true;
                        add_input.tag = dot.tag;
                        for (std::size_t lane = 0; lane < kOutputs; ++lane)
                            add_input.data[lane] = mxm_numeric::fp32_add_bits(
                                dot.data[lane], below.data[lane]);
                    }
                }
                mxm_timing::shift_pipeline(next.cells[row][column].partial,
                    current.cells[row][column].partial, add_input);
            }
        }
    }

    void update_result_return(State& next, const State& current)
    {
        std::array<Fp32Token, kColumns> top{};
        for (std::size_t column = 0; column < kColumns; ++column)
            top[column] = current.cells[kRows - 1][column].partial.back();
        mxm_timing::advance_result_return(next.result_return.down,
            next.result_return.left, current.result_return.down,
            current.result_return.left, top);
    }

    static bool operation_reads(BackendOperation operation) noexcept
    { return operation == BackendOperation::AccStore || operation == BackendOperation::AccOut; }
    static bool operation_writes(BackendOperation operation) noexcept
    { return operation == BackendOperation::Store || operation == BackendOperation::AccStore; }

    void evaluate_backend_outputs(const State& state, Outputs& out) const
    {
        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto& issue = state.backend_wave.payload[column];
            out.backend_issue_valid_o[column] = inputs_.run_i
                && state.backend_wave.valid[column];
            out.backend_issue_bf16_o[column] = issue.out_bf16;
            out.backend_issue_operation_o[column] = issue.opcode;
            out.backend_issue_address_o[column] = issue.address;
            const auto& backend = state.backend[column];
            const auto& slot = backend.context[1];
            out.backend_slot_valid_o[column] = inputs_.run_i && slot.valid;
            out.backend_slot_bf16_o[column] = slot.bf16;
            out.backend_slot_operation_o[column] = slot.operation;
            out.backend_slot_address_o[column] = slot.address;

            const auto bank = static_cast<std::size_t>(slot.address & 1u);
            const auto& read = backend.banks[bank];
            const bool response_match = read.read_valid
                && read.read_address == (slot.address >> 1u);
            out.alu_valid_o[column] = inputs_.run_i && slot.valid && slot.ok
                && observed_return_valid(state, column)
                && operation_reads(slot.operation) && response_match;
            out.operand_a_o[column] = observed_return_data(state, column);
            out.operand_b_o[column] = read.read_data;
            out.alu_address_o[column] = slot.address;
            out.alu_tag_o[column] = observed_return_tag(state, column);

            const auto& sum = backend.sum.back();
            const auto& bypass = backend.bypass.back();
            const bool accumulated_write = sum.valid
                && sum.operation == BackendOperation::AccStore;
            const bool direct_write = bypass.valid
                && bypass.operation == BackendOperation::Store;
            const auto& writeback = accumulated_write ? sum : bypass;
            const bool writeback_valid = accumulated_write || direct_write;
            const bool first_read = out.backend_issue_valid_o[column]
                && operation_reads(issue.opcode);
            for (std::size_t b = 0; b < 2; ++b) {
                const bool read_here = first_read && (issue.address & 1u) == b;
                const bool write_here = writeback_valid
                    && (writeback.address & 1u) == b;
                out.bank_request_valid_o[column][b] = read_here || write_here;
                out.bank_write_o[column][b] = write_here;
                out.bank_address_o[column][b] = static_cast<std::uint16_t>(
                    (write_here ? writeback.address : issue.address) >> 1u);
            }
            out.result_valid_o[column] = inputs_.run_i && backend.output.valid;
            out.result_bf16_o[column] = backend.output.bf16;
            out.result_o[column] = backend.output.data;
            out.result_address_o[column] = backend.output.address;
            out.result_tag_o[column] = backend.output.tag;
        }
    }

    static bool observed_return_valid(const State& state, std::size_t column)
    { return state.result_return.left[column][0].valid; }
    static const Fp32Vector& observed_return_data(const State& state, std::size_t column)
    { return state.result_return.left[column][0].data; }
    static std::uint16_t observed_return_tag(const State& state, std::size_t column)
    { return state.result_return.left[column][0].tag; }

    void update_backend(State& next, const State& current)
    {
        cycle_backend_fault_ = false;
        for (std::size_t column = 0; column < kColumns; ++column) {
            auto& next_column = next.backend[column];
            const auto& column_state = current.backend[column];
            const auto& issue = current.backend_wave.payload[column];
            const bool issue_valid = inputs_.run_i
                && current.backend_wave.valid[column];

            const auto& sum = column_state.sum.back();
            const auto& bypass = column_state.bypass.back();
            const bool accumulated_write = sum.valid
                && sum.operation == BackendOperation::AccStore;
            const bool direct_write = bypass.valid
                && bypass.operation == BackendOperation::Store;
            const bool writeback_valid = accumulated_write || direct_write;
            const auto& writeback = accumulated_write ? sum : bypass;
            const bool first_read = issue_valid && operation_reads(issue.opcode);
            bool address_hazard = false;
            for (std::size_t index = 0; index < column_state.pending_valid.size(); ++index)
                address_hazard = address_hazard
                    || (issue_valid && column_state.pending_valid[index]
                        && column_state.pending_address[index] == issue.address
                        && (first_read || operation_writes(issue.opcode)));
            const bool bank_collision = first_read && writeback_valid
                && ((issue.address ^ writeback.address) & 1u) == 0;
            const bool address_in_range = issue.address < kAccumulatorDepth;
            const bool first_ok = inputs_.run_i && !bank_collision
                && !address_hazard && address_in_range;
            const auto issue_bank = static_cast<std::size_t>(issue.address & 1u);
            const auto issue_row = static_cast<std::size_t>(issue.address >> 1u);
            const bool invalid_read = first_read && first_ok
                && !storage_->accumulator_valid[column][issue_bank][issue_row];
            const bool accepted_read = first_read && first_ok && !invalid_read;

            for (std::size_t bank = 0; bank < 2; ++bank) {
                auto& next_bank = next_column.banks[bank];
                const auto& bank_state = column_state.banks[bank];
                next_bank.capture_valid = false;
                next_bank.read_valid = bank_state.capture_valid;
                if (bank_state.capture_valid) {
                    next_bank.read_data = bank_state.capture_data;
                    next_bank.read_address = bank_state.capture_address;
                }
                const bool read_here = accepted_read && issue_bank == bank;
                const bool write_here = writeback_valid && !bank_collision
                    && ((writeback.address & 1u) == bank);
                if (read_here) {
                    next_bank.capture_valid = true;
                    next_bank.capture_data
                        = storage_->accumulator[column][bank][issue_row];
                    next_bank.capture_address = static_cast<std::uint16_t>(issue_row);
                }
                if (write_here) {
                    stage_accumulator_write(column, bank,
                        writeback.address >> 1u, writeback.data);
                }
            }

            BackendToken context_input{};
            context_input.valid = issue_valid;
            context_input.ok = issue_valid && (
                ((issue.opcode == BackendOperation::DirectOut
                    || issue.opcode == BackendOperation::Store) && first_ok)
                || accepted_read);
            context_input.bf16 = issue.out_bf16;
            context_input.operation = issue.opcode;
            context_input.address = issue.address;
            mxm_timing::shift_pipeline(
                next_column.context, column_state.context, context_input);

            const auto& slot = column_state.context[1];
            const auto bank = static_cast<std::size_t>(slot.address & 1u);
            const auto& read = column_state.banks[bank];
            const bool response_match = read.read_valid
                && read.read_address == (slot.address >> 1u);
            const bool partial_valid = observed_return_valid(current, column);
            const bool alu_valid = slot.valid && slot.ok && partial_valid
                && operation_reads(slot.operation) && response_match;
            BackendToken sum_input{};
            sum_input.valid = alu_valid;
            sum_input.ok = alu_valid;
            sum_input.bf16 = slot.bf16;
            sum_input.operation = slot.operation;
            sum_input.address = slot.address;
            sum_input.tag = observed_return_tag(current, column);
            if (alu_valid) {
                for (std::size_t lane = 0; lane < kOutputs; ++lane)
                    sum_input.data[lane] = mxm_numeric::fp32_add_bits(
                        observed_return_data(current, column)[lane],
                        read.read_data[lane]);
            }
            mxm_timing::shift_pipeline(next_column.sum, column_state.sum, sum_input);

            BackendToken bypass_input{};
            bypass_input.valid = slot.valid && slot.ok && partial_valid
                && !operation_reads(slot.operation);
            bypass_input.ok = bypass_input.valid;
            bypass_input.bf16 = slot.bf16;
            bypass_input.operation = slot.operation;
            bypass_input.address = slot.address;
            bypass_input.tag = observed_return_tag(current, column);
            bypass_input.data = observed_return_data(current, column);
            mxm_timing::shift_pipeline(
                next_column.bypass, column_state.bypass, bypass_input);

            const bool accumulated_output = sum.valid
                && sum.operation == BackendOperation::AccOut;
            const bool direct_output = bypass.valid
                && bypass.operation == BackendOperation::DirectOut;
            const auto& output_source = accumulated_output ? sum : bypass;
            next_column.output = {};
            if (accumulated_output || direct_output) {
                next_column.output = output_source;
                next_column.output.valid = true;
                if (output_source.bf16) {
                    for (auto& word : next_column.output.data)
                        word = mxm_numeric::fp32_to_bf16(word);
                }
            }

            next_column.pending_valid[0] = issue_valid && first_ok
                && (issue.opcode == BackendOperation::Store || accepted_read);
            next_column.pending_address[0] = issue.address;
            for (std::size_t index = 1; index < next_column.pending_valid.size(); ++index) {
                next_column.pending_valid[index] = column_state.pending_valid[index - 1];
                next_column.pending_address[index] = column_state.pending_address[index - 1];
            }

            const bool response_mismatch = slot.valid && slot.ok
                && operation_reads(slot.operation) && !response_match;
            cycle_backend_fault_ = cycle_backend_fault_
                || (slot.valid != partial_valid) || bank_collision
                || address_hazard || invalid_read || response_mismatch
                || (issue_valid && !address_in_range)
                || (issue_valid && issue.out_bf16
                    && issue.opcode != BackendOperation::DirectOut
                    && issue.opcode != BackendOperation::AccOut)
                || (accumulated_output && direct_output)
                || (accumulated_write && direct_write);
        }
    }

    void clear_runtime_state(State& state)
    {
        state.compute_wave = {};
        state.load_wave = {};
        state.param_wave = {};
        state.dequant_wave = {};
        state.backend_wave = {};
        state.next_tag = 0;
        state.load_remaining = {};
        state.load_index = {};
        state.dequant = {};
        state.activation_link = {};
        state.weight_link = {};
        state.cells = {};
        state.result_return = {};
        state.backend = {};
        state.frontend_fault = false;
        state.backend_fault = false;
        cycle_frontend_fault_ = false;
        cycle_backend_fault_ = false;
    }

    void stage_weight_write(std::size_t row, std::size_t column,
        std::size_t buffer, std::size_t group, const Bf16Vector& data, bool valid)
    {
        if (pending_weight_write_count_ >= pending_weight_writes_.size())
            throw std::logic_error("MXM weight sparse write overflow");
        pending_weight_writes_[pending_weight_write_count_++] =
            WeightWrite{row, column, buffer, group, data, valid};
    }

    void stage_accumulator_write(std::size_t column, std::size_t bank,
        std::size_t row, const Fp32Vector& data)
    {
        if (row >= kAccumulatorBankDepth)
            throw std::out_of_range("MXM accumulator write address");
        if (pending_acc_write_count_ >= pending_acc_writes_.size())
            throw std::logic_error("MXM accumulator sparse write overflow");
        pending_acc_writes_[pending_acc_write_count_++] =
            AccumulatorWrite{column, bank, row, data};
    }

    static void check_column_address(std::size_t column, std::size_t address)
    {
        if (column >= kColumns || address >= kAccumulatorDepth)
            throw std::out_of_range("MXM accumulator probe address");
    }

    Inputs inputs_{};
    mutable Outputs observed_{};
    Outputs pre_posedge_observed_{};
    std::unique_ptr<timing::ClockedState<State>> state_;
    std::unique_ptr<BulkStorage> storage_{};
    std::array<WeightWrite, kRows * kColumns * kOutputs> pending_weight_writes_{};
    std::size_t pending_weight_write_count_{};
    std::array<AccumulatorWrite, kColumns> pending_acc_writes_{};
    std::size_t pending_acc_write_count_{};
    bool clear_weight_valid_{};
    bool clear_accumulator_valid_{};
    bool cycle_frontend_fault_{};
    bool cycle_backend_fault_{};
};

inline MxmCore::MxmCore()
    : state_(std::make_unique<timing::ClockedState<State>>())
    , storage_(std::make_unique<BulkStorage>())
{}

inline MxmCore::~MxmCore() = default;

} // namespace ftlpu
