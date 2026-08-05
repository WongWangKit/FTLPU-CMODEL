#include "ftlpu/vxm/compiler/compiler.hpp"
#include "compiler_test_execution.hpp"
#include "compiler_test_feedback.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <variant>

namespace {

using namespace ftlpu;
using namespace ftlpu::vxm::compiler;

bool has_operation(const VxmCompiledPhase& phase, std::size_t stage,
                   const VxmLaneOperation& operation)
{
    for (const auto& command : phase.config_commands) {
        const auto decoded = VxmCompactInstructionCodec::decode(
            command.stage, command.packet);
        if (command.stage == stage
            && decoded.instruction.operation == operation) {
            return true;
        }
    }
    return false;
}

const VxmStreamRequirement& require_stream(
    const VxmCompiledPhase& phase, ValueId value,
    VxmStreamDirection direction)
{
    for (const auto& requirement : phase.stream_requirements) {
        if (requirement.value == value
            && requirement.direction == direction) {
            return requirement;
        }
    }
    throw std::logic_error("required stream was not generated");
}

bool has_stream(
    const VxmCompiledPhase& phase, ValueId value,
    VxmStreamDirection direction)
{
    return std::any_of(
        phase.stream_requirements.begin(),
        phase.stream_requirements.end(),
        [value, direction](const auto& requirement) {
            return requirement.value == value
                && requirement.direction == direction;
        });
}

bool head_uses_feedback(const VxmCompiledPhase& phase)
{
    for (const auto& command : phase.config_commands) {
        if (command.stage != 0) continue;
        const auto decoded = VxmCompactInstructionCodec::decode(
            command.stage, command.packet);
        if (decoded.instruction.lhs.kind
            == VxmLaneOperandKind::Feedback) {
            return true;
        }
    }
    return false;
}

void check_fp16_kernel_interface(const VxmKernel& kernel)
{
    for (const auto& node : kernel.nodes) {
        assert(node.type.element_type == VxmIrElementType::Float16);
    }
}

VxmIrNode& mutable_named(VxmKernel& kernel, const std::string& name)
{
    for (auto& node : kernel.nodes) {
        if (node.name == name) return node;
    }
    throw std::logic_error("test node was not found");
}

void expect_invalid_kernel(const VxmKernel& kernel)
{
    auto rejected = false;
    try {
        kernel.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

void check_kernel_validation()
{
    {
        auto kernel = make_softmax_kernel(1, 8);
        mutable_named(kernel, "exponent").inputs.push_back(
            kernel.named("maximum").output);
        expect_invalid_kernel(kernel);
    }
    {
        auto kernel = make_softmax_kernel(1, 8);
        mutable_named(kernel, "exponent").type.shape = {1, 4};
        expect_invalid_kernel(kernel);
    }
    {
        auto kernel = make_softmax_kernel(1, 8);
        mutable_named(kernel, "maximum").axis.reset();
        expect_invalid_kernel(kernel);
    }
    {
        auto kernel = make_softmax_kernel(1, 8);
        mutable_named(kernel, "maximum").type.shape = {1, 8};
        expect_invalid_kernel(kernel);
    }
}

void check_dead_code_elimination()
{
    auto builder = VxmKernelBuilder{"dead_code"};
    const auto x = builder.input("x", {{8}});
    (void)builder.input("unused_input", {{8}});
    const auto live = builder.negate(x, "live");
    (void)builder.exp(x, "dead");
    builder.output(live);

    const auto kernel = builder.finish();
    assert(kernel.nodes.size() == 2);
    assert(kernel.inputs.size() == 1);
    assert(kernel.inputs.front() == x.id);
    assert(kernel.named("live").output == live.id);

    auto dead_was_removed = false;
    try {
        (void)kernel.named("dead");
    } catch (const std::out_of_range&) {
        dead_was_removed = true;
    }
    assert(dead_was_removed);
}

void check_divide_expansion()
{
    auto builder = VxmKernelBuilder{"divide"};
    const auto lhs = builder.input("lhs", {{8}});
    const auto rhs = builder.input("rhs", {{8}});
    const auto quotient = builder.divide(lhs, rhs, "quotient");
    builder.output(quotient);

    const auto kernel = builder.finish();
    const auto& multiply = kernel.producer(quotient.id);
    assert(multiply.opcode == VxmIrOpcode::Multiply);
    assert(multiply.inputs.size() == 2);
    assert(multiply.inputs[0] == lhs.id);

    const auto& reciprocal = kernel.producer(multiply.inputs[1]);
    assert(reciprocal.opcode == VxmIrOpcode::Reciprocal);
    assert(reciprocal.inputs.size() == 1);
    assert(reciprocal.inputs[0] == rhs.id);
    check_fp16_kernel_interface(kernel);

    const auto program = compile_kernel(kernel);
    assert(program.phases.size() == 1);
    assert(program.phases.front().chain_depth == VxmChainDepth::Eight);
    assert(has_operation(
        program.phases.front(), 3, VxmSpecialAluOpcode::Reciprocal));
    assert(has_operation(
        program.phases.front(), 4, VxmAluOpcode::Multiply));
}

void check_dependency_lowering_rules()
{
    {
        auto kernel = make_softmax_kernel(1, 8);
        const auto graph = VxmDependencyGraph{kernel};
        const auto exponent = kernel.named("exponent").output;
        const auto x = kernel.named("x").output;
        const auto maximum = kernel.named("maximum").output;
        const auto decision =
            graph.decide_fanout(exponent, {x, maximum});
        assert(graph.use_count(exponent) == 2);
        assert(decision.action == VxmFanoutAction::Recompute);
        assert(decision.recompute_operations == 2);
        assert(decision.directly_feeds_reduction);
    }
    {
        auto builder = VxmKernelBuilder{"long_recompute"};
        const auto x = builder.input("x", {{1, 8}});
        const auto one = builder.constant(1.0f, "one");
        const auto one_vector =
            builder.broadcast(one, {1, 8}, "one_vector");
        const auto add = builder.add(x, one_vector, "add");
        const auto multiply =
            builder.multiply(add, one_vector, "multiply");
        const auto exponent = builder.exp(multiply, "exponent");
        const auto sum = builder.reduce_add(exponent, -1, "sum");
        const auto inverse = builder.reciprocal(sum, "inverse");
        const auto inverse_vector =
            builder.broadcast(inverse, {1, 8}, "inverse_vector");
        const auto output =
            builder.multiply(exponent, inverse_vector, "output");
        builder.output(output);
        const auto kernel = builder.finish();
        const auto graph = VxmDependencyGraph{kernel};
        const auto decision = graph.decide_fanout(
            exponent.id, {x.id, one.id});
        assert(decision.action == VxmFanoutAction::Materialize);
        assert(decision.recompute_operations > 2);
        assert(decision.directly_feeds_reduction);
        const auto program = compile_kernel(kernel);
        assert(program.phases.size() == 4);
        assert(program.schedule.phases[0].outputs.front().value
               == exponent.id);
        assert(program.schedule.phases[1].inputs.front().value
               == exponent.id);
    }
    {
        const auto placement = detail::place_chain({
            {VxmAluOpcode::Add, false, false},
            {VxmSpecialAluOpcode::Reciprocal, false, false},
        });
        assert(placement.depth == VxmChainDepth::Four);
        assert(placement.stages
               == std::vector<std::size_t>({0, 3}));
        assert(placement.bypass_count == 2);
    }
    {
        auto builder = VxmKernelBuilder{"long_chain"};
        const auto x = builder.input("x", {{8}});
        const auto one = builder.constant(1.0f, "one");
        const auto one_vector =
            builder.broadcast(one, {8}, "one_vector");
        auto value = x;
        for (std::size_t index = 0; index < 10; ++index) {
            value = builder.add(
                value, one_vector, "add_" + std::to_string(index));
        }
        builder.output(value);
        const auto program = compile_kernel(builder.finish());
        assert(program.phases.size() == 2);
        assert(program.phases[0].chain_depth == VxmChainDepth::Eight);
        assert(program.phases[1].chain_depth == VxmChainDepth::Two);
    }
    {
        auto kernels = std::vector<VxmKernel>{};
        kernels.push_back(make_softmax_kernel(1, 8));
        kernels.push_back(make_rmsnorm_kernel(1, 8, 1.0e-5f));
        kernels.push_back(make_swiglu_kernel(8));
        const auto expected_phases =
            std::array<std::size_t, 3>{4, 3, 1};
        for (std::size_t index = 0; index < kernels.size(); ++index) {
            auto& kernel = kernels[index];
            kernel.name = "renamed_kernel_" + std::to_string(index);
            for (auto& node : kernel.nodes) {
                node.name = "renamed_" + std::to_string(node.output);
            }
            const auto program = compile_kernel(std::move(kernel));
            assert(program.phases.size() == expected_phases[index]);
        }
    }
}

void check_fifo_bound(const VxmCompiledProgram& program)
{
    for (const auto& phase : program.phases) {
        auto count_by_stage = std::map<std::size_t, std::size_t>{};
        auto last_arrival = std::map<std::size_t, std::size_t>{};
        for (const auto& command : phase.config_commands) {
            const auto ordinal = count_by_stage[command.stage]++;
            if (ordinal == 0) {
                assert(command.arrival_cycle
                       == phase.config_deadline_cycle);
            } else {
                assert(command.arrival_cycle
                       == last_arrival[command.stage] + 1);
            }
            last_arrival[command.stage] = command.arrival_cycle;
        }
        for (const auto& [stage, count] : count_by_stage) {
            (void)stage;
            assert(count <= 3);
        }
    }
}

std::filesystem::path output_path(const char* filename)
{
    const auto source = std::filesystem::path(__FILE__);
    if (source.is_absolute()) {
        const auto directory = source.parent_path() / "results";
        std::filesystem::create_directories(directory);
        return directory / filename;
    }
    const auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / source)) {
        const auto directory =
            (cwd / source).parent_path() / "results";
        std::filesystem::create_directories(directory);
        return directory / filename;
    }
    if (std::filesystem::exists(cwd / "compiler_test.cpp")) {
        const auto directory = cwd / "results";
        std::filesystem::create_directories(directory);
        return directory / filename;
    }
    const auto directory =
        cwd / "tests" / "vxm" / "compiler" / "results";
    std::filesystem::create_directories(directory);
    return directory / filename;
}

VxmCompiledProgram check_softmax()
{
    auto kernel = make_softmax_kernel(1, 128);
    check_fp16_kernel_interface(kernel);
    const auto& reduction = kernel.named("maximum");
    assert(reduction.axis && *reduction.axis == 1);
    assert(reduction.type.shape == std::vector<std::size_t>({1, 1}));

    const auto program = compile_kernel(std::move(kernel));
    assert(program.phases.size() == 4);
    assert(program.phases[0].chain_depth == VxmChainDepth::Two);
    assert(program.phases[1].chain_depth == VxmChainDepth::Four);
    assert(program.phases[2].chain_depth == VxmChainDepth::Four);
    assert(program.phases[3].chain_depth == VxmChainDepth::Four);
    assert(has_operation(
        program.phases[1], 1, VxmSpecialAluOpcode::Exp));
    assert(has_operation(
        program.phases[2], 3, VxmSpecialAluOpcode::Reciprocal));
    assert(has_operation(
        program.phases[3], 3, VxmAluOpcode::Multiply));

    const auto x = program.kernel.named("x").output;
    const auto maximum = program.kernel.named("maximum").output;
    const auto& x_stream = require_stream(
        program.phases[1], x, VxmStreamDirection::Input);
    const auto& max_stream = require_stream(
        program.phases[1], maximum, VxmStreamDirection::Input);
    assert(x_stream.stream_base == 0);
    assert(max_stream.stream_base == VxmLane::kStreamGroupBytes);
    assert(max_stream.hold && max_stream.transfer_count == 1);
    assert(max_stream.reuse_count == 128);
    assert(program.phases[3].local_scalar_loads.size() == 1);
    assert(program.phases[3].local_scalar_loads[0].destination_stage == 3);
    const auto denominator =
        program.kernel.named("denominator").output;
    assert(program.phases[2].feedback_from_previous);
    assert(head_uses_feedback(program.phases[2]));
    assert(!has_stream(
        program.phases[1], denominator,
        VxmStreamDirection::Output));
    assert(!has_stream(
        program.phases[2], denominator,
        VxmStreamDirection::Input));
    assert(program.phases[2].config_deadline_cycle + 2
           == program.phases[2].data_start_cycle);
    check_fifo_bound(program);
    return program;
}

VxmCompiledProgram check_rmsnorm()
{
    auto kernel = make_rmsnorm_kernel(1, 128, 1.0e-5f);
    check_fp16_kernel_interface(kernel);
    const auto& inverse_rms = kernel.named("inverse_rms");
    assert(inverse_rms.epsilon && *inverse_rms.epsilon == 1.0e-5f);
    auto epsilon_tensor_was_removed = false;
    try {
        (void)kernel.named("epsilon");
    } catch (const std::out_of_range&) {
        epsilon_tensor_was_removed = true;
    }
    assert(epsilon_tensor_was_removed);
    const auto program = compile_kernel(std::move(kernel));
    assert(program.phases.size() == 3);
    assert(program.phases[0].chain_depth == VxmChainDepth::Two);
    assert(program.phases[1].chain_depth == VxmChainDepth::Four);
    assert(program.phases[2].chain_depth == VxmChainDepth::Two);
    assert(has_operation(
        program.phases[0], 0, VxmAluOpcode::Multiply));
    assert(has_operation(
        program.phases[1], 3, VxmSpecialAluOpcode::Rsqrt));
    assert(has_operation(
        program.phases[2], 0, VxmAluOpcode::Multiply));
    assert(has_operation(
        program.phases[2], 1, VxmAluOpcode::Multiply));
    assert(program.phases[2].local_scalar_loads.size() == 1);
    assert(program.phases[2].local_scalar_loads[0].destination_stage == 1);
    const auto square_sum =
        program.kernel.named("square_sum").output;
    assert(program.phases[1].feedback_from_previous);
    assert(head_uses_feedback(program.phases[1]));
    assert(!has_stream(
        program.phases[0], square_sum,
        VxmStreamDirection::Output));
    assert(!has_stream(
        program.phases[1], square_sum,
        VxmStreamDirection::Input));
    assert(program.phases[1].config_deadline_cycle + 2
           == program.phases[1].data_start_cycle);
    check_fifo_bound(program);
    return program;
}

VxmCompiledProgram check_swiglu()
{
    auto kernel = make_swiglu_kernel(256);
    check_fp16_kernel_interface(kernel);
    const auto program = compile_kernel(std::move(kernel));
    assert(program.phases.size() == 1);
    const auto& phase = program.phases.front();
    assert(phase.chain_depth == VxmChainDepth::Eight);
    assert(has_operation(phase, 1, VxmSpecialAluOpcode::Exp));
    assert(has_operation(phase, 3, VxmSpecialAluOpcode::Reciprocal));
    assert(has_operation(phase, 4, VxmAluOpcode::Multiply));
    assert(has_operation(phase, 5, VxmAluOpcode::Multiply));

    const auto gate = program.kernel.named("gate").output;
    const auto up = program.kernel.named("up").output;
    assert(require_stream(
        phase, gate, VxmStreamDirection::Input).stream_base == 0);
    assert(require_stream(
        phase, up, VxmStreamDirection::Input).stream_base
        == VxmLane::kStreamGroupBytes);
    const auto output = program.kernel.named("output").output;
    assert(require_stream(
        phase, output, VxmStreamDirection::Output).stream_base
        == VxmLane::fixed_output_stream_for_block(3));
    check_fifo_bound(program);
    return program;
}

void write_results(const std::vector<VxmCompiledProgram>& programs)
{
    const auto kernel_path = output_path("kernel_ir_results.txt");
    const auto schedule_path = output_path("schedule_ir_results.txt");
    const auto compiled_path = output_path("compiled_program_results.txt");
    auto kernel_file = std::ofstream{kernel_path};
    auto schedule_file = std::ofstream{schedule_path};
    auto compiled_file = std::ofstream{compiled_path};
    assert(kernel_file && schedule_file && compiled_file);

    for (const auto& program : programs) {
        print_kernel_ir(kernel_file, program.kernel);
        print_schedule_ir(schedule_file, program.schedule);
        print_compiled_program(compiled_file, program);
    }

    std::cout << "Initial Kernel IR: " << kernel_path.string() << '\n'
              << "Intermediate Schedule IR: " << schedule_path.string() << '\n'
              << "Final compiled program: " << compiled_path.string() << '\n';
}

void check_fp16_physical_interface(const VxmCompiledProgram& program)
{
    for (const auto& phase : program.phases) {
        for (const auto& command : phase.config_commands) {
            const auto decoded = VxmCompactInstructionCodec::decode(
                command.stage, command.packet);
            assert(decoded.instruction.output_type
                   == VxmCastTarget::Float16);
            if (decoded.instruction.lhs.kind
                    == VxmLaneOperandKind::StreamFloat16
                || decoded.instruction.rhs.kind
                    == VxmLaneOperandKind::StreamFloat16) {
                assert(VxmLane::kStreamGroupBytes == 2);
            }
        }
    }
}

} // namespace

int main()
{
    check_kernel_validation();
    check_dead_code_elimination();
    check_divide_expansion();
    check_dependency_lowering_rules();
    auto programs = std::vector<VxmCompiledProgram>{};
    programs.push_back(check_softmax());
    programs.push_back(check_rmsnorm());
    programs.push_back(check_swiglu());
    for (const auto& program : programs) {
        check_fp16_physical_interface(program);
    }
    write_results(programs);
    vxm_compiler_test::feedback::check();
    vxm_compiler_test::execution::check();
    std::cout << "VXM compiler flow: Kernel IR -> Schedule IR -> "
                 "Feedback Schedule IR -> fixed Stream plan -> "
                 "compact config -> Slice execution passed\n";
    return 0;
}
