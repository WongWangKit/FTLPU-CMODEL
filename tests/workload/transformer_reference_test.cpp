#include "ftlpu/workload/design_space.hpp"
#include "ftlpu/workload/transformer_reference.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

ftlpu::workload::W8ColumnScaledWeights constant_weights(
    std::size_t k,
    std::size_t n,
    std::int8_t value,
    float scale)
{
    ftlpu::workload::W8ColumnScaledWeights result {
        {k, n, value},
        std::vector<float>(n, scale),
    };
    return result;
}

} // namespace

int main()
{
    using namespace ftlpu;
    using namespace ftlpu::workload;

    // Small non-divisible FFN numerically checks the same W8 column-scale,
    // FP16 boundary, SwiGLU and down-projection semantics as f621374.
    Matrix<float> input(3, 5);
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t k = 0; k < input.columns(); ++k) {
            input(row, k) =
                static_cast<float>((row + 1) * (k + 1)) * 0.125f;
        }
    }
    const auto gate = constant_weights(5, 7, 1, 0.5f);
    const auto up = constant_weights(5, 7, 2, 0.25f);
    const auto down = constant_weights(7, 4, 1, 0.125f);
    const auto ffn = ffn_w8a16_reference(input, gate, up, down);
    assert(ffn.output.rows() == 3);
    assert(ffn.output.columns() == 4);
    for (std::size_t row = 0; row < input.rows(); ++row) {
        float projection = 0.0f;
        for (std::size_t k = 0; k < input.columns(); ++k) {
            projection += round_to_fp16(input(row, k)) * 0.5f;
        }
        const auto swiglu = round_to_fp16(
            projection * projection
            / (1.0f + std::exp(-projection)));
        const auto expected = round_to_fp16(7.0f * swiglu * 0.125f);
        for (std::size_t column = 0;
             column < ffn.output.columns();
             ++column) {
            assert(ffn.gate(row, column) == projection);
            assert(ffn.up(row, column) == projection);
            assert(ffn.swiglu(row, column) == swiglu);
            assert(ffn.output(row, column) == expected);
        }
    }

    // GQA attention: zero Q/K makes the causal probabilities exactly the
    // FP16-rounded uniform distribution over visible keys. Non-zero V/O still
    // exercise P×V and output projection.
    Matrix<float> attention_input(5, 6);
    for (std::size_t token = 0; token < attention_input.rows(); ++token) {
        for (std::size_t hidden = 0;
             hidden < attention_input.columns();
             ++hidden) {
            attention_input(token, hidden) =
                static_cast<float>(token + hidden + 1) * 0.0625f;
        }
    }
    const auto q = constant_weights(6, 8, 0, 1.0f);
    const auto k = constant_weights(6, 4, 0, 1.0f);
    auto v = constant_weights(6, 4, 0, 0.125f);
    for (std::size_t hidden = 0; hidden < 6; ++hidden) {
        for (std::size_t column = 0; column < 4; ++column) {
            v.values(hidden, column) =
                static_cast<std::int8_t>(column + 1);
        }
    }
    auto o = constant_weights(8, 6, 0, 0.25f);
    for (std::size_t hidden = 0; hidden < 8; ++hidden) {
        o.values(hidden, hidden % 6) = 1;
    }
    const auto attention = causal_attention_w8a16_reference(
        attention_input,
        q,
        k,
        v,
        o,
        AttentionReferenceConfig {2, 1, 4, 10000.0f});
    const auto probability_index = [](
        std::size_t head, std::size_t query, std::size_t key) {
        return (head * 5 + query) * 5 + key;
    };
    for (std::size_t head = 0; head < 2; ++head) {
        for (std::size_t query = 0; query < 5; ++query) {
            float sum = 0.0f;
            for (std::size_t key = 0; key < 5; ++key) {
                const auto probability =
                    attention.probabilities[
                        probability_index(head, query, key)];
                if (key > query) {
                    assert(probability == 0.0f);
                } else {
                    assert(probability
                        == round_to_fp16(
                            1.0f / static_cast<float>(query + 1)));
                }
                sum += probability;
            }
            assert(std::fabs(sum - 1.0f) < 1.0e-3f);
        }
    }
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
        assert(attention.context(0, dimension)
            == attention.value(0, dimension));
        assert(attention.context(0, 4 + dimension)
            == attention.value(0, dimension));
    }

    constexpr TransformerDimensions full{};
    static_assert(full.seq_len == 128);
    static_assert(full.hidden == 576);
    static_assert(full.ffn_dim == 1536);
    static_assert(full.head_dim == 64);
    static_assert(full.kv_hidden == 192);
    return 0;
}
