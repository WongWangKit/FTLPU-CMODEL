#pragma once

#include "ftlpu/core/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu::workload {

template <typename T>
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t columns, T initial = {})
        : rows_(rows)
        , columns_(columns)
        , values_(rows * columns, initial)
    {
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }
    std::vector<T>& values() noexcept { return values_; }
    const std::vector<T>& values() const noexcept { return values_; }

    T& operator()(std::size_t row, std::size_t column)
    {
        return values_.at(row * columns_ + column);
    }
    const T& operator()(std::size_t row, std::size_t column) const
    {
        return values_.at(row * columns_ + column);
    }

private:
    std::size_t rows_{0};
    std::size_t columns_{0};
    std::vector<T> values_{};
};

struct W8ColumnScaledWeights {
    Matrix<std::int8_t> values{};
    std::vector<float> scales{};

    void validate() const
    {
        if (values.rows() == 0 || values.columns() == 0
            || scales.size() != values.columns()) {
            throw std::invalid_argument(
                "W8 weights need one non-empty scale per output column");
        }
    }

    float dequantized(std::size_t k, std::size_t n) const
    {
        return round_to_fp16(
            static_cast<float>(values(k, n)) * scales.at(n));
    }
};

inline Matrix<float> project_w8a16(
    const Matrix<float>& activations,
    const W8ColumnScaledWeights& weights,
    bool round_output = false)
{
    weights.validate();
    if (activations.columns() != weights.values.rows()) {
        throw std::invalid_argument(
            "projection activation K does not match weight K");
    }
    Matrix<float> output(
        activations.rows(), weights.values.columns(), 0.0f);
    for (std::size_t m = 0; m < output.rows(); ++m) {
        for (std::size_t n = 0; n < output.columns(); ++n) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < activations.columns(); ++k) {
                sum += round_to_fp16(activations(m, k))
                    * weights.dequantized(k, n);
            }
            output(m, n) = round_output ? round_to_fp16(sum) : sum;
        }
    }
    return output;
}

struct FfnReferenceResult {
    Matrix<float> gate{};
    Matrix<float> up{};
    Matrix<float> swiglu{};
    Matrix<float> output{};
};

inline FfnReferenceResult ffn_w8a16_reference(
    const Matrix<float>& input,
    const W8ColumnScaledWeights& gate_weights,
    const W8ColumnScaledWeights& up_weights,
    const W8ColumnScaledWeights& down_weights)
{
    auto gate = project_w8a16(input, gate_weights, false);
    auto up = project_w8a16(input, up_weights, false);
    if (gate.rows() != up.rows() || gate.columns() != up.columns()
        || down_weights.values.rows() != gate.columns()) {
        throw std::invalid_argument("FFN gate/up/down shapes do not compose");
    }

    Matrix<float> swiglu(gate.rows(), gate.columns(), 0.0f);
    for (std::size_t row = 0; row < gate.rows(); ++row) {
        for (std::size_t column = 0; column < gate.columns(); ++column) {
            const auto sigmoid =
                1.0f / (1.0f + std::exp(-gate(row, column)));
            swiglu(row, column) = round_to_fp16(
                gate(row, column) * up(row, column) * sigmoid);
        }
    }
    auto output = project_w8a16(swiglu, down_weights, true);
    return {
        std::move(gate),
        std::move(up),
        std::move(swiglu),
        std::move(output),
    };
}

struct AttentionReferenceConfig {
    std::size_t query_heads{0};
    std::size_t kv_heads{0};
    std::size_t head_dim{0};
    float rope_theta{10000.0f};
};

struct AttentionReferenceResult {
    Matrix<float> query{};
    Matrix<float> key{};
    Matrix<float> value{};
    std::vector<float> probabilities{}; // [q_head, query, key]
    Matrix<float> context{};
    Matrix<float> output{};
};

inline AttentionReferenceResult causal_attention_w8a16_reference(
    const Matrix<float>& input,
    const W8ColumnScaledWeights& query_weights,
    const W8ColumnScaledWeights& key_weights,
    const W8ColumnScaledWeights& value_weights,
    const W8ColumnScaledWeights& output_weights,
    AttentionReferenceConfig config)
{
    if (config.query_heads == 0 || config.kv_heads == 0
        || config.head_dim == 0
        || config.query_heads % config.kv_heads != 0
        || query_weights.values.columns()
            != config.query_heads * config.head_dim
        || key_weights.values.columns()
            != config.kv_heads * config.head_dim
        || value_weights.values.columns()
            != config.kv_heads * config.head_dim) {
        throw std::invalid_argument("attention head/weight shapes are invalid");
    }
    auto query = project_w8a16(input, query_weights, false);
    auto key = project_w8a16(input, key_weights, false);
    auto value = project_w8a16(input, value_weights, true);
    const auto seq = input.rows();

    const auto apply_rope = [&](Matrix<float>& matrix, std::size_t heads) {
        if (config.head_dim % 2 != 0) {
            throw std::invalid_argument("RoPE head dimension must be even");
        }
        const auto half = config.head_dim / 2;
        auto original = matrix;
        for (std::size_t token = 0; token < seq; ++token) {
            for (std::size_t head = 0; head < heads; ++head) {
                for (std::size_t pair = 0; pair < half; ++pair) {
                    const auto lo_column =
                        head * config.head_dim + pair;
                    const auto hi_column = lo_column + half;
                    const auto inverse_frequency = 1.0f / std::pow(
                        config.rope_theta,
                        static_cast<float>(2 * pair)
                            / static_cast<float>(config.head_dim));
                    const auto angle =
                        static_cast<float>(token) * inverse_frequency;
                    const auto cosine = round_to_fp16(std::cos(angle));
                    const auto sine = round_to_fp16(std::sin(angle));
                    const auto lo = original(token, lo_column);
                    const auto hi = original(token, hi_column);
                    matrix(token, lo_column) =
                        round_to_fp16(lo * cosine - hi * sine);
                    matrix(token, hi_column) =
                        round_to_fp16(hi * cosine + lo * sine);
                }
            }
        }
    };
    apply_rope(query, config.query_heads);
    apply_rope(key, config.kv_heads);

    const auto probability_index =
        [seq](std::size_t head, std::size_t q, std::size_t k) {
            return (head * seq + q) * seq + k;
        };
    std::vector<float> probabilities(
        config.query_heads * seq * seq, 0.0f);
    const auto head_group = config.query_heads / config.kv_heads;
    const auto scale = 1.0f / std::sqrt(
        static_cast<float>(config.head_dim));

    // Explicit three-pass softmax: max, exp/sum, normalize.
    for (std::size_t q_head = 0; q_head < config.query_heads; ++q_head) {
        const auto kv_head = q_head / head_group;
        for (std::size_t q = 0; q < seq; ++q) {
            std::vector<float> logits(seq, 0.0f);
            auto maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t k = 0; k <= q; ++k) {
                float dot = 0.0f;
                for (std::size_t d = 0; d < config.head_dim; ++d) {
                    dot += query(q, q_head * config.head_dim + d)
                        * key(k, kv_head * config.head_dim + d);
                }
                logits[k] = dot * scale;
                maximum = std::max(maximum, logits[k]);
            }
            float sum = 0.0f;
            for (std::size_t k = 0; k <= q; ++k) {
                logits[k] = std::exp(logits[k] - maximum);
                sum += logits[k];
            }
            for (std::size_t k = 0; k <= q; ++k) {
                probabilities[probability_index(q_head, q, k)] =
                    round_to_fp16(logits[k] / sum);
            }
        }
    }

    Matrix<float> context(
        seq, config.query_heads * config.head_dim, 0.0f);
    for (std::size_t q_head = 0; q_head < config.query_heads; ++q_head) {
        const auto kv_head = q_head / head_group;
        for (std::size_t q = 0; q < seq; ++q) {
            for (std::size_t d = 0; d < config.head_dim; ++d) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < seq; ++k) {
                    sum += probabilities[
                               probability_index(q_head, q, k)]
                        * value(k, kv_head * config.head_dim + d);
                }
                context(q, q_head * config.head_dim + d) =
                    round_to_fp16(sum);
            }
        }
    }
    auto output = project_w8a16(context, output_weights, true);
    return {
        std::move(query),
        std::move(key),
        std::move(value),
        std::move(probabilities),
        std::move(context),
        std::move(output),
    };
}

} // namespace ftlpu::workload
