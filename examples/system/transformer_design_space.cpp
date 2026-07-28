#include "ftlpu/workload/design_space.hpp"

#include <array>
#include <iomanip>
#include <iostream>

int main()
{
    using namespace ftlpu::workload;
    constexpr std::array<std::size_t, 4> edges {16, 32, 64, 320};

    std::cout
        << "workload,array,m_blocks,n_blocks,k_blocks,"
           "tail_m,tail_n,tail_k,pe_fill_percent,"
           "fp16_weight_bytes,weight_load_cycles,"
           "hemi_stream_demand,hemi_stream_capacity,stream_deficit\n";
    for (const auto& shape : transformer_gemms()) {
        for (const auto edge : edges) {
            const auto result = evaluate_gemm(shape, edge);
            std::cout << shape.name << ',' << edge << ','
                      << result.m_blocks << ','
                      << result.n_blocks << ','
                      << result.k_blocks << ','
                      << result.tail_rows << ','
                      << result.tail_cols << ','
                      << result.tail_k << ','
                      << std::fixed << std::setprecision(2)
                      << result.pe_fill_ratio() * 100.0 << ','
                      << result.design.fp16_weight_bytes_per_array << ','
                      << result.design.fp16_weight_load_cycles << ','
                      << result.design.aggregate_streams << ','
                      << result.design.stream_capacity << ','
                      << result.design.stream_deficit << '\n';
        }
    }

    const auto groq = sram_geometry<ftlpu::hw::GroqLikeConfig>();
    const auto transformer =
        sram_geometry<ftlpu::hw::TransformerEvalConfig>();
    std::cout << "\nconfig,row_bytes,banks,rows_per_bank,slice_capacity_bytes\n"
              << "GroqLike," << groq.row_bytes << ',' << groq.banks << ','
              << groq.rows_per_bank << ',' << groq.slice_capacity_bytes << '\n'
              << "TransformerEval," << transformer.row_bytes << ','
              << transformer.banks << ',' << transformer.rows_per_bank << ','
              << transformer.slice_capacity_bytes << '\n';
    return 0;
}
