#include "softmax_dataflow_harness.hpp"

int main()
{
    using namespace ftlpu::test::softmax_dataflow;
    return run({
        "SmolLM2 prefill attention",
        16,
        0.125f,
        {5, 8, 12, 16, 5, 8, 12, 16},
    });
}
