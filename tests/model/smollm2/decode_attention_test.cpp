#include "softmax_dataflow_harness.hpp"

int main()
{
    using namespace ftlpu::test::softmax_dataflow;
    return run({
        "SmolLM2 decode attention",
        12,
        0.125f,
        {12, 10, 7, 4, 12, 10, 7, 4},
    });
}
