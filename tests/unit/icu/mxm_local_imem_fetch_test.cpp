#include "ftlpu/icu/mem_local_frontend_fetch.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

int main()
{
    using Frontend = ftlpu::MxmLocalFrontendFetchModel<8, 2, 2, 1>;
    Frontend frontend;
    std::array<ftlpu::IcuRawImemWord128, 3> image{};
    for (std::size_t address = 0; address != image.size(); ++address) {
        for (std::size_t lane = 0; lane != 4; ++lane)
            image[address].lanes[lane] = static_cast<std::uint32_t>(
                0x1000 * address + lane);
        frontend.load_word(address + 2, image[address]);
    }

    std::vector<ftlpu::IcuRawImemWord128> observed;
    std::vector<bool> observed_last;
    std::size_t request_fire_count = 0;
    std::size_t response_fire_count = 0;
    bool saw_request_backpressure = false;
    bool saw_decoder_hold = false;
    bool saw_done = false;

    for (std::size_t cycle = 0; cycle != 32 && observed.size() != 3; ++cycle) {
        Frontend::Inputs inputs;
        inputs.start_valid_i = cycle == 0;
        inputs.start_base_address_i = 2;
        inputs.start_word_count_i = 3;
        inputs.imem_request_accept_enable_i = cycle != 2;
        inputs.decoder_word_ready_i = cycle != 8;
        frontend.cycle(inputs);
        const auto& ports = frontend.ports();
        request_fire_count += ports.imem_read_req_fire;
        response_fire_count += ports.imem_read_rsp_fire;
        saw_request_backpressure = saw_request_backpressure
            || (ports.imem_read_req_valid_o
                && !ports.imem_read_req_ready_i);
        saw_decoder_hold = saw_decoder_hold
            || (ports.decoder_word_valid_i
                && !ports.decoder_word_ready_o);
        saw_done = saw_done || frontend.done();
        if (ports.decoder_word_fire) {
            observed.push_back(ports.decoder_word_i);
            observed_last.push_back(ports.decoder_word_last_i);
        }
    }

    assert(observed.size() == image.size());
    assert(request_fire_count == image.size());
    assert(response_fire_count == image.size());
    assert(saw_request_backpressure);
    assert(saw_decoder_hold);
    for (std::size_t i = 0; i != image.size(); ++i)
        assert(observed[i].lanes == image[i].lanes);
    assert((observed_last == std::vector<bool>{false, false, true}));
    assert(saw_done);
    assert(!frontend.fault());
}
