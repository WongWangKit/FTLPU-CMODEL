#include "ftlpu/system/tsp_slice_system.hpp"

#include "../../support/sxm_macro_fixture.hpp"
#include <iostream>
#include <stdexcept>

using namespace ftlpu;
namespace {
constexpr auto K = hw::kLanesPerTile;
constexpr auto S = hw::kTileRows;
constexpr auto distance_to_mem = hw::kMemGroups + hw::kC2cToSxmStreamRegisterColumns;
constexpr auto beats = S * K;
constexpr std::size_t source_row = 0, scratch_row = beats + S, result_row = 2*(beats+S);
static_assert(result_row + beats <= hw::kSramDepthRows);

void check(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

std::uint16_t value(std::size_t row, std::size_t sl, std::size_t lane)
{
    return static_cast<std::uint16_t>(0x3400u + (row*S + sl)*K + lane);
}

// Real MEM program. The two bytes occupy different slices in MEM group zero.
// SR names are local fabric directions and work on either hemisphere.
void program_identity_mxm(TspSliceSystem& system, std::size_t compute)
{
    auto& icu = system.icu();
    constexpr std::size_t iw = 32;
    for (std::size_t col = 0; col < K; ++col)
        for (std::size_t p = 0; p < 2; ++p) {
            const auto slice = 8 + 2*col+p;
            for (std::size_t block=0; block<S; ++block)
                for (std::size_t sl=0; sl<S; ++sl)
                    for (std::size_t lane=0; lane<K; ++lane) {
                        const std::uint16_t weight = block==sl && col==lane ? 0x3f80 : 0;
                        system.initialize_mem_sram_lane_byte(Hemisphere::East,slice,sl,block,lane,
                            static_cast<std::uint8_t>(weight >> (8*p)));
                    }
            const auto queue = InstructionControlUnit::mem_queue(Hemisphere::East,slice,0);
            icu.enqueue_mem_nop(queue,iw-(distance_to_mem+1-slice/hw::kMemSlicesPerGroup));
            icu.enqueue_mem(queue,MemInstruction::Read(0,StreamId::East(2*col+p)));
            icu.enqueue_mem_repeat(queue,S-1,1,1);
        }
    icu.enqueue_mxm_load_nop(0,iw);
    for (std::size_t block=0; block<S; ++block)
        icu.enqueue_mxm(0,MxmControlInstruction::IWDirect16(0,block));
    icu.enqueue_mxm_compute_nop(0,compute);
    icu.enqueue_mxm(0,MxmControlInstruction::Compute(0,0,16,0,1,MxmAccumulatorDestination::Stream,
        MxmDataFormat::BFloat16,true,MxmAccumulatorOutputFormat::BFloat16));
    icu.enqueue_mxm_compute_repeat(0,beats-1,1);
}

void run(Hemisphere side, bool from_mxm)
{
    auto system = std::make_unique<TspSliceSystem>();
    auto& icu = system->icu();
    constexpr std::size_t compute = 48;
    const std::size_t read = from_mxm ? compute-distance_to_mem-1 : 4;
    const auto transpose = from_mxm ? compute+S : read+distance_to_mem;
    if (from_mxm) program_identity_mxm(*system,compute);
    const auto save = transpose+K+distance_to_mem;
    const auto load = save+beats+S+2;
    const auto rotate = load+distance_to_mem;
    const auto write = rotate+S+distance_to_mem;
    const auto output_read = write+beats+S;
    for (std::size_t p = 0; p < 2; ++p) {
        for (std::size_t r = 0; r < beats; ++r)
            for (std::size_t sl = 0; sl < S; ++sl)
                for (std::size_t lane = 0; lane < K; ++lane)
                    system->initialize_mem_sram_lane_byte(side,p,sl,source_row+r,lane,
                        static_cast<std::uint8_t>(value(r,sl,lane) >> (8*p)));

        const auto src = InstructionControlUnit::mem_queue(side,p,0);
        const auto tmp = InstructionControlUnit::mem_queue(side,2+p,0);
        icu.enqueue_mem_nop(src,read);
        icu.enqueue_mem(src,MemInstruction::Read(source_row,StreamId::East((from_mxm ? 0 : 16)+p)));
        icu.enqueue_mem_repeat(src,beats-1,1,1);
        icu.enqueue_mem_nop(src,write-read-beats);
        for (std::size_t h = 0; h < S; ++h)
            for (std::size_t c = 0; c < K; ++c)
                icu.enqueue_mem(src, MemInstruction::DiagonalWrite(result_row+c*S+h, StreamId::West(6+p)));
        icu.enqueue_mem_nop(src,output_read-write-beats);
        for (std::size_t row = 0; row < S; ++row)
            for (std::size_t c = 0; c < K; ++c)
                icu.enqueue_mem(src, MemInstruction::Read(result_row+c*S+row, StreamId::East(24+p)));

        icu.enqueue_mem_nop(tmp,save);
        for (std::size_t row = 0; row < S; ++row)
            for (std::size_t c = 0; c < K; ++c)
                icu.enqueue_mem(tmp, MemInstruction::Write(scratch_row+c*S+row, StreamId::West(2+p)));
        icu.enqueue_mem_nop(tmp,load-save-beats);
        for (std::size_t h = 0; h < S; ++h)
            for (std::size_t c = 0; c < K; ++c)
                icu.enqueue_mem(tmp, MemInstruction::DiagonalRead(scratch_row+c*S+(S-h)%S, StreamId::East(20+p)));
    }

    const auto t = SxmInstruction::LocalTranspose({(from_mxm ? StreamId::West(16) : StreamId::East(16)).packed()}, {StreamId::West(2).packed()});
    icu.enqueue_sxm_transpose_nop(side,transpose);
    icu.enqueue_sxm_transpose(side,t);
    icu.enqueue_sxm_transpose_repeat(side,S-1,K);
    icu.enqueue_sxm_permute_nop(side,rotate);
    for (std::size_t h = 0; h < S; ++h) {
        icu.enqueue_sxm_permute(side, SxmInstruction::VectorRotate({StreamId::East(20).packed()},
            {StreamId::West(6).packed()}, (S-h)%S));
        icu.enqueue_sxm_permute_repeat(side, K-1);
    }

    const auto out = system->sxm_unit(side).ports().output_column(StreamDirection::West);
    TspSliceSystem::LogSinks sinks{};
    sinks.capture_mem_trace = true;
    std::size_t read_segments = 0, written_segments = 0;
    const auto drain = hw::kHemispheres*system->stream_fabric(side).column_count()+2;
    for (std::size_t cycle=0; cycle<output_read+beats+S+drain; ++cycle) {
        system->tick(sinks);
        const auto& mem = system->mem_array(side);
        for (const auto& transfer : mem.executed_transfers()) {
            if (transfer.kind == MemArrayModel::MemTransfer::Kind::StoreStreamToSram
                && transfer.mem_slice < 2) {
                const auto first = write + transfer.tile;
                check(cycle >= first && cycle < first+beats,"diagonal write cycle");
                const auto beat = cycle-first;
                const auto source_sl = (transfer.tile+beat/K)%S;
                check(transfer.bank == 0 && transfer.address == result_row+(beat%K)*S+source_sl,
                    "wrong destination SL or diagonal write address");
                ++written_segments;
            }
            if (transfer.kind != MemArrayModel::MemTransfer::Kind::LoadSramToStream
                || (transfer.mem_slice != 2 && transfer.mem_slice != 3)) continue;
            const auto first = load + transfer.tile;
            check(cycle >= first && cycle < first + beats, "diagonal read sampling cycle");
            const auto beat = cycle-first;
            const auto row_group = scratch_row + (beat%K)*S;
            const auto phase = (S+transfer.tile-beat/K)%S;
            check(transfer.bank == 0 && transfer.address == row_group+phase,
                "diagonal SRAM row or bank");
            ++read_segments;
        }
        for (std::size_t sl=0; sl<S; ++sl) {
            const auto first=rotate+S+sl;
            const bool active=cycle>=first && cycle<first+beats;
            for (std::size_t p=0; p<2; ++p)
                for (std::size_t lane=0; lane<K; ++lane) {
                    const auto cell=system->stream_fabric(side).cell(out,sl,lane,StreamId::West(6+p));
                    check(cell.valid==active,"unexpected vector Permute output cycle");
                    if (active) {
                        const auto i=cycle-first, src=(sl+i/K)%S;
                        const auto expected=value(sl*K+lane,src,i%K);
                        check(cell.data==static_cast<std::uint8_t>(expected>>(8*p)),
                            "wrong rotated segment or local transpose");
                    }
                    const auto column = mem.ports().output_column(p,StreamDirection::East);
                    const auto data = system->stream_fabric(side).cell(column,sl,lane,StreamId::East(24+p));
                    const auto first_row = output_read+sl;
                    const bool row_valid = cycle>=first_row && cycle<first_row+beats;
                    check(data.valid==row_valid,"dense transpose read cycle");
                    if (row_valid) {
                        // For output (r,c), the reference is exactly input (c,r).
                        const auto r = cycle-first_row;
                        const auto expected = value(sl*K+lane,r/K,r%K);
                        check(data.data==static_cast<std::uint8_t>(expected>>(8*p)),
                            "dense matrix transpose mismatch");
                    }
                }
        }
    }

    check(read_segments == 2*beats*S && written_segments == 2*beats*S,"missing diagonal transfer");
    for (std::size_t i=0; i<beats; ++i)
        for (std::size_t sl=0; sl<S; ++sl)
            for (std::size_t lane=0; lane<K; ++lane)
                for (std::size_t p=0; p<2; ++p) {
                    const auto mid=value(i/K*K+lane,sl,i%K);
                    const auto src=(sl+i/K)%S;
                    const auto expected=value(sl*K+lane,src,i%K);
                    const auto local_row=(i%K)*S+i/K;
                    check(system->read_mem_sram_lane_byte(side,2+p,sl,scratch_row+local_row,lane)
                        ==static_cast<std::uint8_t>(mid>>(8*p)), "intermediate SRAM value");
                    check(system->read_mem_sram_lane_byte(side,p,sl,result_row+(i%K)*S+src,lane)
                        ==static_cast<std::uint8_t>(expected>>(8*p)), "destination SRAM value");
                }
    check(icu.program_queues_done() && system->program_resources_drained(),"system failed to drain");
}

void check_raw_macro(IcuMacroQueueKind kind, const SxmInstruction& op, std::size_t interval)
{
    test::SxmMacroFixture program;
    program.add(20,3,interval,op);
    auto image=program.image(kind==IcuMacroQueueKind::SxmTranspose);
    auto test = [&](auto& queue) {
        queue.load_raw_macro(kind,image);
        for (unsigned tick=0; tick<100 && !queue.raw_macro_ready(); ++tick)
            queue.prefetch_raw_macro();
        check(queue.raw_macro_ready(),"SXM raw macro prime");
        queue.launch();
        for (std::size_t t=0; t<=21+2*interval; ++t) {
            auto issued=queue.tick();
            const bool expected=t==20 || t==20+interval || t==20+2*interval;
            check(issued.has_value()==expected,"raw macro issue cycle");
            if (issued) check(issued->opcode==op.opcode && issued->shift_distance==op.shift_distance
                && issued->src_streams==op.src_streams && issued->dst_streams==op.dst_streams,
                "raw macro changed native operands");
        }
    };
    auto icu=std::make_unique<InstructionControlUnit>();
    if (kind==IcuMacroQueueKind::SxmTranspose) test(icu->sxm_transpose_iq(Hemisphere::East));
    else test(icu->sxm_permute_iq(Hemisphere::East));
}

void check_packets()
{
    const auto store = MemInstruction::DiagonalWrite(result_row+S-1,StreamId::West(6));
    const auto decoded = isa::decode_mem_instruction(isa::encode_mem_instruction(store));
    check(decoded.opcode==store.opcode && decoded.address==store.address && decoded.stream==store.stream,
        "native diagonal write");
    const auto read = MemInstruction::DiagonalRead(scratch_row+S-1,StreamId::West(3));
    const auto native = isa::decode_mem_instruction(isa::encode_mem_instruction(read));
    check(native.opcode == read.opcode && native.address == read.address
        && native.stream == read.stream,"native diagonal read");
    for (auto planes : {1u, 2u}) {
        const auto op = SxmInstruction::VectorRotate({0},{4},S-1,planes);
        const auto raw = isa::decode_sxm_instruction(isa::encode_sxm_instruction(op));
        check(raw.opcode==op.opcode && raw.shift_distance==S-1 && raw.src_streams==op.src_streams,"raw rotate");

    }
}
} // namespace

int main()
try {
    check_packets();
    check_raw_macro(IcuMacroQueueKind::SxmTranspose,SxmInstruction::LocalTranspose({2},{6}),K);
    check_raw_macro(IcuMacroQueueKind::SxmPermute,SxmInstruction::VectorRotate({4},{8},S-1),1);
    // This identity-MXM fixture uses the 4 x 8 MXM stream assignment.
    // Other geometries source the same BF16 vectors directly from MEM.
    run(Hemisphere::East,S == 4 && K == 8);
    run(Hemisphere::West,false);
    std::cout << "local transpose -> SRAM -> vector rotate: byte values, addresses and cycles passed\n";
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
