#include "ftlpu/routing/control_unit.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "ftlpu/core/bf16.hpp"
#include <iostream>
#include <memory>
#include <numeric>

using namespace ftlpu;
static void require(bool b, const char *m) {
  if (!b)
    throw std::runtime_error(m);
}
template <class F> static void rejects(F f) {
  bool hit = false;
  try {
    f();
  } catch (const std::exception &) {
    hit = true;
  }
  require(hit, "expected rejection");
}
struct Rig {
  std::array<MemArrayModel, hw::kHemispheres> mem;
  std::array<StreamRegisterFabric, hw::kHemispheres> sr{
      StreamRegisterFabric(hw::kMemBoundaryStreamRegisterColumns),
      StreamRegisterFabric(hw::kMemBoundaryStreamRegisterColumns)};
  RoutingControlUnit rcu;
  std::uint64_t cycle{};
  explicit Rig(std::size_t latency = 1) { rcu.configure({true, latency}); }
  void tick() {
    for (auto &f : sr)
      f.begin_cycle();
    rcu.tick(mem, sr);
    for (std::size_t h = 0; h < mem.size(); ++h) {
      mem[h].evaluate(sr[h]);
      sr[h].commit_cycle();
    }
    ++cycle;
  }
  RoutingCompletion run(RoutingDescriptor d, std::uint32_t id) {
    rcu.start({d.opcode, 0, id}, d, 1);
    for (std::size_t c = 0; c < 2000000; ++c) {
      tick();
      if (auto done = rcu.take_completion())
        return *done;
    }
    throw std::runtime_error("RCU timeout");
  }
  void put(const RoutingTensorView &v, std::size_t r, std::size_t c,
           std::uint32_t value) {
    for (std::size_t b = 0; b < v.bytes(); ++b) {
      auto p = v.planes[b];
      mem[p.hemisphere].set_sram_byte(
          p.slice, p.bank, v.row(r, c, b), c % hw::kPhysicalVectorBytes,
          static_cast<std::uint8_t>(value >> (8 * b)));
    }
  }
  std::uint32_t get(const RoutingTensorView &v, std::size_t r, std::size_t c) {
    std::uint32_t x = 0;
    for (std::size_t b = 0; b < v.bytes(); ++b) {
      auto p = v.planes[b];
      x |= std::uint32_t(mem[p.hemisphere].sram_byte(
               p.slice, p.bank, v.row(r, c, b), c % hw::kPhysicalVectorBytes))
           << (8 * b);
    }
    return x;
  }
};
static RoutingTensorView view(RoutingElement f, std::uint32_t rows,
                              std::uint32_t cols, std::uint32_t slice,
                              std::uint32_t base = 0, std::uint32_t h = 0,
                              std::uint32_t bank = 0) {
  RoutingTensorView v;
  v.element = f;
  v.rows = rows;
  v.columns = cols;
  v.feature_stride = rows;
  for (std::size_t p = 0; p < v.bytes(); ++p)
    v.planes[p] = {h, slice + static_cast<std::uint32_t>(p), bank, base};
  return v;
}
static void scenario(std::size_t latency) {
  auto rig = std::make_unique<Rig>(latency);
  auto &r = *rig;
  constexpr std::uint32_t T = 7, E = 5, K = 3, H = 64;
  RoutingDescriptor select;
  auto &s = select.select;
  s.tokens = T;
  s.experts = E;
  s.k = K;
  s.choice_scores = view(RoutingElement::F32, T, E, 0);
  s.raw_scores = view(RoutingElement::F32, T, E, 4);
  s.eligibility = view(RoutingElement::U32, T, E, 8);
  s.selected_ids = view(RoutingElement::U32, T, K, 12);
  s.selected_scores = view(RoutingElement::F32, T, K, 16);
  std::vector<std::uint32_t> expected, raw, counts(E), active;
  for (std::size_t t = 0; t < T; ++t) {
    std::vector<std::uint32_t> order(E);
    std::iota(order.begin(), order.end(), 0);
    const auto score = [&](auto e) {
      return static_cast<float>((t + e * 2) % 5);
    };
    std::erase_if(order, [&](auto e) { return e == 4 && (t & 1); });
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
      return score(a) == score(b) ? a < b : score(a) > score(b);
    });
    for (std::size_t e = 0; e < E; ++e) {
      r.put(s.choice_scores, t, e, std::bit_cast<std::uint32_t>(score(e)));
      r.put(s.raw_scores, t, e,
            std::bit_cast<std::uint32_t>(-0.125f *
                                         static_cast<float>(1 + t * E + e)));
      r.put(s.eligibility, t, e, e == 4 && (t & 1) ? 0 : 1);
    }
    for (std::size_t k = 0; k < K; ++k) {
      auto e = order[k];
      expected.push_back(e);
      raw.push_back(r.get(s.raw_scores, t, e));
      if (!counts[e]++)
        active.push_back(e);
    }
  }
  auto descriptors = std::vector<RoutingDescriptor>{select};
  auto words = encode_routing_descriptors(descriptors);
  require(encode_routing_descriptors(decode_routing_descriptors(words)) ==
              words,
          "descriptor roundtrip");
  auto instruction = RoutingInstruction{RoutingOpcode::Select, 0, 1};
  require(decode_routing_instruction(encode_routing_instruction(instruction))
                  .completion == 1,
          "instruction codec");
  rejects([&] {
    auto bad = encode_routing_instruction(instruction);
    bad.lanes[0] |= 0x80000000u;
    decode_routing_instruction(bad);
  });
  auto done = r.run(select, 1);
  require(done.instance == 1 && done.opcode == RoutingOpcode::Select,
          "SELECT completion");
  require(r.rcu.selected_experts() == expected, "TopK expert IDs");
  require(r.rcu.selected_score_bits() == raw, "raw score bits");
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t k = 0; k < K; ++k) {
      require(r.get(s.selected_ids, t, k) == expected[t * K + k],
              "ID export before completion");
      require(r.get(s.selected_scores, t, k) == raw[t * K + k],
              "score export before completion");
    }
  auto weights = view(RoutingElement::F32, T, K, 20);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t k = 0; k < K; ++k)
      r.put(weights, t, k, 0x3e800000u + static_cast<std::uint32_t>(t * K + k));
  RoutingDescriptor plan;
  plan.opcode = RoutingOpcode::Plan;
  plan.plan.weights = weights;
  r.run(plan, 2);
  require(r.rcu.counts() == counts && r.rcu.active_experts() == active,
          "count/active order");
  require(r.rcu.offsets().back() == T * K, "count conservation");
  for (std::size_t e = 0; e < E; ++e) {
    auto p = r.rcu.offsets()[e];
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t k = 0; k < K; ++k)
        if (expected[t * K + k] == e) {
          auto a = r.rcu.assignments()[p++];
          require(a.token == t && a.slot == k &&
                      a.weight_bits == r.get(weights, t, k),
                  "stable assignment/opaque final weight");
        }
    require(p == r.rcu.offsets()[e + 1], "CSR segment");
  }
  auto src = view(RoutingElement::BF16, T, H, 24, 64, 0, 0);
  auto dst = view(RoutingElement::BF16, 2, H, 26, 128, 1, 1);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t c = 0; c < H; ++c)
      r.put(src, t, c, static_cast<std::uint32_t>(t * 100 + c));
  RoutingDescriptor address;
  address.opcode = RoutingOpcode::Address;
  address.address.max_rows = 2;
  address.address.source = src;
  address.address.destination = dst;
  r.run(address, 3);
  const auto block = r.rcu.block_info();
  require(block.expert_id == active.front() &&
              block.valid_rows == std::min(2u, counts[active.front()]),
          "block info");
  for (std::size_t t = 0; t < block.valid_rows; ++t) {
    auto token =
        r.rcu.assignments()[r.rcu.offsets()[block.expert_id] + t].token;
    for (std::size_t c = 0; c < H; ++c)
      require(r.get(dst, t, c) == r.get(src, token, c),
              "real cross-hemisphere Gather payload");
  }
  require(r.rcu.statistics().gathered_vectors ==
              block.valid_rows * (H / hw::kPhysicalVectorBytes) * 2,
          "Gather vector count");
  require(r.rcu.statistics().commit_segments ==
              r.rcu.statistics().gathered_vectors * hw::kTileRows,
          "physical SRAM commits");
  address.address.row_offset = 1;
  r.run(address, 5);
  const auto middle = r.rcu.block_info();
  require(middle.valid_rows == std::min(2u, counts[middle.expert_id] - 1) &&
              middle.more == (counts[middle.expert_id] > 3),
          "middle block flags");
  for (std::size_t row = 0; row < middle.valid_rows; ++row) {
    const auto token =
        r.rcu.assignments()[r.rcu.offsets()[middle.expert_id] + 1 + row].token;
    for (std::size_t c = 0; c < H; ++c)
      require(r.get(dst, row, c) == r.get(src, token, c),
              "middle block used wrong assignment");
  }
  const auto before = r.rcu.statistics().gathered_vectors;
  address.address.row_offset = 100;
  r.run(address, 4);
  require(r.rcu.block_info().valid_rows == 0 &&
              r.rcu.statistics().gathered_vectors == before,
          "empty block touched data");
  require(r.rcu.statistics().peak_addresses <= RoutingConfig::address_depth,
          "finite FIFO");
  require(r.cycle > r.rcu.statistics().comparisons, "missing physical latency");
  std::cout << "latency=" << latency << " cycles=" << r.cycle
            << " comparisons=" << r.rcu.statistics().comparisons
            << " metadata=" << r.rcu.statistics().metadata_accesses
            << " io=" << r.rcu.statistics().control_requests
            << " bank_wait=" << r.rcu.statistics().bank_wait_cycles
            << " address_wait=" << r.rcu.statistics().address_wait_cycles
            << " sr_wait=" << r.rcu.statistics().stream_wait_cycles
            << " fifo_peak=" << r.rcu.statistics().peak_addresses << '\n';
  auto bad = select;
  bad.select.k = 17;
  rejects([&] { r.run(bad, 5); });
  bad = address;
  bad.address.destination = src;
  rejects([&] { r.run(bad, 6); });
}
static void limits_and_same_bank() {
  for (auto geometry : std::array<std::array<std::uint32_t, 3>, 3>{
           {{1, 16, 16}, {1, 1024, 1}, {128, 1, 1}}}) {
    auto r = std::make_unique<Rig>();
    auto [t, e, k] = geometry;
    RoutingDescriptor d;
    d.select.tokens = t;
    d.select.experts = e;
    d.select.k = k;
    d.select.choice_scores = view(RoutingElement::F32, t, e, 0);
    for (std::size_t row = 0; row < t; ++row)
      for (std::size_t col = 0; col < e; ++col)
        r->put(d.select.choice_scores, row, col,
               std::bit_cast<std::uint32_t>(
                   e == 16 ? std::numeric_limits<float>::infinity()
                           : static_cast<float>(col)));
    r->run(d, 1);
    require(r->rcu.statistics().comparisons == std::uint64_t(t) * e * k,
            "comparator throughput is not one comparison/clock");
    for (std::size_t row = 0; row < t; ++row)
      for (std::size_t j = 0; j < k; ++j)
        require(r->rcu.selected_experts()[row * k + j] ==
                    (e == 16 ? j : e - 1 - j),
                "boundary/tie TopK");
    RoutingDescriptor plan;
    plan.opcode = RoutingOpcode::Plan;
    r->run(plan, 2);
    require(r->rcu.offsets().back() == t * k, "boundary count conservation");
  }
  auto r = std::make_unique<Rig>();
  RoutingDescriptor d;
  d.select.tokens = 4;
  d.select.experts = d.select.k = 1;
  d.select.choice_scores = view(RoutingElement::F32, 4, 1, 0);
  r->run(d, 1);
  RoutingDescriptor p;
  p.opcode = RoutingOpcode::Plan;
  r->run(p, 2);
  for (auto type : {RoutingElement::F16, RoutingElement::BF16}) {
    auto src = view(type, 4, 32, 24, 64, 0, 1),
         dst = view(type, 3, 32, 24, 256, 0, 1);
    for (std::size_t t = 0; t < 4; ++t)
      for (std::size_t c = 0; c < 32; ++c)
        r->put(src, t, c, static_cast<std::uint32_t>(t * 100 + c));
    for (std::size_t t = 0; t < 3; ++t)
      for (std::size_t c = 0; c < 32; ++c)
        r->put(dst, t, c, 0xbeef);
    RoutingDescriptor a;
    a.opcode = RoutingOpcode::Address;
    a.address.source = src;
    a.address.destination = dst;
    a.address.row_offset = 3;
    a.address.max_rows = 3;
    r->run(a, type == RoutingElement::F16 ? 3 : 4);
    require(r->rcu.block_info().valid_rows == 1 && !r->rcu.block_info().more,
            "tail block length");
    for (std::size_t c = 0; c < 32; ++c) {
      require(r->get(dst, 0, c) == 300 + c, "same single-port bank Gather");
      require(r->get(dst, 1, c) == 0xbeef && r->get(dst, 2, c) == 0xbeef,
              "Gather wrote padding rows");
    }
  }
  for (bool mask : {false, true}) {
    auto bad = std::make_unique<Rig>();
    RoutingDescriptor x;
    x.select.tokens = 1;
    x.select.experts = 2;
    x.select.k = 2;
    x.select.choice_scores = view(RoutingElement::F32, 1, 2, 0);
    if (mask)
      x.select.eligibility = view(RoutingElement::U32, 1, 2, 4);
    else
      bad->put(x.select.choice_scores, 0, 0, 0x7fc00001u);
    rejects([&] { bad->run(x, 1); });
    require(bad->rcu.busy(), "faulted task silently became complete");
  }
}
static void ownership_and_backpressure() {
  auto r = std::make_unique<Rig>();
  RoutingDescriptor s;
  s.select.tokens = 4;
  s.select.experts = s.select.k = 1;
  s.select.choice_scores = view(RoutingElement::F32, 4, 1, 0);
  r->mem[0].enqueue_instruction(3, 0,
                                MemInstruction::Read(0, StreamId::East(0)));
  r->rcu.start({RoutingOpcode::Select, 0, 1}, s, 1);
  r->tick();
  require(!r->mem[0].routing_bank_owned(0, 0),
          "partially granted a multibank lease");
  for (std::size_t n = 0; !r->rcu.idle() && n < 10000; ++n) {
    r->tick();
    r->rcu.take_completion();
  }
  require(r->rcu.idle() && r->rcu.statistics().bank_wait_cycles > 0,
          "bank drain was not modeled");
  RoutingDescriptor plan;
  plan.opcode = RoutingOpcode::Plan;
  r->run(plan, 2);
  RoutingDescriptor a;
  a.opcode = RoutingOpcode::Address;
  a.address.source = view(RoutingElement::BF16, 4, 64, 24, 64);
  a.address.destination = view(RoutingElement::BF16, 4, 64, 26, 128, 1, 1);
  a.address.max_rows = 4;
  for (std::size_t t = 0; t < 4; ++t)
    for (std::size_t c = 0; c < 64; ++c)
      r->put(a.address.source, t, c, static_cast<std::uint32_t>(100 * t + c));
  r->rcu.start({RoutingOpcode::Address, 0, 3}, a, 1);
  rejects([&] { r->rcu.reset_program(); });
  std::size_t elapsed = 0;
  bool complete = false;
  for (; elapsed < 20000; ++elapsed) {
    for (auto &f : r->sr)
      f.begin_cycle();
    const auto occupy = [&](std::size_t h, std::size_t column) {
      StreamOutputPort out(r->sr[h], column, StreamDirection::East,
                           "backpressure fixture");
      for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        out.write_payload_segment(tile, 31, {}, 0xfffe);
    };
    if (elapsed < 40)
      occupy(0, r->mem[0].ports().output_column(24, StreamDirection::East));
    if (elapsed < 80)
      occupy(1, r->mem[1].ports().input_column(26, StreamDirection::East));
    r->rcu.tick(r->mem, r->sr);
    for (std::size_t h = 0; h < r->mem.size(); ++h) {
      r->mem[h].evaluate(r->sr[h]);
      r->sr[h].commit_cycle();
    }
    ++r->cycle;
    if (elapsed < 80)
      require(r->rcu.statistics().commit_segments == 0,
              "write consumed an uncredited/late burst");
    if (auto done = r->rcu.take_completion()) {
      require(done->instance == 3 && elapsed >= 80,
              "early INPUT_READY under backpressure");
      complete = true;
      break;
    }
  }
  require(complete && r->rcu.statistics().stream_wait_cycles >= 40,
          "SR backpressure was bypassed");
  require(r->rcu.statistics().peak_addresses == RoutingConfig::address_depth,
          "address FIFO never exercised full credit");
  for (std::size_t t = 0; t < 4; ++t)
    for (std::size_t c = 0; c < 64; ++c)
      require(r->get(a.address.destination, t, c) == 100 * t + c,
              "lost/repeated Gather under backpressure");
  require(r->rcu.statistics().address_wait_cycles > 0,
          "full address FIFO was not modeled as a credit stall");
  std::cout << "stress cycles=" << r->cycle
            << " comparisons=" << r->rcu.statistics().comparisons
            << " metadata=" << r->rcu.statistics().metadata_accesses
            << " io=" << r->rcu.statistics().control_requests
            << " bank_wait=" << r->rcu.statistics().bank_wait_cycles
            << " address_wait=" << r->rcu.statistics().address_wait_cycles
            << " sr_wait=" << r->rcu.statistics().stream_wait_cycles
            << " fifo_peak=" << r->rcu.statistics().peak_addresses
            << " commits=" << r->rcu.statistics().commit_segments << '\n';
  require(r->rcu.trace().size() == 3 && r->rcu.trace().back().end_cycle >
                                            r->rcu.trace().back().begin_cycle,
          "bounded command trace missing");
}
static void combine_case(bool half=false) {
  const auto encode=[&](float x){return half?VxmDataFormat::float_to_fp16_bits(x):Bf16::from_float(x).bits();};
  const auto decode=[&](std::uint16_t x){return half?VxmDataFormat::fp16_bits_to_float(x):Bf16::from_bits(x).to_float();};
  SystemHardwareConfiguration cfg;cfg.routing.enabled=true;
  auto chip=std::make_unique<TspSliceSystem>(cfg);
  const auto put=[&](const RoutingTensorView &v,unsigned r,unsigned c,std::uint32_t bits) {
    for(unsigned p=0;p<v.bytes();++p) {const auto q=v.planes[p];chip->mem_array(static_cast<Hemisphere>(q.hemisphere)).set_sram_byte(q.slice,q.bank,v.row(r,c,p),c%32,static_cast<std::uint8_t>(bits>>(8*p)));}
  };
  const auto get=[&](const RoutingTensorView &v,unsigned r,unsigned c) {
    std::uint32_t bits=0;for(unsigned p=0;p<v.bytes();++p){const auto q=v.planes[p];bits|=std::uint32_t(chip->mem_array(static_cast<Hemisphere>(q.hemisphere)).sram_byte(q.slice,q.bank,v.row(r,c,p),c%32))<<(8*p);}return bits;
  };
  RoutingDescriptor select;select.select.tokens=2;select.select.experts=2;select.select.k=2;
  select.select.choice_scores=view(RoutingElement::F32,2,2,20);
  RoutingDescriptor plan;plan.opcode=RoutingOpcode::Plan;plan.plan.weights=view(RoutingElement::F32,2,2,24);
  for(unsigned t=0;t<2;++t)for(unsigned s=0;s<2;++s) {
    put(select.select.choice_scores,t,s,std::bit_cast<std::uint32_t>(t==s?2.f:1.f));
    put(plan.plan.weights,t,s,std::bit_cast<std::uint32_t>(s==0?0.1f:0.9f));
  }
  for(const auto &d:{select,plan}) {
    chip->routing().start({d.opcode,0,1},d,1);
    for(unsigned n=0;!chip->routing().idle();++n){require(n<10000,"combine setup timeout");chip->tick({});}
  }
  RouteCombineDescriptor d;d.source=view(half?RoutingElement::F16:RoutingElement::BF16,4,64,0,0,1);
  d.destination=view(d.source.element,2,64,10,0,0,1);
  for(unsigned j=0;j<4;++j) {
    const auto a=chip->routing().assignments()[j];
    for(unsigned c=0;c<64;++c)put(d.source,j,c,encode(a.token+1.f+a.slot*0.0625f+c/256.f));
  }
  auto sentinel=d.destination;sentinel.rows=1;sentinel.columns=32;
  for(auto &p:sentinel.planes)p.base_row+=4;
  for(unsigned c=0;c<32;++c)put(sentinel,0,c,0xbeef);
  auto &controller=chip->routing_combine();
  rejects([&]{controller.start({RoutingOpcode::Combine,0,2},d,1,chip->routing());});
  controller.publish_outputs(1,chip->routing().plan_epoch(),1,d.source);
  rejects([&]{controller.start({RoutingOpcode::Combine,0,2},d,2,chip->routing());});
  auto invalid=d;invalid.destination=d.source;
  rejects([&]{validate_route_combine(invalid);});
  require(chip->mem_array(Hemisphere::East).acquire_routing_bank(10,1,777),"combine backpressure setup");
  controller.start({RoutingOpcode::Combine,0,2},d,1,chip->routing());
  for(unsigned n=0;n<20;++n)chip->tick({});
  require(!controller.idle() && controller.statistics().committed_segments==0,"combine completed during bank backpressure");
  chip->mem_array(Hemisphere::East).release_routing_bank(10,1,777);
  for(unsigned n=0;!controller.idle();++n){require(n<20000,"combine pipeline timeout");require(!chip->can_load_program(),"program switch during combine");chip->tick({});}
  for(unsigned t=0;t<2;++t)for(unsigned c=0;c<64;++c) {
    const float first=decode(encode(t+1.f+c/256.f))*0.1f;
    const float second=decode(encode(t+1.f+0.0625f+c/256.f))*0.9f;
    require(get(d.destination,t,c)==encode(first+second),"VXM weighted combine result mismatch");
    require(get(sentinel,0,c%32)==0xbeef,"combine tail overwrite");
  }
  require(controller.statistics().contributions==8 && controller.statistics().committed_segments==32 && controller.statistics().bank_wait>0,"combine accounting");
  rejects([&]{controller.start({RoutingOpcode::Combine,0,2},d,1,chip->routing());});
  std::cout<<"combine "<<(half?"FP16":"BF16")<<" K2 H64 cycles="<<controller.statistics().cycles<<" exact\n";
}
int main() {
  try {
    scenario(1);
    scenario(3);
    limits_and_same_bank();
    ownership_and_backpressure();
    combine_case();
    combine_case(true);
    std::cout << "routing control passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
