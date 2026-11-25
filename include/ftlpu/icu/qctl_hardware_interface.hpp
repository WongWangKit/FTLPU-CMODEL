#pragma once

#include "ftlpu/icu/queue_control_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftlpu {

// Canonical CModel Hardware Interface View for the RTL QCTL boundary.  These
// are logical cycle signals, not physical SRAM-macro pins.  Fields retain the
// RTL port names so an independent scoreboard can compare one sampled cycle
// without translating aliases.
struct IcuQctlPoolInterfaceView {
    bool rst_ni{true};
    bool clear_i{false};
    bool load_mode_i{false};
    bool partition_valid_i{false};
    bool partition_ready_o{false};
    std::uint32_t partition_queue_i{0};
    std::uint32_t partition_base_i{0};
    std::uint32_t partition_count_i{0};
    std::uint64_t partition_generation_i{0};
    bool load_valid_i{false};
    bool load_ready_o{false};
    std::uint32_t load_address_i{0};
    IcuQueueControl load_data_i{0,static_cast<IcuControlOpcode>(0),0};
    std::vector<bool> request_valid_i{};
    std::vector<bool> request_ready_o{};
    std::vector<std::uint32_t> request_index_i{};
    std::vector<std::uint64_t> request_generation_i{};
    bool response_valid_o{false};
    std::uint32_t response_queue_o{0};
    IcuQueueControl response_data_o{0,static_cast<IcuControlOpcode>(0),0};
    bool response_error_o{false};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

// Canonical view of the current RTL load owner.  The legacy Pool view above
// remains only for source compatibility with old diagnostics; new scoreboards
// use this Load Manager and the Relay/Store views below.
struct IcuQctlLoadManagerInterfaceView {
    bool rst_ni{true};
    bool clear_i{false};
    bool load_mode_i{false};
    bool partition_valid_i{false};
    bool partition_ready_o{false};
    std::uint32_t partition_queue_i{0};
    std::uint32_t partition_base_i{0};
    std::uint32_t partition_count_i{0};
    std::uint64_t partition_generation_i{0};
    bool load_valid_i{false};
    bool load_ready_o{false};
    std::uint32_t load_address_i{0};
    IcuQueueControl load_data_i{};
    bool relay_packet_valid_o{false};
    bool relay_packet_ready_i{false};
    bool relay_packet_kind_o{false};
    std::uint32_t relay_packet_queue_o{0};
    std::uint32_t relay_packet_offset_o{0};
    std::uint32_t relay_packet_count_o{0};
    std::uint64_t relay_packet_generation_o{0};
    IcuQueueControl relay_packet_record_o{};
    bool idle_o{true};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

struct IcuQctlRelayInterfaceView {
    bool chain_packet_valid_i{false};
    bool chain_packet_ready_o{false};
    bool chain_packet_fire{false};
    bool local_fire{false};
    bool next_packet_valid_o{false};
    bool next_packet_ready_i{false};
    std::uint32_t fifo_level{0};
    bool idle_o{true};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

struct IcuQctlCacheInterfaceView {
    bool rst_ni{true};
    bool clear_i{false};
    bool configure_valid_i{false};
    std::uint32_t configure_count_i{0};
    std::uint64_t configure_generation_i{0};
    bool pool_request_valid_o{false};
    bool pool_request_ready_i{false};
    std::uint32_t pool_request_index_o{0};
    std::uint64_t pool_request_generation_o{0};
    bool pool_response_valid_i{false};
    IcuQueueControl pool_response_data_i{};
    bool pool_response_error_i{false};
    bool head_valid_o{false};
    IcuQueueControl head_record_o{};
    bool head_retire_i{false};
    bool configured_o{false};
    bool primed_o{false};
    bool done_o{false};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

struct IcuQctlControllerInterfaceView {
    bool rst_ni{true};
    bool clear_i{false};
    bool configure_valid_i{false};
    bool run_i{false};
    bool configured_i{false};
    bool primed_i{false};
    bool done_i{false};
    std::uint32_t scheduler_cycle_i{0};
    bool scheduler_counter_ce_o{false};
    bool head_valid_i{false};
    IcuQueueControl head_record_i{};
    bool head_retire_o{false};
    bool sync_arrive_valid_o{false};
    bool sync_release_valid_i{false};
    bool notify_valid_o{false};
    bool notify_ready_i{false};
    bool event_valid_i{false};
    std::uint32_t event_tag_i{0};
    bool event_wait_o{false};
    bool event_consume_o{false};
    bool mem_packet_valid_o{false};
    std::uint32_t mem_packet_ordinal_o{0};
    bool mem_packet_retired_i{false};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

struct IcuQctlEndpointInterfaceView {
    bool rst_ni{true};
    bool clear_i{false};
    bool run_i{false};
    bool configure_valid_i{false};
    std::uint32_t configure_count_i{0};
    std::uint64_t configure_generation_i{0};
    std::uint32_t store_head_index_o{0};
    std::uint64_t store_head_generation_o{0};
    bool store_loaded_i{false};
    bool store_head_valid_i{false};
    IcuQueueControl store_head_record_i{};
    bool store_head_error_i{false};
    bool pool_request_valid_o{false};
    bool pool_request_ready_i{false};
    std::uint32_t pool_request_index_o{0};
    std::uint64_t pool_request_generation_o{0};
    bool pool_response_valid_i{false};
    IcuQueueControl pool_response_data_i{};
    bool pool_response_error_i{false};
    std::uint32_t scheduler_cycle_i{0};
    bool scheduler_counter_ce_o{false};
    bool event_valid_i{false};
    std::uint32_t event_tag_i{0};
    bool event_wait_o{false};
    bool event_consume_o{false};
    bool sync_arrive_valid_o{false};
    bool sync_release_valid_i{false};
    bool notify_valid_o{false};
    bool notify_ready_i{false};
    bool mem_packet_valid_o{false};
    std::uint32_t mem_packet_ordinal_o{0};
    bool mem_packet_retired_i{false};
    bool configured_o{false};
    bool primed_o{false};
    bool done_o{false};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

struct IcuQctlEventBufferInterfaceView {
    bool rst_ni{true};
    bool clear_i{false};
    bool event_in_valid_i{false};
    bool event_in_ready_o{false};
    std::uint32_t event_in_tag_i{0};
    std::vector<bool> event_in_target_mask_i{};
    std::vector<bool> event_wait_i{};
    std::vector<bool> event_consume_i{};
    std::vector<bool> event_valid_o{};
    std::uint32_t event_tag_o{0};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

struct IcuQctlFabricInterfaceView {
    bool rst_ni{true};
    bool clear_i{false};
    std::vector<bool> endpoint_pool_request_valid_i{};
    std::vector<bool> endpoint_pool_request_ready_o{};
    std::vector<std::uint32_t> endpoint_pool_request_index_i{};
    std::vector<std::uint64_t> endpoint_pool_request_generation_i{};
    std::vector<bool> pool_request_valid_o{};
    std::vector<bool> pool_request_ready_i{};
    std::vector<std::uint32_t> pool_request_index_o{};
    std::vector<std::uint64_t> pool_request_generation_o{};
    bool pool_response_valid_i{false};
    std::uint32_t pool_response_queue_i{0};
    IcuQueueControl pool_response_data_i{};
    bool pool_response_error_i{false};
    std::vector<bool> endpoint_pool_response_valid_o{};
    IcuQueueControl endpoint_pool_response_data_o{};
    bool endpoint_pool_response_error_o{false};
    bool event_in_valid_i{false};
    bool event_in_ready_i{false};
    bool event_backpressure_o{false};
    std::vector<bool> endpoint_event_wait_i{};
    std::vector<bool> endpoint_event_consume_i{};
    std::vector<bool> event_wait_o{};
    std::vector<bool> event_consume_o{};
    std::vector<bool> event_buffer_valid_i{};
    std::uint32_t event_buffer_tag_i{0};
    std::vector<bool> endpoint_event_valid_o{};
    std::uint32_t endpoint_event_tag_o{0};
    std::vector<bool> sync_participant_mask_i{};
    std::vector<bool> endpoint_sync_arrive_valid_i{};
    std::vector<bool> endpoint_notify_valid_i{};
    std::vector<bool> endpoint_notify_ready_o{};
    std::vector<bool> endpoint_sync_release_valid_o{};
    std::vector<bool> endpoint_mem_packet_valid_i{};
    std::vector<std::uint32_t> endpoint_mem_packet_ordinal_i{};
    std::vector<bool> mem_packet_valid_o{};
    std::vector<std::uint32_t> mem_packet_ordinal_o{};
    std::vector<bool> mem_packet_retired_i{};
    std::vector<bool> endpoint_mem_packet_retired_o{};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

struct IcuQctlGlobalInterfaceView {
    // Formal lpu_qctl_global_subsystem ports.  Keep these aliases even though
    // the same signals also appear at the nested Pool/Fabric views: the
    // top-level aliases make the joint scoreboard independent of C++ object
    // hierarchy.
    bool rst_ni{true};
    bool clear_i{false};
    bool load_mode_i{false};
    bool partition_valid_i{false};
    bool partition_ready_o{false};
    std::uint32_t partition_queue_i{0};
    std::uint32_t partition_base_i{0};
    std::uint32_t partition_count_i{0};
    std::uint64_t partition_generation_i{0};
    bool load_valid_i{false};
    bool load_ready_o{false};
    std::uint32_t load_address_i{0};
    IcuQueueControl load_data_i{};
    IcuQctlLoadManagerInterfaceView load_manager{};
    std::vector<IcuQctlRelayInterfaceView> relays{};
    IcuQctlPoolInterfaceView pool{};
    IcuQctlEventBufferInterfaceView event_buffer{};
    IcuQctlFabricInterfaceView fabric{};
    std::vector<IcuQctlCacheInterfaceView> caches{};
    std::vector<IcuQctlControllerInterfaceView> controllers{};
    std::vector<IcuQctlEndpointInterfaceView> endpoints{};
    std::vector<bool> endpoint_configure_valid_o{};
    std::vector<std::uint32_t> endpoint_configure_count_o{};
    std::vector<std::uint64_t> endpoint_configure_generation_o{};
    std::vector<bool> endpoint_store_loaded_o{};
    std::vector<bool> endpoint_head_valid_o{};
    std::vector<IcuQueueControl> endpoint_head_record_o{};
    std::vector<bool> endpoint_head_error_o{};
    std::vector<bool> endpoint_configured_i{};
    std::vector<bool> endpoint_primed_i{};
    std::vector<bool> endpoint_done_i{};
    std::vector<bool> endpoint_fault_i{};
    std::vector<std::uint8_t> endpoint_fault_code_i{};
    bool event_in_valid_i{false};
    bool event_in_ready_o{false};
    std::uint32_t event_in_tag_i{0};
    std::vector<bool> event_in_target_mask_i{};
    bool event_backpressure_o{false};
    std::vector<bool> endpoint_event_wait_i{};
    std::vector<bool> endpoint_event_consume_i{};
    std::vector<bool> endpoint_event_valid_o{};
    std::uint32_t endpoint_event_tag_o{0};
    std::vector<bool> sync_participant_mask_i{};
    std::vector<bool> endpoint_sync_arrive_valid_i{};
    std::vector<bool> endpoint_notify_valid_i{};
    std::vector<bool> endpoint_notify_ready_o{};
    std::vector<bool> endpoint_sync_release_valid_o{};
    std::vector<bool> endpoint_mem_packet_valid_i{};
    std::vector<std::uint32_t> endpoint_mem_packet_ordinal_i{};
    std::vector<bool> mem_packet_valid_o{};
    std::vector<std::uint32_t> mem_packet_ordinal_o{};
    std::vector<bool> mem_packet_retired_i{};
    std::vector<bool> endpoint_mem_packet_retired_o{};
    std::vector<bool> queue_active_o{};
    bool load_complete_o{false};
    bool all_primed_o{false};
    bool all_done_o{false};
    bool fault_o{false};
    std::uint8_t fault_code_o{0};
};

} // namespace ftlpu
