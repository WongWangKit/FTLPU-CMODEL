#!/usr/bin/env python3
import csv
import sys
from collections import Counter, deque
from pathlib import Path

COMMON_FIELDS = [
    "half_clk", "cycle", "phase", "clk_level", "rst", "producer_transaction_index",
    "producer_valid", "producer_data", "producer_ready", "producer_fire", "producer_stall",
    "producer_pending_count", "producer_accepted_count", "consumer_ready", "tc_a_rd_en",
    "tc_b_wr_en", "tc_b_rd_en", "transfer_invariant_ok",
]
FLAGS = ["full", "empty", "wr_ready", "rd_available"]
CURRENT = ["wr_ptr_q", "rd_ptr_q", "count_q", "rd_data_q", "rd_data_valid_q"]
NEXT = ["wr_ptr_d", "rd_ptr_d", "count_d", "rd_data_d", "rd_data_valid_d"]


def read_trace(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_hex(path, maximum):
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        value_int = int(token, 16)
        if value_int > maximum:
            raise ValueError(f"out-of-range value at {path}:{line_number}")
        result.append(value_int)
    if not result:
        raise ValueError(f"no entries in {path}")
    return result


def value(row, field):
    return row.get(field, "").strip().lower()


def integer(row, field):
    return int(value(row, field))


def compare_field(items, trace_name, index, field, cpp, rtl):
    cpp_value = value(cpp, field)
    rtl_value = value(rtl, field)
    if cpp_value != rtl_value:
        items.append((trace_name, index, field, cpp_value, rtl_value))


def compare_fifo(items, trace_name, index, prefix, cpp, rtl, phase):
    for field in ["wr_fire", "rd_fire", "state_valid"]:
        compare_field(items, trace_name, index, prefix + field, cpp, rtl)
    both_valid = value(cpp, prefix + "state_valid") == "1" and value(rtl, prefix + "state_valid") == "1"
    if both_valid:
        for field in FLAGS + CURRENT:
            compare_field(items, trace_name, index, prefix + field, cpp, rtl)
        for slot in range(4):
            valid_field = f"{prefix}mem{slot}_valid"
            data_field = f"{prefix}mem{slot}"
            compare_field(items, trace_name, index, valid_field, cpp, rtl)
            if value(cpp, valid_field) == "1" and value(rtl, valid_field) == "1":
                compare_field(items, trace_name, index, data_field, cpp, rtl)
    if phase == "low":
        for field in NEXT:
            compare_field(items, trace_name, index, prefix + field, cpp, rtl)


def compare_cpp_rtl(trace_name, cpp_rows, rtl_rows):
    mismatches = []
    if len(cpp_rows) != len(rtl_rows):
        mismatches.append((trace_name, min(len(cpp_rows), len(rtl_rows)), "row_count",
                           str(len(cpp_rows)), str(len(rtl_rows))))
    for index, (cpp, rtl) in enumerate(zip(cpp_rows, rtl_rows)):
        for field in COMMON_FIELDS:
            compare_field(mismatches, trace_name, index, field, cpp, rtl)
        if value(cpp, "tc_b_wr_en") == "1" and value(rtl, "tc_b_wr_en") == "1":
            compare_field(mismatches, trace_name, index, "tc_b_wr_data", cpp, rtl)
        phase = value(cpp, "phase")
        compare_fifo(mismatches, trace_name, index, "a_", cpp, rtl, phase)
        compare_fifo(mismatches, trace_name, index, "b_", cpp, rtl, phase)
    return mismatches


def longest_run(rows, predicate):
    longest = current = 0
    for row in rows:
        if predicate(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def fifo_memory(row, prefix, slot):
    if value(row, f"{prefix}mem{slot}_valid") != "1":
        return None
    return integer(row, f"{prefix}mem{slot}")


def semantic_audit(rows, transactions, controls):
    failures = Counter()
    low_rows = [row for row in rows if value(row, "phase") == "low"]
    high_rows = [row for row in rows if value(row, "phase") == "high"]
    low_by_cycle = {integer(row, "cycle"): row for row in low_rows}
    high_by_cycle = {integer(row, "cycle"): row for row in high_rows}
    if len(low_rows) != len(controls) or len(high_rows) != len(controls):
        failures["control_length"] += 1

    pending_transfer = deque()
    producer_accept_index = 0
    consumer_index = 0
    transfer_latencies = []

    for cycle, low in enumerate(low_rows):
        high = high_by_cycle.get(cycle)
        if high is None:
            failures["missing_high_row"] += 1
            continue
        rst = integer(low, "rst")
        if cycle < len(controls):
            if rst != ((controls[cycle] >> 1) & 1):
                failures["control_reset"] += 1
            if integer(low, "consumer_ready") != (controls[cycle] & 1):
                failures["control_consumer_ready"] += 1

        tx_index = integer(low, "producer_transaction_index")
        valid = integer(low, "producer_valid")
        ready = integer(low, "producer_ready")
        fire = integer(low, "producer_fire")
        stall = integer(low, "producer_stall")
        expected_valid = int(not rst and tx_index < len(transactions))
        expected_data = transactions[tx_index] if expected_valid else 0
        if valid != expected_valid:
            failures["producer_valid_generation"] += 1
        if integer(low, "producer_data") != expected_data:
            failures["producer_data_generation"] += 1
        if fire != int(valid and ready and not rst):
            failures["producer_fire_formula"] += 1
        if stall != int(valid and not ready and not rst):
            failures["producer_stall_formula"] += 1
        if integer(low, "a_wr_fire") != fire:
            failures["producer_to_a_fire"] += 1
        if integer(low, "producer_accepted_count") != producer_accept_index:
            failures["producer_accepted_count_low"] += 1
        if tx_index != producer_accept_index:
            failures["producer_pointer_low"] += 1
        if integer(low, "producer_pending_count") != len(transactions) - producer_accept_index:
            failures["producer_pending_count_low"] += 1
        accepted_after = producer_accept_index + fire
        if integer(high, "producer_transaction_index") != tx_index:
            failures["producer_pointer_high_is_pre_edge"] += 1
        if integer(high, "producer_fire") != fire:
            failures["producer_fire_high_is_pre_edge"] += 1
        if integer(high, "producer_accepted_count") != accepted_after:
            failures["producer_accepted_count_high"] += 1
        if integer(high, "producer_pending_count") != len(transactions) - accepted_after:
            failures["producer_pending_count_high"] += 1
        if fire:
            if producer_accept_index >= len(transactions) or integer(low, "producer_data") != transactions[producer_accept_index]:
                failures["producer_accept_order"] += 1
            producer_accept_index += 1

        next_low = low_by_cycle.get(cycle + 1)
        if next_low is not None:
            expected_next_index = tx_index + fire
            if integer(next_low, "producer_transaction_index") != expected_next_index:
                failures["producer_pointer_advance"] += 1
            if stall and not integer(next_low, "rst"):
                if integer(next_low, "producer_valid") != 1:
                    failures["producer_valid_hold"] += 1
                if integer(next_low, "producer_data") != integer(low, "producer_data"):
                    failures["producer_data_hold"] += 1
                if integer(next_low, "producer_transaction_index") != tx_index:
                    failures["producer_pointer_hold"] += 1

        for prefix in ("a_", "b_"):
            if value(high, prefix + "state_valid") == "1":
                for next_field, current_field in zip(NEXT, CURRENT):
                    if value(low, prefix + next_field) != value(high, prefix + current_field):
                        failures[prefix + "low_next_to_high_current"] += 1
            if integer(high, prefix + "count_q") not in range(5):
                failures[prefix + "count_range"] += 1
            if rst:
                for field in ("wr_ptr_q", "rd_ptr_q", "count_q", "rd_data_q", "rd_data_valid_q"):
                    if value(high, prefix + field) != "0":
                        failures[prefix + "reset_state"] += 1
                for slot in range(4):
                    if value(high, f"{prefix}mem{slot}_valid") != "0":
                        failures[prefix + "reset_memory_valid"] += 1
            elif value(low, prefix + "state_valid") == "1":
                count = integer(low, prefix + "count_q")
                if integer(low, prefix + "full") != (count == 4):
                    failures[prefix + "full_flag"] += 1
                if integer(low, prefix + "empty") != (count == 0):
                    failures[prefix + "empty_flag"] += 1
                wr_fire = integer(low, prefix + "wr_fire")
                rd_fire = integer(low, prefix + "rd_fire")
                if integer(high, prefix + "count_q") != count + wr_fire - rd_fire:
                    failures[prefix + "count_transition"] += 1
                if rd_fire:
                    slot = integer(low, prefix + "rd_ptr_q")
                    expected = fifo_memory(low, prefix, slot)
                    if expected is None or integer(high, prefix + "rd_data_valid_q") != 1 or integer(high, prefix + "rd_data_q") != expected:
                        failures[prefix + "registered_read"] += 1
                elif integer(high, prefix + "rd_data_valid_q") != 0:
                    failures[prefix + "read_valid_clear"] += 1
                for slot in range(4):
                    low_valid = value(low, f"{prefix}mem{slot}_valid")
                    high_valid = value(high, f"{prefix}mem{slot}_valid")
                    write_slot = wr_fire and integer(low, prefix + "wr_ptr_q") == slot
                    if write_slot:
                        source = "producer_data" if prefix == "a_" else "tc_b_wr_data"
                        if high_valid != "1" or integer(high, f"{prefix}mem{slot}") != integer(low, source):
                            failures[prefix + "write_commit"] += 1
                    else:
                        if high_valid != low_valid:
                            failures[prefix + "memory_hold"] += 1
                        if low_valid == "1" and value(high, f"{prefix}mem{slot}") != value(low, f"{prefix}mem{slot}"):
                            failures[prefix + "memory_hold"] += 1

        if value(low, "a_state_valid") == "1" and not rst:
            a_empty = integer(low, "a_empty")
            a_valid = integer(low, "a_rd_data_valid_q")
            b_full = integer(low, "b_full")
            if integer(low, "tc_a_rd_en") != int(not a_empty and not a_valid and not b_full):
                failures["controller_a_rd_en"] += 1
            if integer(low, "tc_b_wr_en") != a_valid:
                failures["controller_b_wr_en"] += 1
            if a_valid and integer(low, "tc_b_wr_data") != integer(low, "a_rd_data_q"):
                failures["controller_b_wr_data"] += 1
            if integer(low, "tc_b_rd_en") != integer(low, "consumer_ready"):
                failures["controller_b_rd_en"] += 1
            if integer(low, "transfer_invariant_ok") != int(not (a_valid and b_full)):
                failures["controller_invariant_signal"] += 1
            if a_valid and b_full:
                failures["pending_response_with_b_full"] += 1

        if integer(low, "b_wr_fire"):
            if not pending_transfer:
                failures["transfer_without_response"] += 1
            else:
                read_cycle, data = pending_transfer.popleft()
                latency = cycle - read_cycle
                transfer_latencies.append(latency)
                if latency != 1:
                    failures["transfer_latency"] += 1
                if integer(low, "tc_b_wr_data") != data:
                    failures["transfer_data_order"] += 1
        if integer(low, "a_rd_fire"):
            slot = integer(low, "a_rd_ptr_q")
            data = fifo_memory(low, "a_", slot)
            if data is None:
                failures["a_read_invalid_memory"] += 1
            else:
                pending_transfer.append((cycle, data))
        if integer(low, "b_rd_fire"):
            if consumer_index >= len(transactions) or integer(high, "b_rd_data_valid_q") != 1 or integer(high, "b_rd_data_q") != transactions[consumer_index]:
                failures["consumer_order"] += 1
            consumer_index += 1

        if next_low is not None:
            for prefix in ("a_", "b_"):
                if value(high, prefix + "state_valid") == "1" and value(next_low, prefix + "state_valid") == "1":
                    for field in CURRENT:
                        if value(high, prefix + field) != value(next_low, prefix + field):
                            failures[prefix + "falling_edge_hold"] += 1
                    for slot in range(4):
                        for suffix in ("_valid", ""):
                            field = f"{prefix}mem{slot}{suffix}"
                            if value(high, field) != value(next_low, field):
                                failures[prefix + "falling_edge_memory_hold"] += 1

    if producer_accept_index != len(transactions):
        failures["not_all_transactions_accepted"] += 1
    if consumer_index != len(transactions):
        failures["not_all_transactions_consumed"] += 1
    if pending_transfer:
        failures["pending_transfer_responses"] += len(pending_transfer)
    return failures, transfer_latencies, producer_accept_index, consumer_index


def coverage(rows, transactions, controls, transfer_latencies, accepted_count, consumer_count):
    low = [row for row in rows if value(row, "phase") == "low"]
    valid_low = [row for row in low if value(row, "a_state_valid") == "1"]
    high = [row for row in rows if value(row, "phase") == "high"]
    last_high = high[-1]
    stats = {
        "producer_transaction_count": len(transactions),
        "producer_valid_cycles": sum(integer(row, "producer_valid") for row in low),
        "producer_ready_cycles": sum(not integer(row, "rst") and integer(row, "producer_ready") for row in low),
        "producer_fire_cycles": sum(integer(row, "producer_fire") for row in low),
        "producer_stall_cycles": sum(integer(row, "producer_stall") for row in low),
        "max_consecutive_producer_stall_cycles": longest_run(low, lambda row: integer(row, "producer_stall") == 1),
        "producer_accepted_count": accepted_count,
        "producer_pending_count": len(transactions) - accepted_count,
        "consumer_received_count": consumer_count,
        "reset_cycles": sum(integer(row, "rst") for row in low),
        "transfer_cycles": sum(integer(row, "b_wr_fire") for row in low),
        "transfer_latency_min": min(transfer_latencies) if transfer_latencies else -1,
        "transfer_latency_max": max(transfer_latencies) if transfer_latencies else -1,
        "final_a_count": integer(last_high, "a_count_q"),
        "final_b_count": integer(last_high, "b_count_q"),
        "final_pending_transfer": integer(last_high, "a_rd_data_valid_q"),
    }
    for prefix in ("a_", "b_"):
        stats[prefix + "full_cycles"] = sum(integer(row, prefix + "full") for row in valid_low)
        stats[prefix + "empty_cycles"] = sum(integer(row, prefix + "empty") for row in valid_low)
        stats[prefix + "accepted_writes"] = sum(integer(row, prefix + "wr_fire") for row in low)
        stats[prefix + "accepted_reads"] = sum(integer(row, prefix + "rd_fire") for row in low)
        stats[prefix + "write_pointer_wraps"] = sum(integer(row, prefix + "wr_fire") and integer(row, prefix + "wr_ptr_q") == 3 for row in valid_low)
        stats[prefix + "read_pointer_wraps"] = sum(integer(row, prefix + "rd_fire") and integer(row, prefix + "rd_ptr_q") == 3 for row in valid_low)
        stats[prefix + "simultaneous_rw"] = sum(integer(row, prefix + "wr_fire") and integer(row, prefix + "rd_fire") for row in low)
    reset_positions = [i for i, control in enumerate(controls) if control & 2]
    checks = {
        "shared_inputs_match_expected_sizes": len(low) == len(controls) and len(transactions) >= 29,
        "initial_reset_only": reset_positions == list(range(len(reset_positions))) and len(reset_positions) >= 1,
        "producer_backpressure_seen": stats["producer_stall_cycles"] > 0,
        "multi_cycle_producer_stall_seen": stats["max_consecutive_producer_stall_cycles"] > 1,
        "all_transactions_accepted": accepted_count == len(transactions),
        "no_pending_transactions": stats["producer_pending_count"] == 0,
        "all_transactions_consumed": consumer_count == len(transactions),
        "a_full_and_empty_seen": stats["a_full_cycles"] > 0 and stats["a_empty_cycles"] > 0,
        "b_full_and_empty_seen": stats["b_full_cycles"] > 0 and stats["b_empty_cycles"] > 0,
        "a_pointer_wraps": stats["a_write_pointer_wraps"] > 0 and stats["a_read_pointer_wraps"] > 0,
        "b_pointer_wraps": stats["b_write_pointer_wraps"] > 0 and stats["b_read_pointer_wraps"] > 0,
        "registered_transfer_latency": bool(transfer_latencies) and all(item == 1 for item in transfer_latencies),
        "fully_drained": stats["final_a_count"] == 0 and stats["final_b_count"] == 0 and stats["final_pending_transfer"] == 0,
    }
    return stats, checks


def main():
    if len(sys.argv) != 8:
        print("usage: compare.py <producer_transactions.hex> <consumer_control.hex> <rtl.csv> <cpp1.csv> <cpp2.csv> <cpp3.csv> <cpp4.csv>", file=sys.stderr)
        return 2
    transactions = read_hex(sys.argv[1], 0xFF)
    controls = read_hex(sys.argv[2], 0x3)
    rtl_rows = read_trace(sys.argv[3])
    cpp_traces = [(Path(path).name, read_trace(path)) for path in sys.argv[4:]]
    reference_name, reference_rows = cpp_traces[0]

    schedule_mismatches = []
    for name, rows in cpp_traces[1:]:
        if rows != reference_rows:
            limit = min(len(rows), len(reference_rows))
            first = next((i for i in range(limit) if rows[i] != reference_rows[i]), limit)
            schedule_mismatches.append((name, first, len(rows), len(reference_rows)))
    mismatches = []
    for name, rows in cpp_traces:
        mismatches.extend(compare_cpp_rtl(name, rows, rtl_rows))

    audits, latencies, accepted_count, consumer_count = semantic_audit(reference_rows, transactions, controls)
    stats, checks = coverage(reference_rows, transactions, controls, latencies, accepted_count, consumer_count)
    audit_failures = sum(audits.values())
    expected_samples = len(controls) * 2
    passed = (not schedule_mismatches and not mismatches and audit_failures == 0 and all(checks.values()) and
              len(rtl_rows) == expected_samples and all(len(rows) == expected_samples for _, rows in cpp_traces))

    print(f"rtl_samples={len(rtl_rows)}")
    for name, rows in cpp_traces:
        print(f"cpp_samples.{name}={len(rows)}")
    print(f"schedule_trace_mismatches={len(schedule_mismatches)}")
    print(f"cpp_rtl_mismatches={len(mismatches)}")
    print(f"producer_protocol_violations={sum(count for key, count in audits.items() if key.startswith('producer_'))}")
    print(f"semantic_audit_failures={audit_failures}")
    print(f"first_schedule_mismatch={schedule_mismatches[0] if schedule_mismatches else 'none'}")
    print(f"first_cpp_rtl_mismatch={mismatches[0] if mismatches else 'none'}")
    if audits:
        for key, count in sorted(audits.items()):
            print(f"semantic_audit.{key}={count}")
    else:
        print("semantic_audit=PASS")
    print("coverage_summary=" + ";".join(f"{key}={val}" for key, val in stats.items()))
    for key, passed_check in checks.items():
        print(f"coverage_check.{key}={'PASS' if passed_check else 'FAIL'}")
    print(f"coverage_result={'PASS' if all(checks.values()) else 'FAIL'}")
    print(f"RESULT={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
