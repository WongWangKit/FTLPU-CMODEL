#!/usr/bin/env python3
import csv
import sys
from collections import Counter, deque
from pathlib import Path


COMMON_FIELDS = [
    "half_clk", "cycle", "phase", "clk_level", "rst", "producer_valid",
    "producer_data", "consumer_ready", "tc_a_rd_en", "tc_b_wr_en",
    "tc_b_rd_en", "producer_ready", "transfer_invariant_ok",
]
FLAGS = ["full", "empty", "wr_ready", "rd_available"]
CURRENT = ["wr_ptr_q", "rd_ptr_q", "count_q", "rd_data_q", "rd_data_valid_q"]
NEXT = ["wr_ptr_d", "rd_ptr_d", "count_d", "rd_data_d", "rd_data_valid_d"]


def read_trace(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def value(row, field):
    return row.get(field, "").strip().lower()


def integer(row, field):
    return int(value(row, field))


def add_mismatch(items, trace_name, index, field, cpp_value, rtl_value):
    items.append((trace_name, index, field, cpp_value, rtl_value))


def compare_field(items, trace_name, index, field, cpp, rtl):
    cpp_value = value(cpp, field)
    rtl_value = value(rtl, field)
    if cpp_value != rtl_value:
        add_mismatch(items, trace_name, index, field, cpp_value, rtl_value)


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
        add_mismatch(mismatches, trace_name, min(len(cpp_rows), len(rtl_rows)),
                     "row_count", str(len(cpp_rows)), str(len(rtl_rows)))

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
    longest = 0
    current = 0
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


def semantic_audit(rows):
    failures = Counter()
    low_rows = [row for row in rows if value(row, "phase") == "low"]
    high_rows = [row for row in rows if value(row, "phase") == "high"]
    high_by_cycle = {integer(row, "cycle"): row for row in high_rows}
    low_by_cycle = {integer(row, "cycle"): row for row in low_rows}

    for low in low_rows:
        cycle = integer(low, "cycle")
        high = high_by_cycle[cycle]
        rst = integer(low, "rst")

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
                expected_count = count + wr_fire - rd_fire
                if integer(high, prefix + "count_q") != expected_count:
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
                        expected_data = integer(low, "producer_data" if prefix == "a_" else "tc_b_wr_data")
                        if high_valid != "1" or integer(high, f"{prefix}mem{slot}") != expected_data:
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
            if integer(low, "tc_a_rd_en") != ((not a_empty) and (not a_valid) and (not b_full)):
                failures["controller_a_rd_en"] += 1
            if integer(low, "tc_b_wr_en") != bool(a_valid):
                failures["controller_b_wr_en"] += 1
            if a_valid and integer(low, "tc_b_wr_data") != integer(low, "a_rd_data_q"):
                failures["controller_b_wr_data"] += 1
            if integer(low, "tc_b_rd_en") != bool(integer(low, "consumer_ready")):
                failures["controller_b_rd_en"] += 1
            if integer(low, "transfer_invariant_ok") != (not (a_valid and b_full)):
                failures["controller_invariant_signal"] += 1
            if a_valid and b_full:
                failures["pending_response_with_b_full"] += 1

        next_low = low_by_cycle.get(cycle + 1)
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
    return failures


def coverage_and_scoreboard(rows):
    low = [row for row in rows if value(row, "phase") == "low"]
    high_by_cycle = {integer(row, "cycle"): row for row in rows if value(row, "phase") == "high"}
    valid_low = [row for row in low if value(row, "a_state_valid") == "1"]
    stats = {}

    for prefix in ("a_", "b_"):
        stats[prefix + "full_cycles"] = sum(integer(row, prefix + "full") for row in valid_low)
        stats[prefix + "empty_cycles"] = sum(integer(row, prefix + "empty") for row in valid_low)
        stats[prefix + "accepted_writes"] = sum(integer(row, prefix + "wr_fire") for row in low)
        stats[prefix + "accepted_reads"] = sum(integer(row, prefix + "rd_fire") for row in low)
        wr_en_field = "producer_valid" if prefix == "a_" else "tc_b_wr_en"
        rd_en_field = "tc_a_rd_en" if prefix == "a_" else "tc_b_rd_en"
        stats[prefix + "rejected_writes"] = sum(
            not integer(row, "rst") and integer(row, wr_en_field) and not integer(row, prefix + "wr_fire") for row in low)
        stats[prefix + "rejected_reads"] = sum(
            not integer(row, "rst") and integer(row, rd_en_field) and not integer(row, prefix + "rd_fire") for row in low)
        stats[prefix + "write_pointer_wraps"] = sum(
            integer(row, prefix + "wr_fire") and integer(row, prefix + "wr_ptr_q") == 3 for row in valid_low)
        stats[prefix + "read_pointer_wraps"] = sum(
            integer(row, prefix + "rd_fire") and integer(row, prefix + "rd_ptr_q") == 3 for row in valid_low)
        stats[prefix + "simultaneous_rw"] = sum(
            integer(row, prefix + "wr_fire") and integer(row, prefix + "rd_fire") for row in low)

    stats["transfer_cycles"] = sum(integer(row, "b_wr_fire") for row in low)
    stats["transfer_blocked_cycles"] = sum(
        not integer(row, "rst") and not integer(row, "a_empty") and
        not integer(row, "a_rd_data_valid_q") and integer(row, "b_full") for row in valid_low)
    stats["a_stopped_by_b_full_cycles"] = stats["transfer_blocked_cycles"]
    stats["producer_blocked_cycles"] = sum(
        not integer(row, "rst") and integer(row, "producer_valid") and not integer(row, "a_wr_fire") for row in low)
    stats["consumer_received"] = sum(integer(row, "b_rd_fire") for row in low)
    stats["reset_cycles"] = sum(integer(row, "rst") for row in low)
    stats["max_consecutive_producer_backpressure"] = longest_run(
        low, lambda row: not integer(row, "rst") and integer(row, "producer_valid") and not integer(row, "a_wr_fire"))

    expected = deque()
    order_errors = 0
    pending_reads = deque()
    transfer_latencies = []
    for row in low:
        cycle = integer(row, "cycle")
        if integer(row, "rst"):
            expected.clear()
            pending_reads.clear()
            continue
        if integer(row, "a_wr_fire"):
            expected.append(integer(row, "producer_data"))
        if integer(row, "a_rd_fire"):
            pending_reads.append(cycle)
        if integer(row, "b_wr_fire"):
            if not pending_reads:
                order_errors += 1
            else:
                transfer_latencies.append(cycle - pending_reads.popleft())
        if integer(row, "b_rd_fire"):
            high = high_by_cycle[cycle]
            if not expected:
                order_errors += 1
            else:
                expected_value = expected.popleft()
                if integer(high, "b_rd_data_valid_q") != 1 or integer(high, "b_rd_data_q") != expected_value:
                    order_errors += 1

    stats["pending_transfer_responses"] = len(pending_reads)
    stats["undrained_scoreboard_items"] = len(expected)
    stats["end_to_end_order_errors"] = order_errors
    stats["transfer_latency_min"] = min(transfer_latencies) if transfer_latencies else -1
    stats["transfer_latency_max"] = max(transfer_latencies) if transfer_latencies else -1
    stats["transfer_latency_one_cycles"] = sum(latency == 1 for latency in transfer_latencies)

    last_high = high_by_cycle[max(high_by_cycle)]
    stats["final_a_count"] = integer(last_high, "a_count_q")
    stats["final_b_count"] = integer(last_high, "b_count_q")

    checks = {
        "a_full_and_empty_seen": stats["a_full_cycles"] > 0 and stats["a_empty_cycles"] > 0,
        "b_full_and_empty_seen": stats["b_full_cycles"] > 0 and stats["b_empty_cycles"] > 0,
        "producer_backpressure_seen": stats["producer_blocked_cycles"] > 0,
        "b_full_stops_a": stats["a_stopped_by_b_full_cycles"] > 0,
        "a_pointer_wraps": stats["a_write_pointer_wraps"] > 0 and stats["a_read_pointer_wraps"] > 0,
        "b_pointer_wraps": stats["b_write_pointer_wraps"] > 0 and stats["b_read_pointer_wraps"] > 0,
        "a_simultaneous_rw": stats["a_simultaneous_rw"] > 0,
        "b_simultaneous_rw": stats["b_simultaneous_rw"] > 0,
        "transfer_occurred": stats["transfer_cycles"] > 0,
        "two_reset_cycles": stats["reset_cycles"] >= 2,
        "registered_transfer_latency": bool(transfer_latencies) and all(latency == 1 for latency in transfer_latencies),
        "no_pending_response": stats["pending_transfer_responses"] == 0,
        "end_to_end_order": stats["end_to_end_order_errors"] == 0 and stats["undrained_scoreboard_items"] == 0,
        "fully_drained": stats["final_a_count"] == 0 and stats["final_b_count"] == 0,
    }
    return stats, checks


def main():
    if len(sys.argv) != 6:
        print("usage: compare.py <rtl_trace.csv> <cpp1.csv> <cpp2.csv> <cpp3.csv> <cpp4.csv>", file=sys.stderr)
        return 2

    rtl_rows = read_trace(sys.argv[1])
    cpp_traces = [(Path(path).name, read_trace(path)) for path in sys.argv[2:]]
    reference_name, reference_rows = cpp_traces[0]

    schedule_mismatches = []
    for name, rows in cpp_traces[1:]:
        if rows != reference_rows:
            limit = min(len(rows), len(reference_rows))
            first = next((index for index in range(limit) if rows[index] != reference_rows[index]), limit)
            schedule_mismatches.append((name, first, len(rows), len(reference_rows)))

    mismatches = []
    for name, rows in cpp_traces:
        mismatches.extend(compare_cpp_rtl(name, rows, rtl_rows))

    audits = semantic_audit(reference_rows)
    stats, checks = coverage_and_scoreboard(reference_rows)
    coverage_ok = all(checks.values())
    audit_failures = sum(audits.values())

    print(f"rtl_samples={len(rtl_rows)}")
    for name, rows in cpp_traces:
        print(f"cpp_samples.{name}={len(rows)}")
    print(f"schedule_trace_mismatches={len(schedule_mismatches)}")
    print(f"cpp_rtl_mismatches={len(mismatches)}")
    print(f"semantic_audit_failures={audit_failures}")

    if schedule_mismatches:
        print(f"first_schedule_mismatch={schedule_mismatches[0]}")
    else:
        print("first_schedule_mismatch=none")
    if mismatches:
        print(f"first_cpp_rtl_mismatch={mismatches[0]}")
    else:
        print("first_cpp_rtl_mismatch=none")
    if audits:
        for key, count in sorted(audits.items()):
            print(f"semantic_audit.{key}={count}")
    else:
        print("semantic_audit=PASS")

    print("coverage_summary=" + ";".join(f"{key}={val}" for key, val in stats.items()))
    for key, passed in checks.items():
        print(f"coverage_check.{key}={'PASS' if passed else 'FAIL'}")
    print(f"coverage_result={'PASS' if coverage_ok else 'FAIL'}")

    passed = (not schedule_mismatches and not mismatches and audit_failures == 0 and coverage_ok and
              len(rtl_rows) == 192 and all(len(rows) == 192 for _, rows in cpp_traces))
    print(f"RESULT={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
