#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


BASE_FIELDS = [
    "half_clk", "cycle", "phase", "clk_level", "state_valid",
    "rst", "wr_en", "wr_data", "rd_en", "wr_fire", "rd_fire",
]
FLAGS = ["full", "empty", "wr_ready", "rd_available"]
CURRENT_STATE = [
    "wr_ptr_q", "rd_ptr_q", "count_q", "rd_data_q", "rd_data_valid_q",
]
NEXT_STATE = [
    "wr_ptr_d", "rd_ptr_d", "count_d", "rd_data_d", "rd_data_valid_d",
]


def read_trace(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalize(row, field):
    return row.get(field, "").strip().lower()


def add_mismatch(mismatches, index, field, cpp_value, rtl_value):
    mismatches.append((index, field, cpp_value, rtl_value))


def compare_field(mismatches, index, field, cpp, rtl):
    cpp_value = normalize(cpp, field)
    rtl_value = normalize(rtl, field)
    if cpp_value != rtl_value:
        add_mismatch(mismatches, index, field, cpp_value, rtl_value)


def compare_memory(mismatches, index, cpp, rtl):
    for slot in range(4):
        valid_field = f"mem{slot}_valid"
        data_field = f"mem{slot}"
        compare_field(mismatches, index, valid_field, cpp, rtl)
        if normalize(cpp, valid_field) == "1" and normalize(rtl, valid_field) == "1":
            compare_field(mismatches, index, data_field, cpp, rtl)


def integer(row, field):
    return int(normalize(row, field))


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


def coverage(rows):
    low = [row for row in rows if normalize(row, "phase") == "low"]
    high_by_cycle = {
        normalize(row, "cycle"): row
        for row in rows
        if normalize(row, "phase") == "high"
    }
    valid_low = [row for row in low if normalize(row, "state_valid") == "1"]

    result = {
        "full_cycles": sum(integer(row, "full") for row in valid_low),
        "empty_cycles": sum(integer(row, "empty") for row in valid_low),
        "accepted_writes": sum(integer(row, "wr_fire") for row in low),
        "rejected_writes": sum(
            integer(row, "rst") == 0 and integer(row, "wr_en") == 1 and
            integer(row, "wr_fire") == 0 for row in low
        ),
        "accepted_reads": sum(integer(row, "rd_fire") for row in low),
        "rejected_reads": sum(
            integer(row, "rst") == 0 and integer(row, "rd_en") == 1 and
            integer(row, "rd_fire") == 0 for row in low
        ),
        "simultaneous_accepted_rw": sum(
            integer(row, "wr_fire") == 1 and integer(row, "rd_fire") == 1
            for row in low
        ),
        "full_simultaneous_requests": sum(
            integer(row, "full") == 1 and integer(row, "wr_en") == 1 and
            integer(row, "rd_en") == 1 and integer(row, "rst") == 0
            for row in valid_low
        ),
        "empty_simultaneous_requests": sum(
            integer(row, "empty") == 1 and integer(row, "wr_en") == 1 and
            integer(row, "rd_en") == 1 and integer(row, "rst") == 0
            for row in valid_low
        ),
        "write_pointer_wraps": sum(
            integer(row, "wr_fire") == 1 and integer(row, "wr_ptr_q") == 3
            for row in valid_low
        ),
        "read_pointer_wraps": sum(
            integer(row, "rd_fire") == 1 and integer(row, "rd_ptr_q") == 3
            for row in valid_low
        ),
        "reset_cycles": sum(integer(row, "rst") for row in low),
        "reset_during_traffic": sum(
            integer(row, "rst") == 1 and
            (integer(row, "wr_en") == 1 or integer(row, "rd_en") == 1)
            for row in low
        ),
        "rd_data_valid_cycles": sum(
            integer(row, "rd_data_valid_q") for row in rows
            if normalize(row, "phase") == "high" and normalize(row, "state_valid") == "1"
        ),
    }

    result["max_consecutive_accepted_writes"] = longest_run(
        low, lambda row: integer(row, "wr_fire") == 1
    )
    result["max_consecutive_accepted_reads"] = longest_run(
        low, lambda row: integer(row, "rd_fire") == 1
    )
    result["max_consecutive_accepted_rw"] = longest_run(
        low, lambda row: integer(row, "wr_fire") == 1 and integer(row, "rd_fire") == 1
    )
    result["max_consecutive_rejected_writes"] = longest_run(
        low, lambda row: integer(row, "rst") == 0 and
        integer(row, "wr_en") == 1 and integer(row, "wr_fire") == 0
    )

    transitions = {
        "empty_to_nonempty": 0,
        "became_full": 0,
        "full_to_nonfull": 0,
        "became_empty": 0,
    }
    for low_row in valid_low:
        high_row = high_by_cycle[normalize(low_row, "cycle")]
        before = integer(low_row, "count_q")
        after = integer(high_row, "count_q")
        transitions["empty_to_nonempty"] += before == 0 and after > 0
        transitions["became_full"] += before < 4 and after == 4
        transitions["full_to_nonfull"] += before == 4 and after < 4
        transitions["became_empty"] += before > 0 and after == 0
    result.update(transitions)
    return result


def coverage_passed(stats):
    checks = {
        "full_seen": stats["full_cycles"] > 0,
        "empty_seen": stats["empty_cycles"] > 0,
        "write_accept_and_reject": stats["accepted_writes"] > 0 and stats["rejected_writes"] > 0,
        "read_accept_and_reject": stats["accepted_reads"] > 0 and stats["rejected_reads"] > 0,
        "simultaneous_accept": stats["simultaneous_accepted_rw"] > 0,
        "full_simultaneous": stats["full_simultaneous_requests"] > 0,
        "empty_simultaneous": stats["empty_simultaneous_requests"] > 0,
        "pointer_wraps": stats["write_pointer_wraps"] > 0 and stats["read_pointer_wraps"] > 0,
        "fill_and_drain": stats["became_full"] > 0 and stats["became_empty"] > 0,
        "boundary_transitions": stats["empty_to_nonempty"] > 0 and stats["full_to_nonfull"] > 0,
        "initial_and_mid_reset": stats["reset_cycles"] >= 3 and stats["reset_during_traffic"] > 0,
        "sustained_writes": stats["max_consecutive_accepted_writes"] >= 4,
        "sustained_reads": stats["max_consecutive_accepted_reads"] >= 3,
        "sustained_rw": stats["max_consecutive_accepted_rw"] >= 3,
        "backpressure": stats["max_consecutive_rejected_writes"] >= 2,
        "registered_read_response": stats["rd_data_valid_cycles"] > 0,
    }
    return checks, all(checks.values())


def main():
    if len(sys.argv) != 3:
        print("usage: compare.py <cpp_trace.csv> <rtl_trace.csv>", file=sys.stderr)
        return 2

    cpp_rows = read_trace(Path(sys.argv[1]))
    rtl_rows = read_trace(Path(sys.argv[2]))
    mismatches = []

    if len(cpp_rows) != len(rtl_rows):
        add_mismatch(mismatches, min(len(cpp_rows), len(rtl_rows)),
                     "row_count", str(len(cpp_rows)), str(len(rtl_rows)))

    for index, (cpp, rtl) in enumerate(zip(cpp_rows, rtl_rows)):
        for field in BASE_FIELDS:
            compare_field(mismatches, index, field, cpp, rtl)

        both_valid = normalize(cpp, "state_valid") == "1" and normalize(rtl, "state_valid") == "1"
        phase = normalize(cpp, "phase")
        if phase == "low":
            for field in NEXT_STATE:
                compare_field(mismatches, index, field, cpp, rtl)
            if both_valid:
                for field in FLAGS + CURRENT_STATE:
                    compare_field(mismatches, index, field, cpp, rtl)
                compare_memory(mismatches, index, cpp, rtl)
        elif phase == "high" and both_valid:
            for field in FLAGS + CURRENT_STATE:
                compare_field(mismatches, index, field, cpp, rtl)
            compare_memory(mismatches, index, cpp, rtl)

    stats = coverage(cpp_rows)
    checks, all_coverage = coverage_passed(stats)
    low_samples = sum(normalize(row, "phase") == "low" for row in cpp_rows)
    high_samples = sum(normalize(row, "phase") == "high" for row in cpp_rows)

    print(f"cpp_samples={len(cpp_rows)}")
    print(f"rtl_samples={len(rtl_rows)}")
    print(f"low_samples={low_samples}")
    print(f"high_samples={high_samples}")
    print(f"mismatches={len(mismatches)}")

    if mismatches:
        index, field, cpp_value, rtl_value = mismatches[0]
        print(f"first_mismatch=row:{index},field:{field},cpp:{cpp_value},rtl:{rtl_value}")
        if index < len(cpp_rows):
            row = cpp_rows[index]
            print(f"first_mismatch_half_clk={normalize(row, 'half_clk')}")
            print(f"first_mismatch_cycle={normalize(row, 'cycle')}")
            print(f"first_mismatch_phase={normalize(row, 'phase')}")
        print("context:")
        for context_index in range(max(0, index - 2), min(len(cpp_rows), len(rtl_rows), index + 3)):
            print(f"  CPP[{context_index}]={cpp_rows[context_index]}")
            print(f"  RTL[{context_index}]={rtl_rows[context_index]}")
    else:
        print("first_mismatch=none")

    summary = ";".join(f"{key}={value}" for key, value in stats.items())
    print(f"coverage_summary={summary}")
    for key, value in checks.items():
        print(f"coverage_check.{key}={'PASS' if value else 'FAIL'}")
    print(f"coverage_result={'PASS' if all_coverage else 'FAIL'}")

    passed = not mismatches and len(cpp_rows) == len(rtl_rows) and all_coverage
    print(f"RESULT={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
