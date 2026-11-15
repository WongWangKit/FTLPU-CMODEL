#!/usr/bin/env python3
import csv
import sys
from collections import Counter, deque
from pathlib import Path

COMMON = [
    "phase_count", "half_clk", "cycle", "phase", "phase_name", "clk_level", "rst",
    "producer_transaction_index", "producer_valid", "producer_data", "producer_ready",
    "producer_fire", "producer_stall", "producer_pending_count", "producer_accepted_count",
    "consumer_ready", "tc_a_rd_en", "tc_b_wr_en", "tc_b_rd_en",
    "transfer_invariant_ok", "mac_valid", "mac_input", "mac_mul0", "mac_add0", "mac_mul1",
    "mac_add1", "mac_mul2", "mac_result16", "mac_result8", "mac_pre_edge_result",
    "mac_post_edge_result",
]
FLAGS = ["full", "empty", "wr_ready", "rd_available"]
CURRENT = ["wr_ptr_q", "rd_ptr_q", "count_q", "rd_data_q", "rd_data_valid_q"]
NEXT = ["wr_ptr_d", "rd_ptr_d", "count_d", "rd_data_d", "rd_data_valid_d"]
PHASE_NAMES = ["low_evaluate", "posedge_commit", "post_edge_settle", "negedge"]
CLK_LEVELS = [0, 1, 1, 0]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_hex(path, maximum):
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        item = int(token, 16)
        if item > maximum:
            raise ValueError(f"out of range at {path}:{line_number}")
        result.append(item)
    return result


def v(row, field):
    return row.get(field, "").strip().lower()


def n(row, field):
    return int(v(row, field))


def mac_formula(x):
    m0 = x * 3
    a0 = m0 + 17
    m1 = a0 * 5
    a1 = m1 + 29
    m2 = a1 * 7
    r16 = m2 + 43
    return {"input": x, "mul0": m0, "add0": a0, "mul1": m1, "add1": a1,
            "mul2": m2, "result16": r16, "result8": r16 & 0xFF}


def add_mismatch(items, name, index, field, left, right):
    items.append((name, index, field, left, right))


def compare_runtime(name, cpp, rtl):
    mismatches = []
    if len(cpp) != len(rtl):
        add_mismatch(mismatches, name, min(len(cpp), len(rtl)), "row_count", len(cpp), len(rtl))
    for index, (left, right) in enumerate(zip(cpp, rtl)):
        for field in COMMON:
            if v(left, field) != v(right, field):
                add_mismatch(mismatches, name, index, field, v(left, field), v(right, field))
        if v(left, "tc_b_wr_en") == "1" and v(right, "tc_b_wr_en") == "1" and \
                v(left, "tc_b_wr_data") != v(right, "tc_b_wr_data"):
            add_mismatch(mismatches, name, index, "tc_b_wr_data",
                         v(left, "tc_b_wr_data"), v(right, "tc_b_wr_data"))
        for prefix in ("a_", "b_"):
            for field in FLAGS + ["wr_fire", "rd_fire", "state_valid"] + CURRENT + NEXT:
                key = prefix + field
                if v(left, key) != v(right, key):
                    add_mismatch(mismatches, name, index, key, v(left, key), v(right, key))
            for slot in range(4):
                valid_key = f"{prefix}mem{slot}_valid"
                data_key = f"{prefix}mem{slot}"
                if v(left, valid_key) != v(right, valid_key):
                    add_mismatch(mismatches, name, index, valid_key, v(left, valid_key), v(right, valid_key))
                if v(left, valid_key) == "1" and v(right, valid_key) == "1" and v(left, data_key) != v(right, data_key):
                    add_mismatch(mismatches, name, index, data_key, v(left, data_key), v(right, data_key))
    return mismatches


def audit_mac_exhaustive(rows, label):
    failures = Counter()
    if len(rows) != 256:
        failures[label + ".row_count"] += 1
    for index, row in enumerate(rows):
        if index >= 256:
            break
        expected = mac_formula(index)
        for field, expected_value in expected.items():
            if n(row, field) != expected_value:
                failures[label + "." + field] += 1
    return failures


def longest_run(rows, field):
    best = current = 0
    for row in rows:
        if n(row, field):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def memory_data(row, prefix, slot):
    if v(row, f"{prefix}mem{slot}_valid") != "1":
        return None
    return n(row, f"{prefix}mem{slot}")


def audit_runtime(rows, transactions, controls):
    failures = Counter()
    expected_rows = len(controls) * 4
    if len(rows) != expected_rows:
        failures["runtime.row_count"] += 1
    groups = {}
    for index, row in enumerate(rows):
        if n(row, "phase_count") != index or n(row, "half_clk") != index:
            failures["clock.phase_count"] += 1
        phase = index & 3
        cycle = index >> 2
        if n(row, "phase") != phase or n(row, "cycle") != cycle:
            failures["clock.phase_or_cycle"] += 1
        if v(row, "phase_name") != PHASE_NAMES[phase]:
            failures["clock.phase_name"] += 1
        if n(row, "clk_level") != CLK_LEVELS[phase]:
            failures["clock.clk_level"] += 1
        groups.setdefault(cycle, {})[phase] = row

    pending = deque()
    accepted = 0
    consumed = 0
    latencies = []
    for cycle in range(min(len(controls), len(groups))):
        phases = groups[cycle]
        if set(phases) != {0, 1, 2, 3}:
            failures["clock.missing_phase"] += 1
            continue
        p0, p1, p2, p3 = (phases[i] for i in range(4))
        rst = (controls[cycle] >> 1) & 1
        ready_control = controls[cycle] & 1
        for row in (p0, p1, p2, p3):
            if n(row, "rst") != rst or n(row, "consumer_ready") != ready_control:
                failures["control.shared_input"] += 1

        tx_index = n(p0, "producer_transaction_index")
        valid = n(p0, "producer_valid")
        ready = n(p0, "producer_ready")
        fire = n(p0, "producer_fire")
        stall = n(p0, "producer_stall")
        expected_valid = int(not rst and tx_index < len(transactions))
        expected_data = transactions[tx_index] if expected_valid else 0
        if valid != expected_valid or n(p0, "producer_data") != expected_data:
            failures["producer.drive"] += 1
        if fire != int(valid and ready and not rst):
            failures["producer.fire_formula"] += 1
        if stall != int(valid and not ready and not rst):
            failures["producer.stall_formula"] += 1
        if n(p0, "a_wr_fire") != fire or tx_index != accepted:
            failures["producer.accept_pointer"] += 1
        if n(p0, "producer_accepted_count") != accepted or n(p0, "producer_pending_count") != len(transactions) - accepted:
            failures["producer.count_phase0"] += 1
        accepted_after = accepted + fire
        for row in (p1, p2, p3):
            if n(row, "producer_transaction_index") != tx_index or n(row, "producer_fire") != fire:
                failures["producer.pre_edge_event_hold"] += 1
            if n(row, "producer_accepted_count") != accepted_after or n(row, "producer_pending_count") != len(transactions) - accepted_after:
                failures["producer.post_edge_count"] += 1
        if fire:
            if accepted >= len(transactions) or n(p0, "producer_data") != transactions[accepted]:
                failures["producer.accept_order"] += 1
            accepted += 1
        if cycle + 1 in groups:
            next_p0 = groups[cycle + 1][0]
            if n(next_p0, "producer_transaction_index") != tx_index + fire:
                failures["producer.pointer_advance"] += 1
            if stall and not n(next_p0, "rst"):
                if n(next_p0, "producer_valid") != 1 or n(next_p0, "producer_data") != n(p0, "producer_data"):
                    failures["producer.stall_hold"] += 1

        for prefix in ("a_", "b_"):
            for next_field, current_field in zip(NEXT, CURRENT):
                if v(p0, prefix + next_field) != v(p1, prefix + current_field):
                    failures[prefix + "phase0_next_to_phase1_current"] += 1
            for field in NEXT:
                if not (v(p0, prefix + field) == v(p1, prefix + field) ==
                        v(p2, prefix + field) == v(p3, prefix + field)):
                    failures[prefix + "next_mutated_after_phase0"] += 1
            for field in CURRENT + ["state_valid"]:
                if not (v(p1, prefix + field) == v(p2, prefix + field) == v(p3, prefix + field)):
                    failures[prefix + "current_changed_after_commit"] += 1
            for slot in range(4):
                for suffix in ("_valid", ""):
                    field = f"{prefix}mem{slot}{suffix}"
                    if not (v(p1, field) == v(p2, field) == v(p3, field)):
                        failures[prefix + "memory_changed_after_commit"] += 1
            if cycle + 1 in groups:
                next_p0 = groups[cycle + 1][0]
                for field in CURRENT + ["state_valid"]:
                    if v(p3, prefix + field) != v(next_p0, prefix + field):
                        failures[prefix + "negedge_hold"] += 1

        for phase, row in enumerate((p0, p1, p2, p3)):
            if v(row, "mac_input") != "x":
                expected = mac_formula(n(row, "mac_input"))
                for field in ("mul0", "add0", "mul1", "add1", "mul2", "result16", "result8"):
                    if n(row, "mac_" + field) != expected[field]:
                        failures["mac.runtime_" + field] += 1
            if phase in (0, 1) and v(row, "mac_post_edge_result") != v(row, "mac_pre_edge_result"):
                failures["mac.pre_edge_snapshot"] += 1
        for field in ("mac_valid", "mac_input", "mac_mul0", "mac_add0", "mac_mul1", "mac_add1",
                      "mac_mul2", "mac_result16", "mac_result8"):
            if v(p0, field) != v(p1, field):
                failures["mac.phase0_to_phase1_snapshot"] += 1
            if v(p2, field) != v(p3, field):
                failures["mac.phase2_to_phase3_hold"] += 1
        if v(p2, "mac_pre_edge_result") != v(p0, "mac_result8"):
            failures["mac.pre_result_metadata"] += 1
        if v(p2, "mac_post_edge_result") != v(p2, "mac_result8"):
            failures["mac.post_result_metadata"] += 1
        if v(p2, "a_state_valid") == "1" and n(p2, "mac_valid") != n(p2, "a_rd_data_valid_q"):
            failures["mac.valid_mapping"] += 1

        if n(p0, "b_wr_fire"):
            if not pending:
                failures["transfer.response_without_read"] += 1
            else:
                read_cycle, expected = pending.popleft()
                latencies.append(cycle - read_cycle)
                if cycle != read_cycle + 1:
                    failures["transfer.latency"] += 1
                if n(p0, "tc_b_wr_data") != expected:
                    failures["transfer.mac_data"] += 1
        if n(p0, "a_rd_fire"):
            source = memory_data(p0, "a_", n(p0, "a_rd_ptr_q"))
            if source is None:
                failures["transfer.a_invalid_read"] += 1
            else:
                pending.append((cycle, mac_formula(source)["result8"]))
                if n(p2, "mac_valid") != 1 or n(p2, "mac_result8") != mac_formula(source)["result8"]:
                    failures["mac.post_edge_zero_cycle"] += 1
        if n(p0, "b_rd_fire"):
            expected = mac_formula(transactions[consumed])["result8"] if consumed < len(transactions) else None
            if expected is None or n(p1, "b_rd_data_valid_q") != 1 or n(p1, "b_rd_data_q") != expected:
                failures["consumer.order"] += 1
            consumed += 1

    if accepted != len(transactions):
        failures["producer.not_all_accepted"] += 1
    if consumed != len(transactions):
        failures["consumer.not_all_received"] += 1
    if pending:
        failures["transfer.pending_response"] += len(pending)
    last = groups[max(groups)][3]
    if any(n(last, field) != 0 for field in ("producer_pending_count", "a_count_q", "b_count_q",
                                              "a_rd_data_valid_q", "b_rd_data_valid_q")):
        failures["final.not_drained"] += 1
    stats = {
        "cycles": len(controls), "phase_samples": len(rows), "transactions": len(transactions),
        "producer_valid_cycles": sum(n(groups[c][0], "producer_valid") for c in groups),
        "producer_ready_cycles": sum(not n(groups[c][0], "rst") and n(groups[c][0], "producer_ready") for c in groups),
        "producer_fire_cycles": sum(n(groups[c][0], "producer_fire") for c in groups),
        "producer_stall_cycles": sum(n(groups[c][0], "producer_stall") for c in groups),
        "max_consecutive_stall": longest_run([groups[c][0] for c in sorted(groups)], "producer_stall"),
        "accepted": accepted, "consumed": consumed,
        "transfer_latency_min": min(latencies) if latencies else -1,
        "transfer_latency_max": max(latencies) if latencies else -1,
        "final_a_count": n(last, "a_count_q"), "final_b_count": n(last, "b_count_q"),
    }
    return failures, stats


def compare_stage3(stage3_rows, stage4_rows):
    failures = Counter()
    old = [row for row in stage3_rows if v(row, "phase") == "low"]
    new = [row for row in stage4_rows if n(row, "phase") == 0]
    if len(old) != len(new):
        failures["baseline.row_count"] += 1
    fields = [
        "rst", "producer_transaction_index", "producer_valid", "producer_data", "producer_ready",
        "producer_fire", "producer_stall", "producer_pending_count", "producer_accepted_count",
        "consumer_ready", "tc_a_rd_en", "tc_b_wr_en", "tc_b_rd_en", "transfer_invariant_ok",
        "a_full", "a_empty", "a_wr_ready", "a_rd_available", "a_wr_fire", "a_rd_fire",
        "a_wr_ptr_q", "a_rd_ptr_q", "a_count_q", "a_rd_data_q", "a_rd_data_valid_q",
        "b_full", "b_empty", "b_wr_ready", "b_rd_available", "b_wr_fire", "b_rd_fire",
        "b_wr_ptr_q", "b_rd_ptr_q", "b_count_q", "b_rd_data_valid_q",
    ]
    for left, right in zip(old, new):
        for field in fields:
            if v(left, field) != v(right, field):
                failures["baseline." + field] += 1
    return failures


def main():
    if len(sys.argv) != 14:
        print("usage: compare.py tx.hex control.hex stage3.csv rtl.csv rtl_mac.csv cpp1..cpp4 cppmac1..cppmac4", file=sys.stderr)
        return 2
    transactions = read_hex(sys.argv[1], 0xFF)
    controls = read_hex(sys.argv[2], 0x3)
    stage3 = read_csv(sys.argv[3])
    rtl = read_csv(sys.argv[4])
    rtl_mac = read_csv(sys.argv[5])
    cpp = [(Path(path).name, read_csv(path)) for path in sys.argv[6:10]]
    cpp_mac = [(Path(path).name, read_csv(path)) for path in sys.argv[10:14]]
    reference_name, reference = cpp[0]

    schedule_mismatches = [(name, len(rows)) for name, rows in cpp[1:] if rows != reference]
    mac_schedule_mismatches = [(name, len(rows)) for name, rows in cpp_mac[1:] if rows != cpp_mac[0][1]]
    runtime_mismatches = []
    for name, rows in cpp:
        runtime_mismatches.extend(compare_runtime(name, rows, rtl))
    mac_rtl_mismatches = []
    for index, (left, right) in enumerate(zip(cpp_mac[0][1], rtl_mac)):
        for field in ("input", "mul0", "add0", "mul1", "add1", "mul2", "result16", "result8"):
            if v(left, field) != v(right, field):
                mac_rtl_mismatches.append((index, field, v(left, field), v(right, field)))

    audits, stats = audit_runtime(reference, transactions, controls)
    audits.update(audit_mac_exhaustive(rtl_mac, "rtl_mac"))
    for name, rows in cpp_mac:
        audits.update(audit_mac_exhaustive(rows, "cpp_mac." + name))
    baseline = compare_stage3(stage3, reference)
    audit_failures = sum(audits.values())
    baseline_failures = sum(baseline.values())
    expected_samples = len(controls) * 4
    passed = (not schedule_mismatches and not mac_schedule_mismatches and not runtime_mismatches and
              not mac_rtl_mismatches and audit_failures == 0 and baseline_failures == 0 and
              len(rtl) == expected_samples and all(len(rows) == expected_samples for _, rows in cpp))

    print(f"rtl_samples={len(rtl)}")
    for name, rows in cpp:
        print(f"cpp_samples.{name}={len(rows)}")
    print(f"schedule_trace_mismatches={len(schedule_mismatches)}")
    print(f"mac_schedule_mismatches={len(mac_schedule_mismatches)}")
    print(f"cpp_rtl_mismatches={len(runtime_mismatches)}")
    print(f"mac_cpp_rtl_mismatches={len(mac_rtl_mismatches)}")
    print(f"stage3_control_timing_failures={baseline_failures}")
    print(f"semantic_audit_failures={audit_failures}")
    print(f"first_cpp_rtl_mismatch={runtime_mismatches[0] if runtime_mismatches else 'none'}")
    print(f"first_mac_mismatch={mac_rtl_mismatches[0] if mac_rtl_mismatches else 'none'}")
    if baseline:
        for key, count in sorted(baseline.items()):
            print(f"baseline_audit.{key}={count}")
    else:
        print("stage3_control_timing=PASS")
    if audits:
        for key, count in sorted(audits.items()):
            print(f"semantic_audit.{key}={count}")
    else:
        print("semantic_audit=PASS")
    print("coverage_summary=" + ";".join(f"{key}={value}" for key, value in stats.items()))
    print(f"RESULT={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
