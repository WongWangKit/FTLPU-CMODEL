"""Normalize flat QCTL Phase 2-B traces and run the shared cycle scoreboard.

The adapter is deliberately schema-strict.  It derives a case-local cycle from
each source's first physical sample; it never searches for a timing offset,
truncates a trace, or compares only the intersection of the two traces.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any

from raw_macro_level1_scoreboard import (
    TraceValidationError,
    run_comparison,
    write_reports,
)


GLOBAL_SCHEMAS = {
    "global_control": {"rst_n", "clear", "run", "load_mode",
                       "sync_participant_mask", "mem_packet_retired_mask"},
    "partition_config": {"valid", "ready", "fire", "queue", "base",
                         "count", "generation"},
    "record_load": {"valid", "ready", "fire", "address", "data"},
    "event_ingress": {"valid", "ready", "fire", "tag", "target_mask",
                      "backpressure"},
    "global_status": {"queue_active", "all_primed", "all_done", "fault",
                      "fault_code"},
    "pool_fabric": {"request_valid_mask", "request_ready_mask",
                    "request_index_vector", "request_generation_vector",
                    "response_valid", "response_queue", "response_data",
                    "response_error"},
    "event_buffer_fabric": {"wait_mask", "consume_mask", "event_valid_mask",
                            "event_tag"},
}

ENDPOINT_FIELDS = {
    "queue", "instance", "configure_valid", "configure_count",
    "configure_generation", "pool_request_valid", "pool_request_ready",
    "pool_request_fire", "pool_request_index", "pool_request_generation",
    "pool_response_valid", "pool_response_data", "pool_response_error",
    "scheduler_ce", "local_cycle", "event_valid", "event_tag", "event_wait",
    "event_consume", "sync_arrive", "sync_release", "notify_valid",
    "notify_ready", "notify_fire", "mem_packet_valid", "mem_packet_ordinal",
    "mem_packet_retired", "mem_packet_fire", "configured", "primed", "done",
    "fault", "fault_code", "program_record", "opcode", "issue_valid",
    "issue_ready", "issue_fire",
}

WIDTHS = {
    "global_control.rst_n": 1, "global_control.clear": 1,
    "global_control.run": 1, "global_control.load_mode": 1,
    "global_control.sync_participant_mask": 3,
    "global_control.mem_packet_retired_mask": 3,
    "partition_config.valid": 1, "partition_config.ready": 1,
    "partition_config.fire": 1, "partition_config.queue": 2,
    "partition_config.base": 4, "partition_config.count": 3,
    "partition_config.generation": 8,
    "record_load.valid": 1, "record_load.ready": 1, "record_load.fire": 1,
    "record_load.address": 4, "record_load.data": 96,
    "event_ingress.valid": 1, "event_ingress.ready": 1,
    "event_ingress.fire": 1, "event_ingress.tag": 32,
    "event_ingress.target_mask": 3, "event_ingress.backpressure": 1,
    "global_status.queue_active": 3, "global_status.all_primed": 1,
    "global_status.all_done": 1, "global_status.fault": 1,
    "global_status.fault_code": 8,
    "pool_fabric.request_valid_mask": 3, "pool_fabric.request_ready_mask": 3,
    "pool_fabric.request_index_vector": 6,
    "pool_fabric.request_generation_vector": 24,
    "pool_fabric.response_valid": 1, "pool_fabric.response_queue": 2,
    "pool_fabric.response_data": 96, "pool_fabric.response_error": 1,
    "event_buffer_fabric.wait_mask": 3,
    "event_buffer_fabric.consume_mask": 3,
    "event_buffer_fabric.event_valid_mask": 3,
    "event_buffer_fabric.event_tag": 32,
    "endpoint.configure_valid": 1, "endpoint.configure_count": 3,
    "endpoint.configure_generation": 8, "endpoint.pool_request_valid": 1,
    "endpoint.pool_request_ready": 1, "endpoint.pool_request_fire": 1,
    "endpoint.pool_request_index": 2, "endpoint.pool_request_generation": 8,
    "endpoint.pool_response_valid": 1, "endpoint.pool_response_data": 96,
    "endpoint.pool_response_error": 1, "endpoint.scheduler_ce": 1,
    "endpoint.local_cycle": 32, "endpoint.event_valid": 1,
    "endpoint.event_tag": 32, "endpoint.event_wait": 1,
    "endpoint.event_consume": 1, "endpoint.sync_arrive": 1,
    "endpoint.sync_release": 1, "endpoint.notify_valid": 1,
    "endpoint.notify_ready": 1, "endpoint.notify_fire": 1,
    "endpoint.mem_packet_valid": 1, "endpoint.mem_packet_ordinal": 32,
    "endpoint.mem_packet_retired": 1, "endpoint.mem_packet_fire": 1,
    "endpoint.configured": 1, "endpoint.primed": 1, "endpoint.done": 1,
    "endpoint.fault": 1, "endpoint.fault_code": 4,
    "endpoint.program_record": 96, "endpoint.opcode": 8,
}

PAYLOAD_GATES = {
    "partition_config.queue": ("valid",),
    "partition_config.base": ("valid",),
    "partition_config.count": ("valid",),
    "partition_config.generation": ("valid",),
    "record_load.address": ("valid",),
    "record_load.data": ("valid",),
    "event_ingress.tag": ("valid",),
    "event_ingress.target_mask": ("valid",),
    "pool_fabric.request_index_vector": ("request_valid_mask",),
    "pool_fabric.request_generation_vector": ("request_valid_mask",),
    "pool_fabric.response_queue": ("response_valid",),
    "pool_fabric.response_data": ("response_valid",),
    "pool_fabric.response_error": ("response_valid",),
    "event_buffer_fabric.event_tag": ("event_valid_mask",),
    "endpoint.configure_count": ("configure_valid",),
    "endpoint.configure_generation": ("configure_valid",),
    "endpoint.pool_request_index": ("pool_request_valid",),
    "endpoint.pool_request_generation": ("pool_request_valid",),
    "endpoint.pool_response_data": ("pool_response_valid",),
    "endpoint.pool_response_error": ("pool_response_valid",),
    "endpoint.event_tag": ("event_valid",),
    "endpoint.mem_packet_ordinal": ("mem_packet_valid",),
}

OMITTED_NULL_FIELDS = {"issue_valid", "issue_ready", "issue_fire"}


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise TraceValidationError(f"{path}:{line_number}: blank JSONL row")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TraceValidationError(f"{path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise TraceValidationError(f"{path}:{line_number}: row is not an object")
        record["_line"] = line_number
        records.append(record)
    if not records:
        raise TraceValidationError(f"{path}: empty trace")
    return records


def _validate_flat(records: list[dict[str, Any]], path: pathlib.Path) -> None:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    last_cycle: dict[str, int] = {}
    for row in records:
        common = {"case_id", "cycle", "phase", "boundary", "_line"}
        if not common <= set(row):
            raise TraceValidationError(f"{path}:{row['_line']}: missing common fields")
        if row["phase"] != "POST_COMB" or not isinstance(row["cycle"], int):
            raise TraceValidationError(f"{path}:{row['_line']}: invalid phase/cycle")
        boundary = row["boundary"]
        if boundary == "endpoint":
            expected = ENDPOINT_FIELDS
        elif boundary in GLOBAL_SCHEMAS:
            expected = GLOBAL_SCHEMAS[boundary]
        else:
            raise TraceValidationError(f"{path}:{row['_line']}: unknown boundary {boundary}")
        actual = set(row) - common
        if actual != expected:
            raise TraceValidationError(
                f"{path}:{row['_line']}: {boundary} fields differ; "
                f"missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
        case_id = row["case_id"]
        if case_id in last_cycle and row["cycle"] < last_cycle[case_id]:
            raise TraceValidationError(f"{path}:{row['_line']}: cycle moved backwards")
        last_cycle[case_id] = row["cycle"]
        buckets[(case_id, row["cycle"])].append(row)
    for bucket, rows in buckets.items():
        counts = collections.Counter(row["boundary"] for row in rows)
        endpoint_queues = sorted(row["queue"] for row in rows
                                 if row["boundary"] == "endpoint")
        if len(rows) != 10 or endpoint_queues != [0, 1, 2] or any(
                counts[name] != 1 for name in GLOBAL_SCHEMAS) or counts["endpoint"] != 3:
            raise TraceValidationError(
                f"{path}: sample bucket {bucket} is not 7 global + 3 endpoint rows")


def _as_int(value: Any, source: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lower().startswith("0x"):
        return int(value, 16)
    raise TraceValidationError(f"{source}: unsupported scalar {value!r}")


def _as_bits(value: Any, width: int, source: str) -> str:
    if isinstance(value, (bool, int)):
        integer = int(value)
        if integer < 0 or integer >= (1 << width):
            raise TraceValidationError(
                f"{source}: value {integer} exceeds {width} bits")
        return format(integer, f"0{width}b")
    if isinstance(value, str) and value.lower().startswith("0x"):
        digits = value[2:]
        translated = []
        for digit in digits:
            if digit in "xX":
                translated.append("xxxx")
            elif digit in "zZ":
                translated.append("zzzz")
            elif digit.lower() in "0123456789abcdef":
                translated.append(format(int(digit, 16), "04b"))
            else:
                raise TraceValidationError(f"{source}: invalid hexadecimal {value!r}")
        bits = "".join(translated)
        if len(bits) > width:
            excess = bits[:-width]
            if any(bit not in "0" for bit in excess):
                raise TraceValidationError(
                    f"{source}: hexadecimal value {value!r} exceeds {width} bits")
            bits = bits[-width:]
        return bits.rjust(width, "0")
    raise TraceValidationError(f"{source}: unsupported scalar {value!r}")


def _payload_valid(row: dict[str, Any], signal: str) -> bool:
    gate = PAYLOAD_GATES.get(signal)
    if gate is None:
        return True
    return all(_as_int(row[field], signal) != 0 for field in gate)


def normalize(source_path: pathlib.Path, source: str,
              destination: pathlib.Path) -> dict[str, Any]:
    rows = _read_jsonl(source_path)
    _validate_flat(rows, source_path)
    first_cycle: dict[str, int] = {}
    for row in rows:
        first_cycle.setdefault(row["case_id"], row["cycle"])
    output: list[dict[str, Any]] = []
    excluded = collections.Counter()
    for row in rows:
        boundary = row["boundary"]
        instance = row.get("instance", boundary)
        local_cycle = row["cycle"] - first_cycle[row["case_id"]]
        fields = ENDPOINT_FIELDS if boundary == "endpoint" else GLOBAL_SCHEMAS[boundary]
        for field in sorted(fields):
            if field in {"queue", "instance"} or field in OMITTED_NULL_FIELDS:
                continue
            signal = f"{boundary}.{field}"
            if signal not in WIDTHS:
                raise TraceValidationError(f"no width declared for {signal}")
            width = WIDTHS[signal]
            value = _as_bits(row[field], width,
                             f"{source_path}:{row['_line']}:{signal}")
            mapping = "DIRECT" if _payload_valid(row, signal) else "NA"
            if mapping == "NA":
                excluded[signal] += 1
            output.append({
                "schema": "ftlpu.qctl.phase2b.normalized.v1",
                "run_id": "qctl_phase2b_20261120",
                "case_id": row["case_id"],
                "source": source,
                "global_cycle": local_cycle,
                "logical_cycle": local_cycle,
                "transport_tick": local_cycle,
                "phase": "POST_COMB",
                "sample_point": "POST_COMB",
                "event_sequence": len(output),
                "module": boundary,
                "instance": instance,
                "signal": signal,
                "width": width,
                "value": value,
                "event_type": "SIGNAL_SAMPLE",
                "mapping_status": mapping,
                "source_ref": f"{source_path}:{row['_line']}:physical_cycle={row['cycle']}",
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in output:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    cases = collections.Counter(row["case_id"] for row in rows)
    return {
        "flat_rows": len(rows), "normalized_rows": len(output),
        "sampled_cycles": len(rows) // 10,
        "case_sampled_cycles": {case: count // 10 for case, count in sorted(cases.items())},
        "first_physical_cycle": dict(sorted(first_cycle.items())),
        "protocol_na_rows": sum(excluded.values()),
        "protocol_na_by_signal": dict(sorted(excluded.items())),
        "omitted_null_fields": sorted(OMITTED_NULL_FIELDS),
    }


def _stimulus_audit(cmodel_path: pathlib.Path, rtl_path: pathlib.Path) -> dict[str, Any]:
    left, right = _read_jsonl(cmodel_path), _read_jsonl(rtl_path)
    def semantic(row: dict[str, Any]) -> tuple[Any, ...]:
        return row["case_id"], row["action"], row["queue"], row["record"], row["opcode"], row["operand"]
    semantic_equal = [semantic(row) for row in left] == [semantic(row) for row in right]
    def localize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        first: dict[str, int] = {}
        result = []
        for row in rows:
            first.setdefault(row["case_id"], row["cycle"])
            item = dict(row)
            item["case_local_cycle"] = row["cycle"] - first[row["case_id"]]
            result.append(item)
        return result
    llocal, rlocal = localize(left), localize(right)
    cycle_differences = []
    for index, (lrow, rrow) in enumerate(zip(llocal, rlocal)):
        if lrow["case_local_cycle"] != rrow["case_local_cycle"]:
            cycle_differences.append({
                "index": index, "case_id": lrow["case_id"], "action": lrow["action"],
                "cmodel_case_local_cycle": lrow["case_local_cycle"],
                "rtl_case_local_cycle": rrow["case_local_cycle"],
            })
    return {"cmodel_rows": len(left), "rtl_rows": len(right),
            "semantic_sequence_equal": semantic_equal,
            "case_local_cycle_difference_count": len(cycle_differences),
            "case_local_cycle_differences": cycle_differences}


def _sampling_races(trace_path: pathlib.Path, stimulus_path: pathlib.Path) -> list[dict[str, Any]]:
    trace = _read_jsonl(trace_path)
    stimulus = _read_jsonl(stimulus_path)
    lookup = {(row["case_id"], row["cycle"], row["boundary"], row.get("queue")): row
              for row in trace}
    races = []
    checks = {
        "partition": ("partition_config", None, "valid", 1),
        "load_complete": ("global_control", None, "load_mode", 0),
        "run_assert": ("global_control", None, "run", 1),
        "event_in": ("event_ingress", None, "valid", 1),
        "mem_packet_retired": ("global_control", None,
                               "mem_packet_retired_mask", "0x1"),
    }
    for action in stimulus:
        if action["action"] not in checks:
            continue
        boundary, queue, field, expected = checks[action["action"]]
        row = lookup.get((action["case_id"], action["cycle"], boundary, queue))
        if row is not None and row[field] != expected:
            races.append({"case_id": action["case_id"], "cycle": action["cycle"],
                          "action": action["action"], "boundary": boundary,
                          "field": field, "expected_after_drive": expected,
                          "sampled": row[field], "trace_line": row["_line"]})
    for row in trace:
        if row["boundary"] != "partition_config" or not row["valid"] or not row["ready"]:
            continue
        endpoint = lookup.get((row["case_id"], row["cycle"], "endpoint", row["queue"]))
        if endpoint is not None and not endpoint["configure_valid"]:
            races.append({"case_id": row["case_id"], "cycle": row["cycle"],
                          "action": "partition_cross_boundary",
                          "boundary": "partition_config->endpoint",
                          "field": "configure_valid", "expected_after_drive": 1,
                          "sampled": endpoint["configure_valid"],
                          "trace_line": endpoint["_line"]})
    return races


def _group_result(result: dict[str, Any], cmodel_norm: pathlib.Path,
                  rtl_norm: pathlib.Path) -> list[dict[str, Any]]:
    traces = {"CMODEL": _read_jsonl(cmodel_norm), "RTL": _read_jsonl(rtl_norm)}
    direct_keys: dict[str, collections.Counter[tuple[str, str, str]]] = {}
    for source, rows in traces.items():
        direct_keys[source] = collections.Counter(
            (row["case_id"], row["module"], row["signal"])
            for row in rows if row["mapping_status"] == "DIRECT")
    diff_counts = collections.Counter(
        (item["case_id"], item["module"], item["signal"])
        for item in result["all_differences"])
    groups = []
    for key in sorted(set(direct_keys["CMODEL"]) | set(direct_keys["RTL"])):
        ccount, rcount = direct_keys["CMODEL"][key], direct_keys["RTL"][key]
        compared = min(ccount, rcount)
        differences = diff_counts[key]
        missing = abs(ccount - rcount)
        value_differences = max(0, differences - missing)
        groups.append({"case_id": key[0], "boundary": key[1], "signal": key[2],
                       "cmodel_eligible": ccount, "rtl_eligible": rcount,
                       "compared": compared, "matched": max(0, compared-value_differences),
                       "differences": differences})
    return groups


def _aggregate_groups(groups: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    totals: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for group in groups:
        key = str(group[field])
        for metric in ("cmodel_eligible", "rtl_eligible", "compared", "matched",
                       "differences"):
            totals[key][metric] += int(group[metric])
    return [{field: key, **dict(values)} for key, values in sorted(totals.items())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel", type=pathlib.Path, required=True)
    parser.add_argument("--rtl", type=pathlib.Path, required=True)
    parser.add_argument("--cmodel-stimulus", type=pathlib.Path, required=True)
    parser.add_argument("--rtl-stimulus", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rules", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name("qctl_phase2b_rules.json"))
    parser.add_argument("--diagnostic-rules", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name(
                            "qctl_phase2b_diagnostic_rules.json"))
    args = parser.parse_args(argv)
    try:
        normalized = args.output_dir / "normalized"
        cmodel_norm = normalized / "cmodel_qctl_normalized.jsonl"
        rtl_norm = normalized / "rtl_qctl_normalized.jsonl"
        csummary = normalize(args.cmodel, "CMODEL", cmodel_norm)
        rsummary = normalize(args.rtl, "RTL", rtl_norm)
        result = run_comparison(cmodel_norm, rtl_norm, args.rules, None, 1_000_000)
        diagnostic_result = run_comparison(
            cmodel_norm, rtl_norm, args.diagnostic_rules,
            args.output_dir / "diagnostic", 1_000_000)
        stimulus_cases = []
        for row in _read_jsonl(args.cmodel_stimulus):
            if row["case_id"] not in stimulus_cases:
                stimulus_cases.append(row["case_id"])
        case_rank = {case: index for index, case in enumerate(stimulus_cases)}
        def signal_rank(item: dict[str, Any]) -> int:
            leaf = item["signal"].rsplit(".", 1)[-1]
            if leaf.endswith("valid") or leaf in {"ready", "fire"}:
                return 0
            if leaf.endswith("ready") or leaf.endswith("fire"):
                return 0
            return 1
        result["all_differences"].sort(key=lambda item: (
            case_rank.get(item["case_id"], len(case_rank)),
            item["global_cycle"], signal_rank(item), item["module"], item["instance"],
            item["signal"], item["category"]))
        result["differences"] = list(result["all_differences"])
        result["first_difference"] = (
            result["all_differences"][0] if result["all_differences"] else None)
        result["joint_verdict"] = (
            "QCTL_PHASE2B_CYCLE_PASS" if result["status"] == "PASS"
            else "QCTL_PHASE2B_CYCLE_FAIL")
        result["diagnostic_comparison"] = {
            "status": diagnostic_result["status"],
            "compared_pairs": diagnostic_result["compared_pairs"],
            "matched_pairs": diagnostic_result["matched_pairs"],
            "difference_count": diagnostic_result["difference_count"],
            "first_difference": diagnostic_result["first_difference"],
            "verdict_effect": "NON_BLOCKING_TB_DIAGNOSTIC"
        }
        result["flat_trace_summary"] = {"CMODEL": csummary, "RTL": rsummary}
        result["stimulus_audit"] = _stimulus_audit(
            args.cmodel_stimulus, args.rtl_stimulus)
        result["rtl_same_delta_sampling_findings"] = _sampling_races(
            args.rtl, args.rtl_stimulus)
        result["blockers"] = ([{
            "classification": "RTL_TRACE_NOT_STABLE_POST_COMB",
            "reason": "The RTL monitor and stimulus tasks both execute on negedge without a settled sampling region; the saved rows contain driven inputs and stale combinational outputs.",
            "source": "D:/LPU/FTLPU-VMODEL-main/sim/tb/lpu_qctl_arch_trace_tb.sv:177",
            "minimum_resolution": "Regenerate the RTL trace from a clocking block or after an explicit post-drive settle point; no VCS rerun was authorized in this task."
        }] if result["rtl_same_delta_sampling_findings"] else [])
        result["grouped_results"] = _group_result(result, cmodel_norm, rtl_norm)
        result["grouped_case_summary"] = _aggregate_groups(
            result["grouped_results"], "case_id")
        result["grouped_boundary_summary"] = _aggregate_groups(
            result["grouped_results"], "boundary")
        result["uncovered_fields"] = [
            "endpoint.issue_valid (both flat traces are null)",
            "endpoint.issue_ready (both flat traces are null)",
            "endpoint.issue_fire (both flat traces are null)",
        ]
        write_reports(result, args.output_dir)
        summary_path = args.output_dir / "qctl_joint_summary.json"
        summary_path.write_text(json.dumps({
            key: result[key] for key in (
                "joint_verdict", "status", "compared_pairs", "matched_pairs",
                "difference_count", "first_difference", "flat_trace_summary",
                "diagnostic_comparison", "stimulus_audit",
                "rtl_same_delta_sampling_findings", "blockers",
                "uncovered_fields", "grouped_case_summary",
                "grouped_boundary_summary", "grouped_results")},
            indent=2), encoding="utf-8")
        text_lines = [
            "QCTL Phase 2-B Joint Scoreboard",
            f"Verdict: {result['joint_verdict']}",
            f"Compared: {result['compared_pairs']}",
            f"Matched: {result['matched_pairs']}",
            f"Differences: {result['difference_count']}",
            f"Non-blocking diagnostic compared/matched/differences: "
            f"{result['diagnostic_comparison']['compared_pairs']}/"
            f"{result['diagnostic_comparison']['matched_pairs']}/"
            f"{result['diagnostic_comparison']['difference_count']}",
            f"CModel flat rows/cycles: {csummary['flat_rows']}/{csummary['sampled_cycles']}",
            f"RTL flat rows/cycles: {rsummary['flat_rows']}/{rsummary['sampled_cycles']}",
            f"Shared stimulus semantic sequence equal: {result['stimulus_audit']['semantic_sequence_equal']}",
            f"RTL same-delta sampling findings: {len(result['rtl_same_delta_sampling_findings'])}",
            f"Uncovered: {', '.join(result['uncovered_fields'])}",
        ]
        if result["first_difference"]:
            first = result["first_difference"]
            text_lines.append(
                f"First difference: {first['case_id']} local_cycle={first['global_cycle']} "
                f"{first['signal']} {first['category']}")
        text_lines.append("Per-case compared/matched/differences:")
        for item in result["grouped_case_summary"]:
            text_lines.append(
                f"  {item['case_id']}: {item['compared']}/{item['matched']}/"
                f"{item['differences']}")
        if result["blockers"]:
            text_lines.append("Blocker: RTL_TRACE_NOT_STABLE_POST_COMB")
        (args.output_dir / "qctl_joint_summary.txt").write_text(
            "\n".join(text_lines) + "\n", encoding="utf-8")
    except (OSError, ValueError, TraceValidationError) as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"joint_verdict": "QCTL_PHASE2B_CYCLE_FAIL",
                   "status": "FAIL", "validation_error": str(error)}
        (args.output_dir / "qctl_joint_summary.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 2
    print(json.dumps({
        "joint_verdict": result["joint_verdict"],
        "compared_pairs": result["compared_pairs"],
        "matched_pairs": result["matched_pairs"],
        "difference_count": result["difference_count"],
        "first_difference": result["first_difference"],
        "rtl_same_delta_sampling_findings":
            len(result["rtl_same_delta_sampling_findings"]),
    }, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
