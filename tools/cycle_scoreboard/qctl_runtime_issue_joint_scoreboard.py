#!/usr/bin/env python3
"""Compare QCTL Runtime control and real MEM/MXM issue traces."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys
from typing import Any

from raw_macro_level1_scoreboard import (
    TraceValidationError,
    run_comparison,
    write_reports,
)


COLUMNS = [
    "global_cycle", "phase", "queue", "fu", "logical_cycle",
    "head_valid", "head_cycle", "opcode", "operand", "scheduler_ce",
    "sync_arrive", "sync_release", "notify_valid", "notify_ready",
    "event_valid", "event_wait", "event_consume", "mem_packet_valid",
    "mem_packet_ordinal", "mem_packet_retired", "issue_valid",
    "issue_ready", "issue_fire", "issue_instruction", "issue_cycle",
    "event_in_valid", "event_in_tag", "fault",
]

WIDTHS = {
    "logical_cycle": 32, "head_valid": 1, "head_cycle": 32,
    "opcode": 8, "operand": 32, "scheduler_ce": 1,
    "sync_arrive": 1, "sync_release": 1, "notify_valid": 1,
    "notify_ready": 1, "event_valid": 1, "event_wait": 1,
    "event_consume": 1, "mem_packet_valid": 1,
    "mem_packet_ordinal": 32, "mem_packet_retired": 1,
    "issue_valid": 1, "issue_ready": 1, "issue_fire": 1,
    "issue_instruction": 64, "issue_cycle": 32,
    "event_in_valid": 1, "event_in_tag": 32, "fault": 1,
}

EXPECTED_FUS = {
    0: "mem", 1: "mem_notifier", 208: "mxm_load",
    212: "mxm_dequant", 216: "mxm_compute",
    220: "mxm_param_load", 224: "mxm_backend",
}

PAYLOAD_GATES = {
    "head_cycle": "head_valid", "opcode": "head_valid",
    "operand": "head_valid",
    "mem_packet_ordinal": "mem_packet_valid",
    "issue_ready": "issue_valid",
    "issue_instruction": "issue_valid", "issue_cycle": "issue_valid",
    "event_in_tag": "event_in_valid",
}


def _read(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise TraceValidationError(
                f"{path}: columns differ; expected={COLUMNS} actual={reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 203:
        raise TraceValidationError(f"{path}: expected 203 rows, got {len(rows)}")
    seen: set[tuple[int, int]] = set()
    for line, row in enumerate(rows, 2):
        cycle = int(row["global_cycle"], 10)
        queue = int(row["queue"], 10)
        fu = row["fu"].strip()
        if row["phase"] != "POST_POSEDGE" or not 0 <= cycle <= 28:
            raise TraceValidationError(f"{path}:{line}: invalid phase/cycle")
        if EXPECTED_FUS.get(queue) != fu:
            raise TraceValidationError(f"{path}:{line}: invalid queue/FU {queue}/{fu}")
        key = (cycle, queue)
        if key in seen:
            raise TraceValidationError(f"{path}:{line}: duplicate sample {key}")
        seen.add(key)
    expected = {(cycle, queue) for cycle in range(29) for queue in EXPECTED_FUS}
    if seen != expected:
        raise TraceValidationError(f"{path}: incomplete cycle/queue matrix")
    return rows


def _bits(text: str, width: int, base: int, source: str) -> str:
    value_text = text.strip()
    if any(value in value_text.lower() for value in ("x", "z")):
        raise TraceValidationError(f"{source}: X/Z is not canonical")
    try:
        value = int(value_text, base)
    except ValueError as error:
        raise TraceValidationError(f"{source}: invalid scalar {text!r}") from error
    if value < 0 or value >= (1 << width):
        raise TraceValidationError(f"{source}: value exceeds {width} bits")
    return format(value, f"0{width}b")


def normalize(source_path: pathlib.Path, source: str,
              destination: pathlib.Path) -> dict[str, Any]:
    rows = _read(source_path)
    records: list[dict[str, Any]] = []
    excluded: collections.Counter[str] = collections.Counter()
    for line, row in enumerate(rows, 2):
        cycle = int(row["global_cycle"], 10)
        queue = int(row["queue"], 10)
        instance = f"queue{queue}_{row['fu'].strip()}"
        for signal, width in WIDTHS.items():
            gate = PAYLOAD_GATES.get(signal)
            status = "DIRECT"
            if gate is not None and int(row[gate].strip(), 10) == 0:
                status = "NA"
                excluded[signal] += 1
            base = 16 if signal == "issue_instruction" else 10
            records.append({
                "schema": "ftlpu.qctl.runtime_issue_joint.normalized.v1",
                "run_id": "qctl_runtime_issue_joint_20261123",
                "case_id": "qctl_runtime_mem_mxm_issue",
                "source": source,
                "global_cycle": cycle,
                "logical_cycle": cycle,
                "transport_tick": cycle,
                "phase": "HIGH",
                "sample_point": "POST_POSEDGE",
                "event_sequence": len(records),
                "module": "qctl_runtime_issue",
                "instance": instance,
                "signal": signal,
                "width": width,
                "value": _bits(row[signal], width, base,
                               f"{source_path}:{line}:{signal}"),
                "event_type": "SIGNAL_SAMPLE",
                "mapping_status": status,
                "source_ref": f"{source_path}:{line}",
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"flat_rows": len(rows), "normalized_rows": len(records),
            "excluded_invalid_payloads": dict(sorted(excluded.items()))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel", type=pathlib.Path, required=True)
    parser.add_argument("--rtl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rules", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name(
                            "qctl_runtime_issue_joint_rules.json"))
    args = parser.parse_args(argv)
    try:
        normalized = args.output_dir / "normalized"
        cmodel_norm = normalized / "cmodel.jsonl"
        rtl_norm = normalized / "rtl.jsonl"
        csummary = normalize(args.cmodel, "CMODEL", cmodel_norm)
        rsummary = normalize(args.rtl, "RTL", rtl_norm)
        result = run_comparison(
            cmodel_norm, rtl_norm, args.rules, None, 1_000_000)
        result["joint_verdict"] = (
            "QCTL_RUNTIME_ISSUE_CYCLE_PASS" if result["status"] == "PASS"
            else "QCTL_RUNTIME_ISSUE_CYCLE_FAIL")
        result["flat_trace_summary"] = {"CMODEL": csummary, "RTL": rsummary}
        result["joint_scope"] = {
            "queues": sorted(EXPECTED_FUS), "cycles": 29,
            "actual_issue_events": 9,
            "control_events": ["SYNC", "NOTIFY", "WAIT_EVENT",
                               "EVENT_CONSUME", "MEM_PACKET_RETIRE",
                               "SCHEDULER_PAUSE_RESUME"],
            "excluded_scope": ["VXM", "SXM"],
        }
        write_reports(result, args.output_dir)
        summary = {
            key: result[key] for key in (
                "joint_verdict", "status", "compared_pairs", "matched_pairs",
                "difference_count", "first_difference", "flat_trace_summary",
                "joint_scope")
        }
        (args.output_dir / "qctl_runtime_issue_joint_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        (args.output_dir / "qctl_runtime_issue_joint_summary.txt").write_text(
            "\n".join([
                "QCTL Runtime + MEM/MXM Issue Joint Scoreboard",
                f"Verdict: {result['joint_verdict']}",
                f"Compared: {result['compared_pairs']}",
                f"Matched: {result['matched_pairs']}",
                f"Differences: {result['difference_count']}",
                "Rows: CModel=203 RTL=203",
                "Actual issue events: 9",
                "Invalid payloads are validity-gated, not value-masked.",
            ]) + "\n", encoding="utf-8")
    except (OSError, ValueError, TraceValidationError) as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"joint_verdict": "QCTL_RUNTIME_ISSUE_CYCLE_FAIL",
                   "status": "FAIL", "validation_error": str(error)}
        (args.output_dir / "qctl_runtime_issue_joint_summary.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
