"""Compare QCTL Phase 3 distributed-load traces with the shared scoreboard.

The adapter is schema-strict and compares the same physical cycle and
PRE_POSEDGE sample.  It never searches for an offset or truncates either trace.
"""

from __future__ import annotations

import argparse
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


WIDTHS = {
    "load_mode": 1,
    "partition_valid": 1,
    "partition_ready": 1,
    "partition_fire": 1,
    "partition_queue": 8,
    "partition_base": 9,
    "partition_count": 5,
    "partition_generation": 64,
    "load_valid": 1,
    "load_ready": 1,
    "load_fire": 1,
    "load_address": 9,
    "load_data": 96,
    "relay_packet_valid": 1,
    "relay_packet_ready": 1,
    "relay_packet_kind": 1,
    "relay_packet_queue": 8,
    "relay_packet_offset": 4,
    "east_r0_input_fire": 1,
    "east_r0_local_fire": 1,
    "east_r0_output_valid": 1,
    "east_r0_output_ready": 1,
    "east_r0_fifo_level": 2,
    "east_r1_input_fire": 1,
    "east_r1_local_fire": 1,
    "east_r1_fifo_level": 2,
    "endpoint_configure_valid": 1,
    "endpoint_store_loaded": 1,
    "endpoint_head_valid": 1,
    "endpoint_head_error": 1,
    "endpoint_configured": 1,
    "endpoint_primed": 1,
    "queue_active": 1,
    "load_complete": 1,
    "all_primed": 1,
    "fault": 1,
    "fault_code": 8,
}

EXPECTED_COLUMNS = ["cycle", "phase", *WIDTHS]


def _module(field: str) -> str:
    if field.startswith("east_") or field.startswith("relay_packet_"):
        return "qctl_relay_chain"
    if field.startswith("endpoint_") or field == "queue_active":
        return "qctl_endpoint"
    if field in {"load_complete", "all_primed", "fault", "fault_code"}:
        return "qctl_global_subsystem"
    return "qctl_load_manager"


def _bits(value: str, width: int, source: str, hexadecimal: bool) -> str:
    value = value.strip()
    if hexadecimal:
        translated = []
        for digit in value:
            if digit in "xX":
                translated.append("xxxx")
            elif digit in "zZ":
                translated.append("zzzz")
            elif digit.lower() in "0123456789abcdef":
                translated.append(format(int(digit, 16), "04b"))
            else:
                raise TraceValidationError(f"{source}: invalid hexadecimal {value!r}")
        bits = "".join(translated)
        if len(bits) > width and any(bit != "0" for bit in bits[:-width]):
            raise TraceValidationError(f"{source}: value exceeds {width} bits")
        return bits[-width:].rjust(width, "0")
    try:
        integer = int(value, 10)
    except ValueError as error:
        raise TraceValidationError(f"{source}: invalid integer {value!r}") from error
    if integer < 0 or integer >= (1 << width):
        raise TraceValidationError(f"{source}: value {integer} exceeds {width} bits")
    return format(integer, f"0{width}b")


def normalize(source_path: pathlib.Path, source: str,
              destination: pathlib.Path) -> dict[str, Any]:
    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise TraceValidationError(
                f"{source_path}: columns differ; expected={EXPECTED_COLUMNS} "
                f"actual={reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise TraceValidationError(f"{source_path}: empty trace")

    normalized = []
    expected_cycle = 0
    for line, row in enumerate(rows, 2):
        try:
            cycle = int(row["cycle"], 10)
        except ValueError as error:
            raise TraceValidationError(
                f"{source_path}:{line}: invalid cycle {row['cycle']!r}") from error
        if cycle != expected_cycle:
            raise TraceValidationError(
                f"{source_path}:{line}: expected cycle {expected_cycle}, got {cycle}")
        expected_cycle += 1
        if row["phase"] != "PRE_POSEDGE":
            raise TraceValidationError(
                f"{source_path}:{line}: expected PRE_POSEDGE, got {row['phase']!r}")
        for field, width in WIDTHS.items():
            module = _module(field)
            normalized.append({
                "schema": "ftlpu.qctl.phase3.distributed.normalized.v1",
                "run_id": "qctl_phase3_distributed_load",
                "case_id": "queue16_two_record_load",
                "source": source,
                "global_cycle": cycle,
                "logical_cycle": cycle,
                "transport_tick": cycle,
                "phase": "PRE_POSEDGE",
                "sample_point": "PRE_POSEDGE",
                "event_sequence": len(normalized),
                "module": module,
                "instance": module,
                "signal": f"{module}.{field}",
                "width": width,
                "value": _bits(row[field], width,
                    f"{source_path}:{line}:{field}", field == "load_data"),
                "event_type": "SIGNAL_SAMPLE",
                "mapping_status": "DIRECT",
                "source_ref": f"{source_path}:{line}",
            })

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in normalized:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"rows": len(rows), "signals_per_cycle": len(WIDTHS),
            "normalized_samples": len(normalized)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel", type=pathlib.Path, required=True)
    parser.add_argument("--rtl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rules", type=pathlib.Path,
        default=pathlib.Path(__file__).with_name(
            "qctl_phase3_distributed_rules.json"))
    args = parser.parse_args(argv)
    try:
        normalized_dir = args.output_dir / "normalized"
        cmodel_summary = normalize(
            args.cmodel, "CMODEL", normalized_dir / "cmodel.jsonl")
        rtl_summary = normalize(
            args.rtl, "RTL", normalized_dir / "rtl.jsonl")
        result = run_comparison(
            normalized_dir / "cmodel.jsonl",
            normalized_dir / "rtl.jsonl",
            args.rules, None, 1_000_000)
        result["joint_verdict"] = (
            "QCTL_PHASE3_DISTRIBUTED_CYCLE_PASS"
            if result["status"] == "PASS"
            else "QCTL_PHASE3_DISTRIBUTED_CYCLE_FAIL")
        result["flat_trace_summary"] = {
            "CMODEL": cmodel_summary,
            "RTL": rtl_summary,
        }
        write_reports(result, args.output_dir)
        (args.output_dir / "qctl_phase3_summary.json").write_text(
            json.dumps({
                key: result[key] for key in (
                    "joint_verdict", "status", "compared_pairs",
                    "matched_pairs", "difference_count", "first_difference",
                    "flat_trace_summary")
            }, indent=2), encoding="utf-8")
        print(result["joint_verdict"])
        print(f"Compared: {result['compared_pairs']}")
        print(f"Matched: {result['matched_pairs']}")
        print(f"Differences: {result['difference_count']}")
        if result["first_difference"]:
            print(f"First difference: {result['first_difference']}")
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, TraceValidationError) as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"QCTL_PHASE3_DISTRIBUTED_CYCLE_FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
