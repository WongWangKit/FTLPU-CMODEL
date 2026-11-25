#!/usr/bin/env python3
"""Normalize and compare the five MXM Scheduler-to-Core command streams."""

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


STREAMS = ("load", "compute", "dequant", "param_load", "backend")
WIDTHS = {"load": 2, "compute": 7, "dequant": 1,
          "param_load": 1, "backend": 13}
FIELDS = ("issue_valid", "issue_ready", "issue_fire",
          "issue_instruction", "issue_cycle")
EXPECTED_COLUMNS = [
    "cycle", "phase",
    *(f"{stream}_{field}" for stream in STREAMS for field in FIELDS),
    "conflict", "fault",
]


def _bits(value: str, width: int, source: str) -> str:
    try:
        integer = int(value.strip(), 10)
    except ValueError as error:
        raise TraceValidationError(
            f"{source}: invalid integer {value!r}") from error
    if integer < 0 or integer >= (1 << width):
        raise TraceValidationError(
            f"{source}: value {integer} exceeds {width} bits")
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
    if len(rows) != 8:
        raise TraceValidationError(
            f"{source_path}: expected 8 cycles, got {len(rows)}")

    normalized: list[dict[str, Any]] = []

    def append(cycle: int, instance: str, signal: str, width: int,
               value: str, status: str, line: int) -> None:
        normalized.append({
            "schema": "ftlpu.mxm.full_path.c7.normalized.v1",
            "run_id": "mxm_full_path_c7",
            "case_id": "five_stream_single_context",
            "source": source,
            "global_cycle": cycle,
            "logical_cycle": cycle,
            "transport_tick": cycle,
            "phase": "LOW",
            "sample_point": "PRE_POSEDGE",
            "event_sequence": len(normalized),
            "module": "lpu_mxm_macro_frontend",
            "instance": instance,
            "signal": signal,
            "width": width,
            "value": _bits(value, width,
                f"{source_path}:{line}:{instance}:{signal}"),
            "event_type": "SIGNAL_SAMPLE",
            "mapping_status": status,
            "source_ref": f"{source_path}:{line}",
        })

    for expected_cycle, row in enumerate(rows):
        line = expected_cycle + 2
        try:
            cycle = int(row["cycle"], 10)
        except ValueError as error:
            raise TraceValidationError(
                f"{source_path}:{line}: invalid cycle") from error
        if cycle != expected_cycle or row["phase"] != "PRE_POSEDGE":
            raise TraceValidationError(
                f"{source_path}:{line}: expected cycle {expected_cycle} PRE_POSEDGE")
        for stream in STREAMS:
            append(cycle, stream, "issue_valid_o", 1,
                row[f"{stream}_issue_valid"], "DIRECT", line)
            append(cycle, stream, "issue_ready_i", 1,
                row[f"{stream}_issue_ready"], "DIRECT", line)
            append(cycle, stream, "issue_fire", 1,
                row[f"{stream}_issue_fire"], "DERIVED", line)
            append(cycle, stream, "issue_instruction_o", WIDTHS[stream],
                row[f"{stream}_issue_instruction"], "DIRECT", line)
            append(cycle, stream, "issue_cycle_o", 32,
                row[f"{stream}_issue_cycle"], "DIRECT", line)
        append(cycle, "mxm0", "conflict_o", 1, row["conflict"],
            "DIRECT", line)
        append(cycle, "mxm0", "fault_o", 1, row["fault"],
            "DIRECT", line)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in normalized:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"rows": len(rows), "signals_per_cycle": 27,
            "normalized_samples": len(normalized)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel", type=pathlib.Path, required=True)
    parser.add_argument("--rtl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rules", type=pathlib.Path,
        default=pathlib.Path(__file__).with_name(
            "mxm_full_path_c7_rules.json"))
    args = parser.parse_args(argv)
    try:
        normalized = args.output_dir / "normalized"
        summaries = {
            "CMODEL": normalize(args.cmodel, "CMODEL",
                normalized / "cmodel.jsonl"),
            "RTL": normalize(args.rtl, "RTL", normalized / "rtl.jsonl"),
        }
        result = run_comparison(normalized / "cmodel.jsonl",
            normalized / "rtl.jsonl", args.rules, None, 1_000_000)
        result["joint_verdict"] = (
            "MXM_C7_CYCLE_PASS" if result["status"] == "PASS"
            else "MXM_C7_CYCLE_FAIL")
        result["flat_trace_summary"] = summaries
        write_reports(result, args.output_dir)
        print(result["joint_verdict"])
        print(f"Compared: {result['compared_pairs']}")
        print(f"Matched: {result['matched_pairs']}")
        print(f"Differences: {result['difference_count']}")
        if result["first_difference"]:
            print(f"First difference: {result['first_difference']}")
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, TraceValidationError) as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"MXM_C7_CYCLE_FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
