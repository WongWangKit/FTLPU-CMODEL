"""Compare the six Phase 3 FU Stream boundaries at the same physical cycle."""

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


BOUNDARIES = (
    "vxm_east_to_mem",
    "mem_east_to_sxm",
    "sxm_east_to_mxm",
    "mxm_west_to_sxm",
    "sxm_west_to_mem",
    "mem_west_to_vxm",
)
WIDTHS = {
    **{field: width for boundary in BOUNDARIES
       for field, width in ((f"{boundary}_valid", 1),
                            (f"{boundary}_data", 8))},
    "stream_conflict": 1,
}
EXPECTED_COLUMNS = ["cycle", "phase", *WIDTHS]


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
    if not rows:
        raise TraceValidationError(f"{source_path}: empty trace")

    normalized = []
    for expected_cycle, row in enumerate(rows):
        line = expected_cycle + 2
        try:
            cycle = int(row["cycle"], 10)
        except ValueError as error:
            raise TraceValidationError(
                f"{source_path}:{line}: invalid cycle {row['cycle']!r}") from error
        if cycle != expected_cycle:
            raise TraceValidationError(
                f"{source_path}:{line}: expected cycle {expected_cycle}, got {cycle}")
        if row["phase"] != "PRE_POSEDGE":
            raise TraceValidationError(
                f"{source_path}:{line}: expected PRE_POSEDGE, "
                f"got {row['phase']!r}")
        for field, width in WIDTHS.items():
            normalized.append({
                "schema": "ftlpu.phase3.stream_interconnect.normalized.v1",
                "run_id": "phase3_r7_stream_interconnect",
                "case_id": "east_west_single_cell",
                "source": source,
                "global_cycle": cycle,
                "logical_cycle": cycle,
                "transport_tick": cycle,
                "phase": "PRE_POSEDGE",
                "sample_point": "PRE_POSEDGE",
                "event_sequence": len(normalized),
                "module": "stream_interconnect",
                "instance": "fu_cluster",
                "signal": f"stream_interconnect.{field}",
                "width": width,
                "value": _bits(
                    row[field], width, f"{source_path}:{line}:{field}"),
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
            "phase3_stream_interconnect_rules.json"))
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
            "PHASE3_R7_STREAM_INTERCONNECT_CYCLE_PASS"
            if result["status"] == "PASS"
            else "PHASE3_R7_STREAM_INTERCONNECT_CYCLE_FAIL")
        result["flat_trace_summary"] = {
            "CMODEL": cmodel_summary,
            "RTL": rtl_summary,
        }
        write_reports(result, args.output_dir)
        (args.output_dir / "phase3_r7_summary.json").write_text(
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
        print(f"PHASE3_R7_STREAM_INTERCONNECT_CYCLE_FAIL: {error}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
