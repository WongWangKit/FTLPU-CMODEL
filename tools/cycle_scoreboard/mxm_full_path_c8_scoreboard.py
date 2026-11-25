#!/usr/bin/env python3
"""Normalize and compare the integrated MXM instruction and datapath trace."""

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


COLUMNS = [
    "cycle", "phase", "compute_fire", "load_fire", "param_fire",
    "dequant_fire", "backend_fire", "compute_queue_level", "load_queue_level",
    "param_queue_level", "dequant_queue_level", "backend_queue_level",
    "compute_row_valid", "load_row_valid",
    "param_row_valid", "dequant_row_valid", "cell_weight_valid",
    "cell_activation_valid", "partial_valid", "partial_data", "partial_tag",
    "returned_valid", "returned_data", "returned_tag", "backend_issue_valid",
    "backend_issue_operation", "backend_issue_address", "backend_slot_valid",
    "alu_valid", "bank_request_valid", "bank_write", "result_valid",
    "result_bf16", "result_data", "result_address", "result_tag",
    "frontend_fault", "backend_fault", "fault",
]


def _hex(value: str, width: int, source: str) -> str:
    text = value.strip()
    if any(bit in text.lower() for bit in ("x", "z")):
        raise TraceValidationError(f"{source}: X/Z is not allowed in C8 canonical trace")
    try:
        integer = int(text, 16)
    except ValueError as error:
        raise TraceValidationError(f"{source}: invalid hex {value!r}") from error
    if integer < 0 or integer >= (1 << width):
        raise TraceValidationError(f"{source}: value exceeds {width} bits")
    return format(integer, f"0{width}b")


def _extract(value: str, offset: int, width: int, source: str) -> str:
    text = value.strip()
    if any(bit in text.lower() for bit in ("x", "z")):
        raise TraceValidationError(f"{source}: X/Z is not allowed in C8 canonical trace")
    try:
        integer = int(text, 16)
    except ValueError as error:
        raise TraceValidationError(f"{source}: invalid hex {value!r}") from error
    return format((integer >> offset) & ((1 << width) - 1), f"0{width}b")


def normalize(source_path: pathlib.Path, source: str,
              destination: pathlib.Path) -> dict[str, Any]:
    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise TraceValidationError(
                f"{source_path}: columns differ; expected={COLUMNS} "
                f"actual={reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 91:
        raise TraceValidationError(f"{source_path}: expected 91 rows, got {len(rows)}")

    records: list[dict[str, Any]] = []

    def append(cycle: int, phase: str, module: str, instance: str,
               signal: str, width: int, value: str, status: str,
               line: int) -> None:
        records.append({
            "schema": "ftlpu.mxm.full_path.c8.normalized.v1",
            "run_id": "mxm_full_path_c8",
            "case_id": "param_weight_compute_direct_out",
            "source": source,
            "global_cycle": cycle,
            "logical_cycle": cycle,
            "transport_tick": cycle,
            "phase": phase,
            "sample_point": "PRE_POSEDGE" if phase == "LOW" else "POST_POSEDGE",
            "event_sequence": len(records),
            "module": module,
            "instance": instance,
            "signal": signal,
            "width": width,
            "value": value,
            "event_type": "SIGNAL_SAMPLE",
            "mapping_status": status,
            "source_ref": f"{source_path}:{line}",
        })

    for expected_cycle, row in enumerate(rows):
        line = expected_cycle + 2
        try:
            cycle = int(row["cycle"], 10)
        except ValueError as error:
            raise TraceValidationError(f"{source_path}:{line}: invalid cycle") from error
        if cycle != expected_cycle or row["phase"] != "BOTH":
            raise TraceValidationError(
                f"{source_path}:{line}: expected cycle {expected_cycle} BOTH")

        for field, signal in (
            ("compute_fire", "compute_launch_fire"),
            ("load_fire", "load_launch_fire"),
            ("param_fire", "param_load_launch_fire"),
            ("dequant_fire", "dequant_launch_fire"),
            ("backend_fire", "backend_launch_fire"),
        ):
            append(cycle, "LOW", "lpu_mxm_command_interface", "mxm0",
                signal, 1, _hex(row[field], 1, f"{source_path}:{line}:{field}"),
                "DERIVED", line)

        for field, instance in (
            ("compute_queue_level", "compute"),
            ("load_queue_level", "load"),
            ("param_queue_level", "param_load"),
            ("dequant_queue_level", "dequant"),
            ("backend_queue_level", "backend"),
        ):
            append(cycle, "HIGH", "lpu_mxm_command_queue", instance,
                "queue_level", 32,
                _hex(row[field], 32, f"{source_path}:{line}:{field}"),
                "DIRECT", line)

        for field, signal, width in (
            ("compute_row_valid", "compute_row_valid_o", 4),
            ("load_row_valid", "load_row_valid_o", 4),
            ("param_row_valid", "param_load_row_valid_o", 4),
            ("dequant_row_valid", "dequant_row_valid_o", 4),
            ("cell_weight_valid", "cell_weight_valid_o", 16),
            ("cell_activation_valid", "cell_activation_valid_o", 16),
            ("backend_slot_valid", "backend_slot_valid_o", 4),
            ("alu_valid", "alu_valid_o", 4),
            ("bank_request_valid", "bank_request_valid_o", 8),
            ("bank_write", "bank_write_o", 8),
            ("frontend_fault", "frontend_fault_o", 1),
            ("backend_fault", "backend_fault_o", 1),
            ("fault", "fault_o", 1),
        ):
            append(cycle, "HIGH", "lpu_mxm_core", "mxm0", signal, width,
                _hex(row[field], width, f"{source_path}:{line}:{field}"),
                "DIRECT", line)

        for column in range(4):
            instance = f"partial{column}"
            append(cycle, "HIGH", "lpu_mxm_frontend", instance,
                "partial_valid_o", 1,
                _extract(row["partial_valid"], column, 1, f"{source_path}:{line}:partial_valid"),
                "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_frontend", instance,
                "partial_o", 256,
                _extract(row["partial_data"], column * 256, 256,
                    f"{source_path}:{line}:partial_data"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_frontend", instance,
                "partial_tag_o", 16,
                _extract(row["partial_tag"], column * 16, 16,
                    f"{source_path}:{line}:partial_tag"), "DIRECT", line)

        for row_index in range(4):
            instance = f"return{row_index}"
            append(cycle, "HIGH", "lpu_mxm_result_return", instance,
                "returned_valid_o", 1,
                _extract(row["returned_valid"], row_index, 1,
                    f"{source_path}:{line}:returned_valid"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_result_return", instance,
                "returned_o", 256,
                _extract(row["returned_data"], row_index * 256, 256,
                    f"{source_path}:{line}:returned_data"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_result_return", instance,
                "returned_tag_o", 16,
                _extract(row["returned_tag"], row_index * 16, 16,
                    f"{source_path}:{line}:returned_tag"), "DIRECT", line)

        for column in range(4):
            instance = f"issue{column}"
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "backend_issue_valid_o", 1,
                _extract(row["backend_issue_valid"], column, 1,
                    f"{source_path}:{line}:backend_issue_valid"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "backend_issue_operation_o", 2,
                _extract(row["backend_issue_operation"], column * 2, 2,
                    f"{source_path}:{line}:backend_issue_operation"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "backend_issue_address_o", 16,
                _extract(row["backend_issue_address"], column * 16, 16,
                    f"{source_path}:{line}:backend_issue_address"), "DIRECT", line)

        for column in range(4):
            instance = f"result{column}"
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "result_valid_o", 1,
                _extract(row["result_valid"], column, 1,
                    f"{source_path}:{line}:result_valid"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "result_bf16_o", 1,
                _extract(row["result_bf16"], column, 1,
                    f"{source_path}:{line}:result_bf16"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "result_o", 256,
                _extract(row["result_data"], column * 256, 256,
                    f"{source_path}:{line}:result_data"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "result_address_o", 16,
                _extract(row["result_address"], column * 16, 16,
                    f"{source_path}:{line}:result_address"), "DIRECT", line)
            append(cycle, "HIGH", "lpu_mxm_backend", instance,
                "result_tag_o", 16,
                _extract(row["result_tag"], column * 16, 16,
                    f"{source_path}:{line}:result_tag"), "DIRECT", line)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"rows": len(rows), "normalized_samples": len(records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel", type=pathlib.Path, required=True)
    parser.add_argument("--rtl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rules", type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("mxm_full_path_c8_rules.json"))
    args = parser.parse_args(argv)
    try:
        normalized = args.output_dir / "normalized"
        summaries = {
            "CMODEL": normalize(args.cmodel, "CMODEL", normalized / "cmodel.jsonl"),
            "RTL": normalize(args.rtl, "RTL", normalized / "rtl.jsonl"),
        }
        result = run_comparison(normalized / "cmodel.jsonl",
            normalized / "rtl.jsonl", args.rules, None, 1_000_000)
        result["joint_verdict"] = (
            "MXM_FULL_PATH_CYCLE_PASS" if result["status"] == "PASS"
            else "MXM_FULL_PATH_CYCLE_FAIL")
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
        print(f"MXM_FULL_PATH_CYCLE_FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
