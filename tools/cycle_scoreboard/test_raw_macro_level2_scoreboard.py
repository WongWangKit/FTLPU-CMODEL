#!/usr/bin/env python3
"""Self-tests for Level2 schema, scope filtering, and coverage audit."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from raw_macro_level1_scoreboard import run_comparison


ROOT = pathlib.Path(__file__).resolve().parent
RULES = ROOT / "raw_macro_level2_rules.json"
EVENTS = [
    (0, "LOW", "PRE_POSEDGE", "MEM_ISSUE_ACCEPT", "ICU", 64),
    (0, "LOW", "PRE_POSEDGE", "MEM_TILE_COMMAND", "MEM", 64),
    (0, "LOW", "PRE_POSEDGE", "MEM_SRAM_READ_SAMPLE", "SRAM", 64),
    (0, "POSEDGE", "POST_POSEDGE", "MEM_READ_RESPONSE", "SRAM", 64),
    (1, "POSEDGE", "POST_POSEDGE", "MEM_SRF_INJECT_COMMIT", "STREAM", 64),
    (1, "POSEDGE", "POST_COMB", "MEM_DRAINED", "MEM", 1),
]


def row(source: str, sequence: int, cycle: int, phase: str,
        sample: str, signal: str, scope: str, width: int,
        event_type: str = "ARCH_EVENT", category: str = "EVENT",
        mapping: str = "SEMANTIC") -> dict[str, object]:
    return {
        "schema": "ftlpu.raw_macro.level2.trace.v1",
        "run_id": f"selftest-{source.lower()}",
        "case_id": "mem_read_single",
        "source": source,
        "global_cycle": cycle,
        "logical_cycle": cycle,
        "transport_tick": cycle,
        "phase": phase,
        "sample_point": sample,
        "event_sequence": sequence,
        "module": "architecture" if event_type == "ARCH_EVENT"
                  else "lpu_mem_execution_frontend",
        "instance": "architecture" if event_type == "ARCH_EVENT"
                    else "mem_e_s0_b0",
        "signal": signal,
        "width": width,
        "value": "0" * (width - 1) + "1",
        "event_type": event_type,
        "mapping_status": mapping,
        "source_ref": "Level2 scoreboard self-test",
        "hemisphere": "EAST",
        "mem_slice": 0,
        "bank": 0,
        "tile": 0,
        "opcode": "READ",
        "address": 32,
        "stream": 1,
        "direction": "E",
        "preserve": 0,
        "data": "0x01",
        "category": category,
        "scope": scope,
    }


def trace(source: str) -> list[dict[str, object]]:
    result = [
        row(source, index, cycle, phase, sample, signal, scope, width)
        for index, (cycle, phase, sample, signal, scope, width)
        in enumerate(EVENTS)
    ]
    result.append(row(source, len(result), 0, "LOW", "PRE_POSEDGE",
        "execution_issue_ready", "ICU", 1, "SIGNAL_SAMPLE", "CANONICAL", "DIRECT"))
    diagnostic = row(source, len(result), 0, "POSEDGE", "POST_COMB",
        "mem_busy_derived", "MEM", 1, "STATE_SAMPLE", "DIAGNOSTIC", "DERIVED")
    diagnostic["module"] = "cmodel_mem_diagnostic"
    result.append(diagnostic)
    return result


def write_jsonl(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


class Level2ScoreboardSelfTest(unittest.TestCase):
    def compare(self, cmodel: list[dict[str, object]],
                rtl: list[dict[str, object]], scopes=None):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            cmodel_path = root / "cmodel.jsonl"
            rtl_path = root / "rtl.jsonl"
            write_jsonl(cmodel_path, cmodel)
            write_jsonl(rtl_path, rtl)
            return run_comparison(cmodel_path, rtl_path, RULES, scopes=scopes)

    def test_identical_all_scopes_pass_and_audit(self) -> None:
        result = self.compare(trace("CMODEL"), trace("RTL"))
        self.assertEqual("PASS", result["status"])
        self.assertEqual(["ICU", "MEM", "SRAM", "STREAM"],
                         result["coverage"]["active_scopes"])
        self.assertIn("EVENT", result["coverage"]["per_category"])
        self.assertIn("CANONICAL", result["coverage"]["per_category"])
        self.assertIn("DIAGNOSTIC", result["coverage"]["per_category"])
        self.assertIn("MEM_READ_RESPONSE",
                      result["coverage"]["audit"]["CMODEL"]["captured_events"])

    def test_value_mismatch_reports_all_differences(self) -> None:
        rtl = trace("RTL")
        rtl[2]["value"] = "1" + "0" * 63
        result = self.compare(trace("CMODEL"), rtl)
        self.assertEqual("FAIL", result["status"])
        self.assertEqual("value_mismatch", result["first_difference"]["category"])
        self.assertEqual(result["difference_count"], len(result["all_differences"]))

    def test_scope_filter_masks_out_of_scope_mismatch(self) -> None:
        rtl = trace("RTL")
        rtl[4]["value"] = "1" + "0" * 63
        result = self.compare(trace("CMODEL"), rtl, scopes=["ICU", "MEM", "SRAM"])
        self.assertEqual("PASS", result["status"])
        self.assertGreater(result["coverage"]["audit"]["RTL"]["masked_by_scope_rows"], 0)

    def test_same_cycle_parallel_event_order_is_not_architectural(self) -> None:
        cmodel = trace("CMODEL")
        rtl = trace("RTL")
        rtl[1], rtl[2] = rtl[2], rtl[1]
        for sequence, record in enumerate(rtl):
            record["event_sequence"] = sequence
        result = self.compare(cmodel, rtl)
        self.assertEqual("PASS", result["status"])
        self.assertEqual("KEYED_IDENTITY",
                         result["coverage"]["architecture"]["comparison_mode"])

    def test_keyed_identity_does_not_hide_cycle_or_tile_change(self) -> None:
        rtl = trace("RTL")
        rtl[2]["global_cycle"] = 1
        rtl[2]["logical_cycle"] = 1
        rtl[2]["transport_tick"] = 1
        rtl[2]["tile"] = 1
        result = self.compare(trace("CMODEL"), rtl)
        self.assertEqual("FAIL", result["status"])
        self.assertIn(result["first_difference"]["category"],
                      {"missing_cmodel", "missing_rtl"})

    def test_unmapped_is_audited_not_compared(self) -> None:
        left = trace("CMODEL")
        right = trace("RTL")
        extra_left = copy.deepcopy(left[-1])
        extra_right = copy.deepcopy(right[-1])
        for extra, source in ((extra_left, "CMODEL"), (extra_right, "RTL")):
            extra["source"] = source
            extra["event_sequence"] = len(left)
            extra["signal"] = "private_state"
            extra["mapping_status"] = "UNMAPPED"
        left.append(extra_left)
        right.append(extra_right)
        result = self.compare(left, right)
        self.assertEqual("PASS", result["status"])
        self.assertEqual(1, result["coverage"]["audit"]["CMODEL"]["missing_mapping_rows"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
