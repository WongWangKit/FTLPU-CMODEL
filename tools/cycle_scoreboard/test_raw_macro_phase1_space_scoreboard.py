#!/usr/bin/env python3
"""Strict identity self-tests for Phase1 MEM spatial coverage."""

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
RULES = ROOT / "raw_macro_phase1_space_rules.json"


def row(source: str, sequence: int, *, hemisphere: str = "EAST",
        bank: int = 0, tile: int = 0, cycle: int = 0,
        signal: str = "MEM_TILE_COMMAND", value: int = 1,
        event_type: str = "ARCH_EVENT", instance: str = "architecture",
        scope: str = "MEM") -> dict[str, object]:
    return {
        "schema": "ftlpu.raw_macro.level2.trace.v1",
        "run_id": f"phase1-selftest-{source.lower()}",
        "case_id": "parallel_space",
        "source": source,
        "global_cycle": cycle,
        "logical_cycle": cycle,
        "transport_tick": cycle,
        "phase": "LOW",
        "sample_point": "PRE_POSEDGE",
        "event_sequence": sequence,
        "module": "architecture" if event_type == "ARCH_EVENT"
                  else "lpu_mem_execution_frontend",
        "instance": instance,
        "signal": signal,
        "width": 64 if event_type == "ARCH_EVENT" else 1,
        "value": format(value, "064b" if event_type == "ARCH_EVENT" else "01b"),
        "event_type": event_type,
        "mapping_status": "SEMANTIC" if event_type == "ARCH_EVENT" else "DIRECT",
        "source_ref": "Phase1 scoreboard self-test",
        "hemisphere": hemisphere,
        "mem_slice": 0,
        "bank": bank,
        "tile": tile,
        "opcode": "READ",
        "address": 32,
        "stream": bank + 1,
        "direction": "E",
        "preserve": 0,
        "data": "0x01",
        "category": "EVENT" if event_type == "ARCH_EVENT" else "CANONICAL",
        "scope": scope,
    }


def write_jsonl(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


class Phase1SpaceScoreboardSelfTest(unittest.TestCase):
    def compare(self, cmodel: list[dict[str, object]],
                rtl: list[dict[str, object]]):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            left = root / "cmodel.jsonl"
            right = root / "rtl.jsonl"
            write_jsonl(left, cmodel)
            write_jsonl(right, rtl)
            return run_comparison(left, right, RULES)

    def base(self, source: str) -> list[dict[str, object]]:
        return [
            row(source, 0, hemisphere="EAST", bank=0),
            row(source, 1, hemisphere="WEST", bank=0),
            row(source, 2, hemisphere="EAST", bank=1),
        ]

    def test_identical_parallel_space_passes(self) -> None:
        self.assertEqual("PASS", self.compare(
            self.base("CMODEL"), self.base("RTL"))["status"])

    def test_hemisphere_cannot_match_wrong_side(self) -> None:
        rtl = self.base("RTL")
        rtl[1]["hemisphere"] = "EAST"
        self.assertEqual("FAIL", self.compare(self.base("CMODEL"), rtl)["status"])

    def test_bank_cannot_be_folded(self) -> None:
        rtl = self.base("RTL")
        rtl[2]["bank"] = 0
        self.assertEqual("FAIL", self.compare(self.base("CMODEL"), rtl)["status"])

    def test_occurrence_count_mismatch_fails(self) -> None:
        left = self.base("CMODEL")
        duplicate = copy.deepcopy(left[0])
        duplicate["event_sequence"] = len(left)
        left.append(duplicate)
        self.assertEqual("FAIL", self.compare(left, self.base("RTL"))["status"])

    def test_cycle_or_tile_change_fails(self) -> None:
        rtl = self.base("RTL")
        rtl[0]["global_cycle"] = 1
        rtl[0]["logical_cycle"] = 1
        rtl[0]["transport_tick"] = 1
        rtl[0]["tile"] = 1
        self.assertEqual("FAIL", self.compare(self.base("CMODEL"), rtl)["status"])

    def test_signal_instance_is_not_canonicalized_to_bank0(self) -> None:
        left = [
            row("CMODEL", 0, bank=0, signal="execution_issue_ready",
                value=0, event_type="SIGNAL_SAMPLE", instance="mem_e_s0_b0",
                scope="ICU"),
            row("CMODEL", 1, bank=1, signal="execution_issue_ready",
                value=1, event_type="SIGNAL_SAMPLE", instance="mem_e_s0_b1",
                scope="ICU"),
        ]
        right = [
            row("RTL", 0, bank=1, signal="execution_issue_ready",
                value=1, event_type="SIGNAL_SAMPLE", instance="mem_e_s0_b1",
                scope="ICU"),
            row("RTL", 1, bank=0, signal="execution_issue_ready",
                value=0, event_type="SIGNAL_SAMPLE", instance="mem_e_s0_b0",
                scope="ICU"),
        ]
        self.assertEqual("PASS", self.compare(left, right)["status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
