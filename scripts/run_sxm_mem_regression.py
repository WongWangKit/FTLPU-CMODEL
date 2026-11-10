#!/usr/bin/env python3
"""Portable, bounded SXM/MEM regression; outputs stay under ignored build*/.

Uses only the Python standard library, CMake, CTest and a configured C++ compiler.
Does not fetch, commit, push, modify source files, or invoke RTL/EDA tools.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


MATRIX_TESTS = [
    ("sxm_matrix_test", "unit.sxm.matrix"),
    ("sxm_matrix_mem_vxm_test", "subsystem.sxm.matrix_mem_vxm"),
]
LEGACY_TESTS = [
    ("sxm_local_blocks_timing_test", "unit.sxm.local_blocks_timing"),
    ("sxm_mem_transpose_test", "subsystem.sxm.mem_transpose_smoke"),
    ("sxm_test", "unit.sxm.slice"),
    ("sxm_wide_transpose_test", "unit.sxm.wide_transpose"),
    ("sxm_attention_transpose_test", "unit.sxm.attention_transpose"),
    ("icu_sxm_run_2d_instruction_test", "unit.icu.sxm_run_2d"),
    ("icu_sxm_tile_program_test", "unit.icu.sxm_tile_program"),
    ("mem_slice_test", "unit.mem.slice"),
    ("mem_array_separated_test", "unit.mem.array"),
    ("mem_replay_copy_test", "unit.mem.replay_copy"),
    ("mem_sync_read_test", "unit.mem.sync_read"),
    ("mem_array_r1_test", "unit.mem.array_r1"),
    ("mem_read_mode_test", "unit.mem.read_mode"),
    ("vector_roundtrip_test", "unit.mem.vector_roundtrip"),
    ("vxm_fp32_input_test", "vxm_fp32_input"),
    ("vxm_fp16_16alu_test", "unit.vxm.fp16_16alu"),
]


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", nargs="+", type=int, choices=[16, 32, 64], default=[16, 32, 64])
    p.add_argument("--matrix-only", action="store_true", help="omit the 16 existing LPU32 compatibility tests")
    p.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    p.add_argument("--build-root", type=Path, default=repo / "build-sxm-mem-regression")
    p.add_argument("--generator", help="CMake generator; omit to use the environment/platform default")
    p.add_argument("--compiler", help="optional CMAKE_CXX_COMPILER")
    p.add_argument("--make-program", help="optional CMAKE_MAKE_PROGRAM")
    p.add_argument("--cmake", default="cmake")
    p.add_argument("--ctest", default="ctest")
    p.add_argument("--cmake-arg", action="append", default=[], help="extra argument, e.g. --cmake-arg=-DNAME=value")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--timeout", type=int, default=900, help="seconds per configure/build/test step")
    p.add_argument("--dry-run", action="store_true", help="print commands without creating files or running tools")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.run_id) or a.jobs < 1 or a.timeout < 1:
        p.error("invalid run-id, jobs or timeout")
    if len(set(a.profiles)) != len(a.profiles):
        p.error("duplicate profiles")
    output = a.build_root.resolve() / a.run_id
    records = []
    if not a.dry_run:
        output.mkdir(parents=True, exist_ok=False)

    def run(name, command):
        print(name + ": " + json.dumps(command), flush=True)
        if a.dry_run:
            return
        (output / (name + "_command.json")).write_text(json.dumps(command, indent=2), encoding="utf-8")
        start = time.monotonic()
        error = None
        code = None
        try:
            with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
        records.append(dict(step=name, exit_code=code, error=error, seconds=round(time.monotonic()-start, 2)))
        (output / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if code != 0 or error:
            raise RuntimeError(f"{name} failed: {error or code}; inspect {output / (name + '.log')}")
        print(name + ": PASS", flush=True)

    for profile in a.profiles:
        build = output / f"lpu{profile}"
        config = repo / ("config/ftlpu-lpu32.json" if profile == 32 else f"config/mem-profiles/ftlpu-lpu{profile}.json")
        tests = MATRIX_TESTS + (LEGACY_TESTS if profile == 32 and not a.matrix_only else [])
        configure = [a.cmake, "-S", str(repo), "-B", str(build)]
        if a.generator:
            configure += ["-G", a.generator]
        if a.compiler:
            configure += ["-DCMAKE_CXX_COMPILER=" + a.compiler]
        if a.make_program:
            configure += ["-DCMAKE_MAKE_PROGRAM=" + a.make_program]
        configure += a.cmake_arg + ["-DCMAKE_BUILD_TYPE=Debug", "-DFTLPU_HARDWARE_CONFIG=" + str(config)]
        run(f"p{profile}_configure", configure)
        run(f"p{profile}_build", [a.cmake, "--build", str(build), "--config", "Debug", "--parallel", str(a.jobs),
                               "--target", *[target for target, _ in tests]])
        regex = "^(" + "|".join(re.escape(name) for _, name in tests) + ")$"
        run(f"p{profile}_ctest", [a.ctest, "--test-dir", str(build), "-C", "Debug", "-R", regex, "-V",
                               "--no-tests=error", "--timeout", str(a.timeout),
                               "--output-junit", str(output / f"p{profile}_ctest.xml")])
    print("DRY RUN only" if a.dry_run else "PASS: " + str(output), flush=True)


if __name__ == "__main__":
    main()
