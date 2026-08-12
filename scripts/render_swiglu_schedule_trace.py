#!/usr/bin/env python3
"""Render the dual-hemisphere W8A16 SwiGLU ICU schedule detail."""

from __future__ import annotations

import argparse
from pathlib import Path

from render_schedule_trace import Window, load_events, render


WINDOWS = (
    Window(0, 190, "Gate/up projection: first output pair and K reduction"),
    Window(2110, 2420, "Gate/up final reduction streams directly into fused SwiGLU"),
    Window(59370, 59675, "Final gate/up block, fused SwiGLU, and MEM writeback"),
    Window(59674, 59884, "Down projection: first 128 output columns and K reduction"),
    Window(90553, 90817, "Down projection: final reduction, FP16 cast, and writeback"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(
        load_events(args.input, WINDOWS),
        WINDOWS,
        args.output,
        "Dual-Hemisphere W8A16 FFN: Exact ICU Schedule Windows",
        "X[128,576] -> gate/up[128,1536] -> SwiGLU[128,1536] -> down[128,576]. MXM tails are omitted; hover bars for exact cycles and operands.",
    )


if __name__ == "__main__":
    main()
