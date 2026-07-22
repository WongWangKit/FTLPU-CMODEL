#!/usr/bin/env python3
"""Render selected cycle windows from an ICU schedule CSV as a scalable SVG."""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    resource: str
    detail: str


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    title: str


DEFAULT_WINDOWS = (
    Window(0, 204, "Q projection: first reduction block"),
    Window(3348, 3432, "Q RoPE: FP32 rotate, FP16 cast, and MEM writeback"),
    Window(32320, 32490, "V projection: FP16 cast -> packed 16-stream MEM layout"),
    Window(33148, 33792, "QK: four MXMs, independent query blocks"),
    Window(37440, 37880, "Softmax: pipelined P1 max, P2 exp/sum, and P3 normalize/cast"),
    Window(52288, 52692, "Post-softmax P layout: packed x16 blocks at II=4"),
    Window(52692, 52980, "P x V: passive cross-hemisphere streams, V IW, and direct P replay"),
    Window(63108, 63420, "o_proj: first reduction window"),
)


COLORS = {
    "MEM.Read": "#8fc8a7",
    "MEM.Write": "#f0ca78",
    "MEM.Accumulate.Sram": "#c5a3d9",
    "MEM.Accumulate.Stream": "#e16b6f",
    "SXM.Transpose": "#71c3bc",
    "SXM.Permute": "#4ca9a0",
    "SXM.Tail": "#c7e5e1",
    "VXM": "#ed996d",
    "MXM.Load": "#91b7e5",
    "MXM.Compute": "#6f9fd8",
    "MXM.Tail": "#dce5ef",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="START:END:TITLE",
        help="Cycle window to render; may be repeated.",
    )
    return parser.parse_args()


def parse_windows(values: list[str]) -> tuple[Window, ...]:
    if not values:
        return DEFAULT_WINDOWS
    windows = []
    for value in values:
        fields = value.split(":", 2)
        if len(fields) != 3:
            raise ValueError(f"invalid window: {value}")
        start, end = int(fields[0]), int(fields[1])
        if end <= start:
            raise ValueError(f"window end must exceed start: {value}")
        windows.append(Window(start, end, fields[2]))
    return tuple(windows)


def load_events(path: Path, windows: tuple[Window, ...]) -> list[Event]:
    low = min(window.start for window in windows)
    high = max(window.end for window in windows)
    events = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            start, end = int(row["start"]), int(row["end"])
            if end <= low or start >= high:
                continue
            if row["resource"].endswith(".Tail"):
                continue
            events.append(Event(start, end, row["resource"], row["detail"]))
    return events


def resource_key(resource: str) -> tuple[int, int, int, str]:
    hemisphere = 0 if ".E" in resource else 1 if ".W" in resource else 2
    if resource.startswith("MEM"):
        if resource.endswith("Accumulate"):
            return (3, hemisphere, 3, resource)
        op = 0 if resource.endswith("Read") else 1
        return (0, hemisphere, op, resource)
    if resource.startswith("SXM"):
        op = 0 if resource.endswith("Transpose") else 1 if resource.endswith("Permute") else 2
        return (1, hemisphere, op, resource)
    if resource.startswith("VXM"):
        match = re.search(r"ALU(\d+)", resource)
        alu = int(match.group(1)) if match else 20
        return (2, 0, alu, resource)
    if resource.startswith("MXM"):
        op = 0 if resource.endswith("Load") else 1 if resource.endswith("Compute") else 2
        return (3, hemisphere, op, resource)
    return (4, hemisphere, 0, resource)


def event_color(event: Event) -> str:
    resource = event.resource
    if resource.startswith("MEM"):
        if resource.endswith(".Accumulate"):
            destination = "Stream" if "dst=stream+clear" in event.detail else "Sram"
            return COLORS["MEM.Accumulate." + destination]
        return COLORS["MEM." + resource.rsplit(".", 1)[-1]]
    if resource.startswith("SXM"):
        return COLORS["SXM." + resource.rsplit(".", 1)[-1]]
    if resource.startswith("VXM"):
        return COLORS["VXM"]
    if resource.endswith(".Load"):
        return COLORS["MXM.Load"]
    if resource.endswith(".Compute"):
        return COLORS["MXM.Compute"]
    return COLORS["MXM.Tail"]


def detail_class(event: Event) -> str:
    if event.resource.startswith("MEM"):
        if event.resource.endswith(".Read"):
            return "continuous reads"
        if event.resource.endswith(".Write"):
            return "continuous writes"
        destination = "stream + clear" if "dst=stream+clear" in event.detail else "SRAM"
        return f"continuous accumulates -> {destination}"
    if event.resource.endswith(".Load"):
        return re.sub(r" column=\d+", "", event.detail)
    return event.detail


def coalesce(events: list[Event]) -> list[tuple[Event, int]]:
    grouped: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for event in events:
        grouped[(event.resource, detail_class(event))].append(event)

    merged: list[tuple[Event, int]] = []
    for (resource, detail), group in grouped.items():
        ordered = sorted(group, key=lambda event: (event.start, event.end))
        start = ordered[0].start
        end = ordered[0].end
        count = 1
        for event in ordered[1:]:
            if event.start <= end:
                end = max(end, event.end)
                count += 1
                continue
            merged.append((Event(start, end, resource, detail), count))
            start, end, count = event.start, event.end, 1
        merged.append((Event(start, end, resource, detail), count))

    return sorted(merged, key=lambda item: (item[0].start, resource_key(item[0].resource)))


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def render(events: list[Event], windows: tuple[Window, ...], output: Path) -> None:
    width = 1800
    left = 245
    right = 45
    plot_width = width - left - right
    row_height = 26
    panel_gap = 54
    header_height = 124

    panel_data = []
    total_height = header_height
    for window in windows:
        selected = [event for event in events if event.end > window.start and event.start < window.end]
        resources = sorted({event.resource for event in selected}, key=resource_key)
        panel_height = 55 + len(resources) * row_height
        panel_data.append((window, selected, resources, total_height, panel_height))
        total_height += panel_height + panel_gap
    total_height += 36

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
        "<style>",
        ".title{font:700 29px 'Segoe UI',Arial,sans-serif;fill:#17212b}",
        ".sub{font:14px 'Segoe UI',Arial,sans-serif;fill:#5d6874}",
        ".panel{font:700 18px 'Segoe UI',Arial,sans-serif;fill:#25313c}",
        ".row{font:600 12px 'Segoe UI',Arial,sans-serif;fill:#34404b}",
        ".tick{font:11px 'Segoe UI',Arial,sans-serif;fill:#66717d}",
        ".bar{font:600 10px 'Segoe UI',Arial,sans-serif;fill:#17212b}",
        ".grid{stroke:#dce2e8;stroke-width:1}",
        ".lane{fill:#fafbfc;stroke:#e1e6eb;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="42" y="43" class="title">SmolLM2 Attention: Exact ICU Schedule Windows</text>',
        '<text x="42" y="69" class="sub">Bars are tile0 instruction issue intervals; MXM/SXM drain tails are omitted. Hover for cycle, resource, slice/stream/address, and repeat details.</text>',
        '<rect x="42" y="82" width="18" height="12" rx="2" fill="#c5a3d9" stroke="#52606c" stroke-width="0.7"/>',
        '<text x="68" y="93" class="sub">Accumulator -&gt; SRAM (keep partial sum)</text>',
        '<rect x="340" y="82" width="18" height="12" rx="2" fill="#e16b6f" stroke="#52606c" stroke-width="0.7"/>',
        '<text x="366" y="93" class="sub">Accumulator -&gt; stream + clear (final sum)</text>',
    ]

    for window, selected, resources, y0, panel_height in panel_data:
        scale = plot_width / (window.end - window.start)
        lines.append(f'<text x="42" y="{y0 + 24}" class="panel">{esc(window.title)}</text>')
        lines.append(
            f'<text x="{width - 45}" y="{y0 + 24}" text-anchor="end" class="sub">'
            f'cycles {window.start}..{window.end} ({window.end - window.start} cycles)</text>'
        )
        plot_y = y0 + 42
        for tick in range(6):
            cycle = window.start + round((window.end - window.start) * tick / 5)
            x = left + plot_width * tick / 5
            lines.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{y0 + panel_height}" class="grid"/>')
            lines.append(f'<text x="{x:.2f}" y="{plot_y - 7}" text-anchor="middle" class="tick">{cycle}</text>')

        resource_y = {}
        for index, resource in enumerate(resources):
            y = plot_y + index * row_height
            resource_y[resource] = y
            lines.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="{row_height - 2}" class="lane"/>')
            lines.append(f'<text x="{left - 12}" y="{y + 17}" text-anchor="end" class="row">{esc(resource)}</text>')

        for event, count in coalesce(selected):
            clipped_start = max(event.start, window.start)
            clipped_end = min(event.end, window.end)
            x = left + (clipped_start - window.start) * scale
            bar_width = max(1.5, (clipped_end - clipped_start) * scale)
            y = resource_y[event.resource] + 3
            color = event_color(event)
            tooltip = f'{event.resource}: cycles {event.start}..{event.end}; {event.detail}'
            if count > 1:
                tooltip += f'; {count} coalesced events'
            lines.append(
                f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="{row_height - 8}" '
                f'rx="2" fill="{color}" stroke="#52606c" stroke-width="0.7" opacity="0.92">'
                f'<title>{esc(tooltip)}</title></rect>'
            )
            if bar_width >= 58:
                label = event.detail
                if count > 1:
                    label += f' x{count}'
                max_chars = max(5, int(bar_width / 7.2))
                if len(label) > max_chars:
                    label = label[: max_chars - 1] + "..."
                lines.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{y + 13}" text-anchor="middle" class="bar" '
                    f'pointer-events="none">{esc(label)}</text>'
                )

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.window)
    render(load_events(args.input, windows), windows, args.output)


if __name__ == "__main__":
    main()
