"""Print comparable latency summaries from logs/benchmark.csv."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from config import get_settings


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    path: Path = get_settings().log_file
    if not path.exists():
        print(f"No benchmark log found at {path}")
        return

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (row["platform"], row["backend"], row["device"], f'{row["width"]}x{row["height"]}')
            groups[key].append(row)

    for (platform_name, backend, device, resolution), rows in sorted(groups.items()):
        success_rows = [row for row in rows if row.get("success", "").lower() == "true"]
        latencies = [float(row["generation_seconds"]) for row in success_rows if row.get("generation_seconds")]
        print(f"\n{platform_name} / {backend} / {device} / {resolution}")
        print(f"success: {len(success_rows)}  fail: {len(rows) - len(success_rows)}")
        if latencies:
            print(f"avg: {sum(latencies) / len(latencies):.2f}s")
            print(f"p50: {percentile(latencies, 0.50):.2f}s")
            print(f"p95: {percentile(latencies, 0.95):.2f}s")


if __name__ == "__main__":
    main()
