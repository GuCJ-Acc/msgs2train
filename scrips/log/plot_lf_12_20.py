#!/usr/bin/env python3

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0


def load_lf_data(csv_path: Path):
    relative_times = []
    lf_values = []
    start_time = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                lf_value = float(row["LF"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            lf_values.append(lf_value)

    if not relative_times:
        raise ValueError(f"No valid LF data found in {csv_path}")

    return relative_times, lf_values


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "data.csv"
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = fig_dir / "lf_12_20.png"

    relative_times, lf_values = load_lf_data(csv_path)

    local_times = []
    local_lf_values = []
    for time_value, lf_value in zip(relative_times, lf_values):
        if LOCAL_START_TIME <= time_value <= LOCAL_END_TIME:
            local_times.append(time_value)
            local_lf_values.append(lf_value)

    if not local_times:
        raise ValueError(
            f"No data found in the time range [{LOCAL_START_TIME}, {LOCAL_END_TIME}] s"
        )

    plt.figure(figsize=(12, 6))
    plt.plot(local_times, local_lf_values, linewidth=1.8, color="#1f77b4")
    plt.xlabel("time (s, start at 0)")
    plt.ylabel("LF")
    plt.title("LF (12s to 20s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
