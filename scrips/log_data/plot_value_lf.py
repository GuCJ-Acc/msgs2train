#!/usr/bin/env python3

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIME_INDEX = 0
VALUE_LF_INDEX = -4


def load_data(csv_path: Path):
    times = []
    values_lf = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                time_motor_state = float(row[TIME_INDEX])
                value_lf = float(row[VALUE_LF_INDEX])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            times.append(time_motor_state)
            values_lf.append(value_lf)

    if not times:
        raise ValueError(f"No valid data found in {csv_path}")

    start_time = times[0]
    relative_times = [time_value - start_time for time_value in times]
    return relative_times, values_lf


def plot_figure(times, values, output_path: Path, title: str):
    plt.figure(figsize=(12, 6))
    plt.plot(times, values, linewidth=1.2, color="#1f77b4")
    plt.xlabel("time_MotorState (s, start at 0)")
    plt.ylabel("Value_LF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / "../../log/data.csv").resolve()
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = fig_dir / "fig_value_lf.png"

    relative_times, values_lf = load_data(csv_path)

    plot_figure(
        relative_times,
        values_lf,
        output_path,
        "Value_LF vs time_MotorState",
    )

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
