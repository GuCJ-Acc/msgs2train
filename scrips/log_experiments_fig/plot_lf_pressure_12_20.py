#!/usr/bin/env python3

from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from script_config import get_data_file


DATA_FILE = get_data_file()
LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0
HIGHLIGHT_START_TIME = 12.0
HIGHLIGHT_END_TIME = 15.8
KALMAN_HIGHLIGHT_START_TIME = 16.0
KALMAN_HIGHLIGHT_THRESHOLD = 560.0
AXIS_LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 15
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 11


# Use an IEEE-style body font family, preferring Times New Roman.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]
matplotlib.rcParams["mathtext.fontset"] = "stix"


def load_lf_pressure_data(csv_path: Path):
    relative_times = []
    lf_pressures = []
    start_time = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                lf_pressure = float(row["LF"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            lf_pressures.append(lf_pressure)

    if not relative_times:
        raise ValueError(f"No valid LF pressure data found in {csv_path}")

    return relative_times, lf_pressures


def load_time_series(csv_path: Path, value_column: str):
    times = []
    values = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                value = float(row[value_column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            times.append(time_value)
            values.append(value)

    if not times:
        raise ValueError(f"No valid '{value_column}' data found in {csv_path}")

    return times, values


def filter_time_range(times, values, start_time, end_time):
    filtered_times = []
    filtered_values = []

    for time_value, value in zip(times, values):
        if start_time <= time_value <= end_time:
            filtered_times.append(time_value)
            filtered_values.append(value)

    if not filtered_times:
        raise ValueError(
            f"No data found in the time range [{start_time}, {end_time}] s"
        )

    return filtered_times, filtered_values


def set_readable_ylim(axis, values_a, values_b):
    all_values = list(values_a) + list(values_b)
    min_value = min(all_values)
    max_value = max(all_values)

    if min_value == max_value:
        padding = max(abs(min_value) * 0.1, 1.0)
    else:
        padding = (max_value - min_value) * 0.08

    axis.set_ylim(min_value - padding, max_value + padding)


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "log_data" / DATA_FILE
    raw_csv_path = data_dir / "data.csv"
    kalman_csv_path = data_dir / "LF" / "data_lf_kalman.csv"
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "lf_pressure_12_20.png"

    relative_times, lf_pressures = load_lf_pressure_data(raw_csv_path)
    local_times, local_pressures = filter_time_range(
        relative_times,
        lf_pressures,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )
    kalman_times, kalman_values = load_time_series(kalman_csv_path, "LF_kalman")
    local_kalman_times, local_kalman_values = filter_time_range(
        kalman_times,
        kalman_values,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )
    local_kalman_times_np = np.array(local_kalman_times, dtype=float)
    local_kalman_values_np = np.array(local_kalman_values, dtype=float)
    kalman_highlight_mask = (
        (local_kalman_times_np >= KALMAN_HIGHLIGHT_START_TIME)
        & (local_kalman_values_np > KALMAN_HIGHLIGHT_THRESHOLD)
    )

    plt.figure(figsize=(12, 6))
    plt.fill_between(
        local_kalman_times_np,
        0,
        1,
        where=kalman_highlight_mask,
        color="#2ecc71",
        alpha=0.16,
        step="post",
        transform=plt.gca().get_xaxis_transform(),
        label="LF in contact during foot-based gait",
        zorder=0,
    )
    plt.plot(
        local_times,
        local_pressures,
        linewidth=1.8,
        color="#1f77b4",
        alpha=0.8,
        label="LF raw tire pressure sensor data",
    )
    plt.plot(
        local_kalman_times_np,
        local_kalman_values_np,
        linewidth=2.0,
        color="#d62728",
        label="LF Kalman-filtered tire pressure sensor data",
    )
    plt.axvspan(
        HIGHLIGHT_START_TIME,
        HIGHLIGHT_END_TIME,
        color="#f1c40f",
        alpha=0.18,
        label="LF in contact during wheeled gait",
    )
    plt.xlabel("Time (s, start at 0)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Tire Pressure Value", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(
        "Experimental Validation of the Tire Pressure-Based Contact Sensing Module",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.xlim(LOCAL_START_TIME, LOCAL_END_TIME)
    set_readable_ylim(plt.gca(), local_pressures, local_kalman_values_np)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Raw CSV: {raw_csv_path}")
    print(f"Kalman CSV: {kalman_csv_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
