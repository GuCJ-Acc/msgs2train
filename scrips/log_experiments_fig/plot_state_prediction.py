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
LOCAL_START_TIME = 150.0
LOCAL_END_TIME = 153.5
KALMAN_LEFT_SHIFT = 0.06
LABEL_FILTER_WINDOW_SIZE = 200
LABEL_FILTER_THRESHOLD = 0.6
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


def load_first_time(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        first_row = next(reader, None)

    if first_row is None:
        raise ValueError(f"No data found in {csv_path}")

    try:
        return float(first_row["time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Failed to read first time from {csv_path}") from exc


def load_relative_time_series(csv_path: Path, value_column: str, reference_start_time: float):
    relative_times = []
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

            relative_times.append(time_value - reference_start_time)
            values.append(value)

    if not relative_times:
        raise ValueError(f"No valid '{value_column}' data found in {csv_path}")

    return relative_times, values


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


def set_readable_ylim(axis, values):
    min_value = min(values)
    max_value = max(values)

    if min_value == max_value:
        padding = max(abs(min_value) * 0.1, 1.0)
    else:
        padding = (max_value - min_value) * 0.08

    axis.set_ylim(min_value - padding, max_value + padding)


def filter_rising_edge_spikes(values, window_size, activation_threshold):
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    binary_values = np.array([1.0 if value >= 0.5 else 0.0 for value in values], dtype=float)
    filtered_values = binary_values.copy()
    left_window = window_size // 2
    right_window = window_size - left_window - 1

    # Only smooth neighborhoods around 0->1 rising edges to suppress chatter
    # without changing the rest of the label sequence.
    rising_edge_indices = np.where(
        (binary_values[1:] >= 0.5) & (binary_values[:-1] < 0.5)
    )[0] + 1

    for edge_index in rising_edge_indices:
        region_start = max(0, edge_index - left_window)
        region_end = min(len(binary_values), edge_index + right_window + 1)

        for sample_index in range(region_start, region_end):
            window_start = max(0, sample_index - left_window)
            window_end = min(len(binary_values), sample_index + right_window + 1)
            window_mean = np.mean(binary_values[window_start:window_end])
            filtered_values[sample_index] = (
                1.0 if window_mean >= activation_threshold else 0.0
            )

    return filtered_values.tolist()


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "log_data" / DATA_FILE
    raw_csv_path = data_dir / "data.csv"
    kalman_csv_path = data_dir / "LF" / "data_lf_kalman.csv"
    inference_csv_path = script_dir / "data_label_with_16_155_inference_results.csv"
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "lf_state_prediction_150_155.png"

    reference_start_time = load_first_time(raw_csv_path)
    kalman_times, kalman_values = load_time_series(
        kalman_csv_path, "LF_kalman"
    )
    local_kalman_times, local_kalman_values = filter_time_range(
        kalman_times,
        kalman_values,
        LOCAL_START_TIME + KALMAN_LEFT_SHIFT,
        LOCAL_END_TIME + KALMAN_LEFT_SHIFT,
    )
    label_times, label_values = load_relative_time_series(
        inference_csv_path,
        "LF_label",
        reference_start_time,
    )
    local_label_times, local_label_values = filter_time_range(
        label_times,
        label_values,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )
    local_label_values_filtered = filter_rising_edge_spikes(
        local_label_values,
        LABEL_FILTER_WINDOW_SIZE,
        LABEL_FILTER_THRESHOLD,
    )
    pred_times, pred_values = load_relative_time_series(
        inference_csv_path,
        "LF_pred",
        reference_start_time,
    )
    local_pred_times, local_pred_values = filter_time_range(
        pred_times,
        pred_values,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )
    local_pred_times_np = np.array(local_pred_times, dtype=float)
    local_pred_values_np = np.array(local_pred_values, dtype=float)
    local_kalman_times_shifted = [
        time_value - KALMAN_LEFT_SHIFT for time_value in local_kalman_times
    ]

    fig, axis_pressure = plt.subplots(figsize=(12, 6))
    axis_pressure.fill_between(
        local_pred_times_np,
        0,
        1,
        where=local_pred_values_np >= 1.0,
        color="#3498db",
        alpha=0.14,
        step="post",
        transform=axis_pressure.get_xaxis_transform(),
        label="LF predicted label = 1",
        zorder=0,
    )
    axis_pressure.plot(
        local_kalman_times_shifted,
        local_kalman_values,
        linewidth=2.2,
        color="#d62728",
        label="LF Kalman-filtered tire pressure sensor data",
    )
    axis_pressure.set_xlabel("Time (s, start at 0)", fontsize=AXIS_LABEL_FONT_SIZE)
    axis_pressure.set_ylabel("Tire Pressure Value", fontsize=AXIS_LABEL_FONT_SIZE)
    axis_pressure.set_title("LF State Prediction on the Asphalt Terrain Dataset (150-153.5 s)", fontsize=TITLE_FONT_SIZE)
    axis_pressure.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    axis_pressure.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    axis_pressure.set_xlim(LOCAL_START_TIME, LOCAL_END_TIME)
    set_readable_ylim(axis_pressure, local_kalman_values)
    axis_pressure.grid(True, linestyle="--", alpha=0.4)

    axis_label = axis_pressure.twinx()
    axis_label.step(
        local_label_times,
        local_label_values_filtered,
        where="post",
        linewidth=1.8,
        color="#2ca02c",
        label="LF supervised label",
    )
    axis_label.set_ylabel("LF Label", fontsize=AXIS_LABEL_FONT_SIZE)
    axis_label.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    axis_label.set_ylim(-0.1, 1.2)
    axis_label.set_yticks([0, 1])

    pressure_handles, pressure_labels = axis_pressure.get_legend_handles_labels()
    label_handles, label_labels = axis_label.get_legend_handles_labels()
    axis_pressure.legend(
        [pressure_handles[0], label_handles[0], pressure_handles[1]],
        [pressure_labels[0], label_labels[0], pressure_labels[1]],
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Raw CSV: {raw_csv_path}")
    print(f"Kalman CSV: {kalman_csv_path}")
    print(f"Inference CSV: {inference_csv_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
