#!/usr/bin/env python3

from pathlib import Path
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIME_INDEX = 0
VALUE_LF_INDEX = -4
VALUE_LH_INDEX = -3
LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0
MEDIAN_WINDOW = 9
EMA_ALPHA = 0.08


def load_data(csv_path: Path):
    times = []
    values_lf = []
    values_lh = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                time_motor_state = float(row[TIME_INDEX])
                value_lf = float(row[VALUE_LF_INDEX])
                value_lh = float(row[VALUE_LH_INDEX])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            times.append(time_motor_state)
            values_lf.append(value_lf)
            values_lh.append(value_lh)

    if not times:
        raise ValueError(f"No valid data found in {csv_path}")

    start_time = times[0]
    relative_times = [time_value - start_time for time_value in times]
    return relative_times, values_lf, values_lh


def median_filter(values, window_size):
    half_window = window_size // 2
    filtered_values = []

    for index in range(len(values)):
        left = max(0, index - half_window)
        right = min(len(values), index + half_window + 1)
        filtered_values.append(statistics.median(values[left:right]))

    return filtered_values


def exponential_moving_average(values, alpha):
    filtered_values = [values[0]]

    for value in values[1:]:
        filtered_values.append(alpha * value + (1.0 - alpha) * filtered_values[-1])

    return filtered_values


def mean_value(values):
    return sum(values) / len(values)


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / "../../log/data.csv").resolve()
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = (fig_dir / "value_lh_median_ema_filtered_12_20.png").resolve()

    relative_times, values_lf, values_lh = load_data(csv_path)
    median_filtered_lh = median_filter(values_lh, MEDIAN_WINDOW)
    smoothed_lh = exponential_moving_average(median_filtered_lh, EMA_ALPHA)

    local_times = []
    local_values_lf = []
    local_values_lh = []
    local_smoothed_lh = []
    for time_value, value_lf, value_lh, filtered_lh in zip(
        relative_times, values_lf, values_lh, smoothed_lh
    ):
        if LOCAL_START_TIME <= time_value <= LOCAL_END_TIME:
            local_times.append(time_value)
            local_values_lf.append(value_lf)
            local_values_lh.append(value_lh)
            local_smoothed_lh.append(filtered_lh)

    if not local_times:
        raise ValueError(
            f"No data found in the time range [{LOCAL_START_TIME}, {LOCAL_END_TIME}] s"
        )

    # Align the filtered LH baseline to the LF mean in the selected interval so the
    # two waveforms are easier to compare directly.
    local_bias = mean_value(local_values_lf) - mean_value(local_smoothed_lh)
    aligned_smoothed_lh = [value + local_bias for value in local_smoothed_lh]

    plt.figure(figsize=(12, 6))
    plt.plot(
        local_times,
        local_values_lf,
        linewidth=1.6,
        color="#1f77b4",
        label="Reference Value_LF",
    )
    plt.plot(
        local_times,
        local_values_lh,
        linewidth=1.0,
        color="#c0c7d1",
        alpha=0.65,
        label="Raw Value_LH",
    )
    plt.plot(
        local_times,
        aligned_smoothed_lh,
        linewidth=2.4,
        color="#ff7f0e",
        label="Median + EMA Filtered Value_LH",
    )
    plt.xlabel("time_MotorState (s, start at 0)")
    plt.ylabel("Value")
    plt.title("Value_LH filtered to match Value_LF trend (12s to 20s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved local figure to: {output_path}")


if __name__ == "__main__":
    main()
