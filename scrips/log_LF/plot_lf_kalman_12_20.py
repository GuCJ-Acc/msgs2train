#!/usr/bin/env python3

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_FILE = "data_09"
LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0
PROCESS_NOISE = 1e-2
MEASUREMENT_NOISE = 25.0


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


def kalman_filter_1d(values, process_noise, measurement_noise):
    filtered_values = []

    estimate = values[0]
    estimate_covariance = 1.0

    for measurement in values:
        estimate_covariance += process_noise

        kalman_gain = estimate_covariance / (estimate_covariance + measurement_noise)
        estimate = estimate + kalman_gain * (measurement - estimate)
        estimate_covariance = (1.0 - kalman_gain) * estimate_covariance

        filtered_values.append(estimate)

    return filtered_values


def save_lf_kalman_csv(output_path: Path, relative_times, filtered_values):
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "LF_kalman"])
        for time_value, filtered_value in zip(relative_times, filtered_values):
            writer.writerow([time_value, filtered_value])


def main():
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    csv_path = save_dir / "data.csv"
    output_csv_path = save_dir / "LF" / "data_lf_kalman.csv"
    fig_dir = save_dir / "LF" / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = fig_dir / "lf_kalman_12_20.png"

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

    filtered_lf_values = kalman_filter_1d(
        local_lf_values,
        process_noise=PROCESS_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
    )
    save_lf_kalman_csv(output_csv_path, local_times, filtered_lf_values)

    plt.figure(figsize=(12, 6))
    plt.plot(local_times, filtered_lf_values, linewidth=2.0, color="#d62728")
    plt.xlabel("time (s, start at 0)")
    plt.ylabel("LF")
    plt.title("LF with Kalman Filtering (12s to 20s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved data to: {output_csv_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
