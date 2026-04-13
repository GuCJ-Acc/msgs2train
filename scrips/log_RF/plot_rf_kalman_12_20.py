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


def resolve_data_csv_path(save_dir: Path):
    csv_path = save_dir / "data.csv"
    if csv_path.exists():
        return csv_path.resolve()
    raise FileNotFoundError(f"Unable to find data.csv: {csv_path}")


def load_rf_data(csv_path: Path):
    relative_times = []
    rf_values = []
    start_time = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                rf_value = float(row["RF"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            rf_values.append(rf_value)

    if not relative_times:
        raise ValueError(f"No valid RF data found in {csv_path}")

    return relative_times, rf_values


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


def save_rf_kalman_csv(output_path: Path, relative_times, filtered_values):
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "RF_kalman"])
        for time_value, filtered_value in zip(relative_times, filtered_values):
            writer.writerow([time_value, filtered_value])


def main():
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    csv_path = resolve_data_csv_path(save_dir)
    output_csv_path = save_dir / "RF" / "data_rf_kalman.csv"
    fig_dir = save_dir / "RF" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "rf_kalman_12_20.png"

    relative_times, rf_values = load_rf_data(csv_path)

    local_times = []
    local_rf_values = []
    for time_value, rf_value in zip(relative_times, rf_values):
        if LOCAL_START_TIME <= time_value <= LOCAL_END_TIME:
            local_times.append(time_value)
            local_rf_values.append(rf_value)

    if not local_times:
        raise ValueError(
            f"No data found in the time range [{LOCAL_START_TIME}, {LOCAL_END_TIME}] s"
        )

    filtered_rf_values = kalman_filter_1d(
        local_rf_values,
        process_noise=PROCESS_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
    )
    save_rf_kalman_csv(output_csv_path, local_times, filtered_rf_values)

    plt.figure(figsize=(12, 6))
    plt.plot(local_times, filtered_rf_values, linewidth=2.0, color="#d62728")
    plt.xlabel("time (s, start at 0)")
    plt.ylabel("RF")
    plt.title("RF with Kalman Filtering (12s to 20s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved data to: {output_csv_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
