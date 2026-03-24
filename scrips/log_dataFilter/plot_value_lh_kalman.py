#!/usr/bin/env python3

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIME_INDEX = 0
VALUE_LH_INDEX = -3
PROCESS_NOISE = 1e-2
MEASUREMENT_NOISE = 25.0


def load_data(csv_path: Path):
    times = []
    values_lh = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                time_motor_state = float(row[TIME_INDEX])
                value_lh = float(row[VALUE_LH_INDEX])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            times.append(time_motor_state)
            values_lh.append(value_lh)

    if not times:
        raise ValueError(f"No valid data found in {csv_path}")

    start_time = times[0]
    relative_times = [time_value - start_time for time_value in times]
    return relative_times, values_lh


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


def plot_figure(
    times,
    raw_values,
    filtered_values,
    output_path: Path,
    title: str,
    raw_linewidth: float,
    filtered_linewidth: float,
):
    plt.figure(figsize=(12, 6))
    plt.plot(
        times,
        raw_values,
        linewidth=raw_linewidth,
        color="#c0c7d1",
        alpha=0.65,
        label="Raw Value_LH",
    )
    plt.plot(
        times,
        filtered_values,
        linewidth=filtered_linewidth,
        color="#ff7f0e",
        label="Kalman Filtered Value_LH",
    )
    plt.xlabel("time_MotorState (s, start at 0)")
    plt.ylabel("Value_LH")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / "../../log/data.csv").resolve()
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = (fig_dir / "value_lh_kalman_filtered.png").resolve()

    relative_times, values_lh = load_data(csv_path)
    filtered_values_lh = kalman_filter_1d(
        values_lh,
        process_noise=PROCESS_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
    )

    plot_figure(
        relative_times,
        values_lh,
        filtered_values_lh,
        output_path,
        "Value_LH with Kalman Filtering",
        0.9,
        1.5,
    )

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
