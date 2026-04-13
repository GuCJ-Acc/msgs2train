#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_FILE = "data_09"
DEFAULT_START_TIME = 12.0
DEFAULT_END_TIME = 20.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read raw RF data from data.csv and filtered RF data from "
            "data_rf_kalman.csv, then plot them for a selected time range."
        )
    )
    parser.add_argument(
        "--start",
        type=float,
        default=DEFAULT_START_TIME,
        help=f"Start time in seconds, default: {DEFAULT_START_TIME}",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=DEFAULT_END_TIME,
        help=f"End time in seconds, default: {DEFAULT_END_TIME}",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Optional path to raw RF CSV file",
    )
    parser.add_argument(
        "--kalman-csv",
        type=Path,
        default=None,
        help="Optional path to filtered RF CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to the output figure",
    )

    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")

    return args


def load_raw_rf_data(csv_path: Path):
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


def load_kalman_rf_data(csv_path: Path):
    relative_times = []
    rf_values = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                rf_value = float(row["RF_kalman"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            relative_times.append(time_value)
            rf_values.append(rf_value)

    if not relative_times:
        raise ValueError(f"No valid RF kalman data found in {csv_path}")

    return relative_times, rf_values


def filter_time_range(times, values, start_time, end_time):
    filtered_times = []
    filtered_values = []

    for time_value, value in zip(times, values):
        if start_time <= time_value <= end_time:
            filtered_times.append(time_value)
            filtered_values.append(value)

    return filtered_times, filtered_values


def require_nonempty(times, start_time, end_time, data_name, csv_path: Path):
    if times:
        return

    raise ValueError(
        f"No {data_name} found in the time range [{start_time}, {end_time}] s "
        f"from {csv_path}"
    )


def format_range_token(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def set_readable_ylim(axis, values_a, values_b):
    all_values = list(values_a) + list(values_b)
    min_value = min(all_values)
    max_value = max(all_values)

    if min_value == max_value:
        padding = max(abs(min_value) * 0.1, 1.0)
    else:
        padding = (max_value - min_value) * 0.08

    axis.set_ylim(min_value - padding, max_value + padding)


def resolve_path(script_dir: Path, explicit_path: Optional[Path], default_names):
    if explicit_path is not None:
        return explicit_path.resolve()

    for default_name in default_names:
        candidate = (script_dir / default_name).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve a valid path from defaults: {default_names}"
    )


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    fig_dir = save_dir / "RF" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = resolve_path(save_dir, args.raw_csv, ["data.csv"])
    kalman_csv_path = resolve_path(
        save_dir / "RF",
        args.kalman_csv,
        ["data_rf_kalman.csv"],
    )

    if args.output is None:
        range_token = f"{format_range_token(args.start)}_{format_range_token(args.end)}"
        output_path = fig_dir / f"rf_kalman_compare_{range_token}.png"
    else:
        output_path = args.output.resolve()

    raw_times, raw_values = load_raw_rf_data(raw_csv_path)
    kalman_times, kalman_values = load_kalman_rf_data(kalman_csv_path)

    raw_times, raw_values = filter_time_range(raw_times, raw_values, args.start, args.end)
    kalman_times, kalman_values = filter_time_range(
        kalman_times, kalman_values, args.start, args.end
    )

    require_nonempty(raw_times, args.start, args.end, "raw RF data", raw_csv_path)
    require_nonempty(
        kalman_times,
        args.start,
        args.end,
        "kalman RF data",
        kalman_csv_path,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        raw_times,
        raw_values,
        linewidth=1.2,
        color="#1f77b4",
        alpha=0.75,
        label="raw",
    )
    plt.plot(
        kalman_times,
        kalman_values,
        linewidth=2.0,
        color="#d62728",
        label="kalman",
    )
    plt.xlabel("time (s, start at 0)")
    plt.ylabel("RF")
    plt.title(f"RF raw vs kalman ({args.start}s to {args.end}s)")
    plt.xlim(args.start, args.end)
    set_readable_ylim(plt.gca(), raw_values, kalman_values)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
