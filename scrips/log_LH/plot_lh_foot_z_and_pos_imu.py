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
            "Read LH foot force and LH foot position CSV files, then plot a selected "
            "time range with dual y-axes."
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
        "--force-csv",
        type=Path,
        default=None,
        help="Optional path to the LH foot force CSV file",
    )
    parser.add_argument(
        "--pos-csv",
        type=Path,
        default=None,
        help="Optional path to the LH foot position IMU CSV file",
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


def resolve_input_path(script_dir: Path, explicit_path: Optional[Path], candidates):
    if explicit_path is not None:
        return explicit_path.resolve()

    for candidate in candidates:
        candidate_path = script_dir / candidate
        if candidate_path.exists():
            return candidate_path.resolve()

    raise FileNotFoundError(
        f"None of the candidate files exist under {script_dir}: {candidates}"
    )


def load_time_series(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != 2 or fieldnames[0] != "time":
            raise ValueError(
                f"Expected two columns with the first one named 'time' in {csv_path}, "
                f"got: {fieldnames}"
            )

        value_key = fieldnames[1]
        times = []
        values = []

        for row_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row["time"]))
                values.append(float(row[value_key]))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

    if not times:
        raise ValueError(f"No valid data found in {csv_path}")

    return value_key, times, values


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
        padding = (max_value - min_value) * 0.1

    axis.set_ylim(min_value - padding, max_value + padding)


def format_range_token(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    fig_dir = save_dir / "LH" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    force_csv_path = resolve_input_path(
        save_dir / "LH",
        args.force_csv,
        ["data_lh_foot_z.csv"],
    )
    pos_csv_path = resolve_input_path(
        save_dir / "LH",
        args.pos_csv,
        ["data_lh_foot_pos_imu_z.csv"],
    )

    if args.output is None:
        range_token = f"{format_range_token(args.start)}_{format_range_token(args.end)}"
        output_path = fig_dir / f"lh_foot_z_and_pos_imu_{range_token}.png"
    else:
        output_path = args.output.resolve()

    force_label, force_times, force_values = load_time_series(force_csv_path)
    pos_label, pos_times, pos_values = load_time_series(pos_csv_path)

    force_times, force_values = filter_time_range(
        force_times, force_values, args.start, args.end
    )
    pos_times, pos_values = filter_time_range(pos_times, pos_values, args.start, args.end)

    fig, axis_force = plt.subplots(figsize=(12, 6))
    axis_pos = axis_force.twinx()

    force_line = axis_force.plot(
        force_times,
        force_values,
        linewidth=1.8,
        color="#d62728",
        label=force_label,
    )[0]
    pos_line = axis_pos.plot(
        pos_times,
        pos_values,
        linewidth=1.8,
        color="#1f77b4",
        label=pos_label,
    )[0]

    axis_force.set_xlabel("time (s)")
    axis_force.set_ylabel(force_label, color=force_line.get_color())
    axis_pos.set_ylabel(pos_label, color=pos_line.get_color())
    axis_force.tick_params(axis="y", colors=force_line.get_color())
    axis_pos.tick_params(axis="y", colors=pos_line.get_color())
    axis_force.grid(True, linestyle="--", alpha=0.4)
    axis_force.set_title(
        f"LH foot force and LH foot pos imu z ({args.start}s to {args.end}s)"
    )
    axis_force.set_xlim(args.start, args.end)

    set_readable_ylim(axis_force, force_values)
    set_readable_ylim(axis_pos, pos_values)

    lines = [force_line, pos_line]
    labels = [line.get_label() for line in lines]
    axis_force.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
