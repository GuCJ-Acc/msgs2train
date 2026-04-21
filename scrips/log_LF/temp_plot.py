#!/usr/bin/env python3

from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Global plot parameters. Update these values to plot any column from data.csv.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from script_config import get_data_file


DATA_FILE = get_data_file()
TARGET_COLUMN = "LH"
LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0
OUTPUT_FILENAME = None
PLOT_TITLE = None
X_LABEL = "time (s, start at 0)"
Y_LABEL = None
LINE_COLOR = "#1f77b4"
LINE_WIDTH = 1.8
FIG_SIZE = (12, 6)
GRID_ENABLED = True
FONT_FAMILY = "DejaVu Sans"
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 13
TICK_FONT_SIZE = 11


def sanitize_name(value: str):
    sanitized = []
    for char in value:
        if char.isalnum() or char in ("_", "-"):
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "plot"


def format_range_token(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def default_output_filename(column_name: str, start_time: float, end_time: float):
    safe_column_name = sanitize_name(column_name)
    return (
        f"{safe_column_name}_{format_range_token(start_time)}_"
        f"{format_range_token(end_time)}.png"
    )


def load_column_data(csv_path: Path, column_name: str):
    relative_times = []
    column_values = []
    start_time = None
    available_columns = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        available_columns = reader.fieldnames or []

        if column_name not in available_columns:
            raise ValueError(
                f"Column '{column_name}' not found in {csv_path}. "
                f"Available columns: {available_columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                column_value = float(row[column_name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            column_values.append(column_value)

    if not relative_times:
        raise ValueError(f"No valid '{column_name}' data found in {csv_path}")

    return relative_times, column_values


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


def main():
    if LOCAL_START_TIME > LOCAL_END_TIME:
        raise ValueError("LOCAL_START_TIME must be less than or equal to LOCAL_END_TIME")

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    csv_path = save_dir / "data.csv"
    fig_dir = save_dir / "LF" / "fig"
    fig_dir.mkdir(exist_ok=True)

    output_filename = OUTPUT_FILENAME or default_output_filename(
        TARGET_COLUMN,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )
    output_path = fig_dir / output_filename

    relative_times, column_values = load_column_data(csv_path, TARGET_COLUMN)
    local_times, local_values = filter_time_range(
        relative_times,
        column_values,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )

    plt.rcParams["font.family"] = FONT_FAMILY

    plt.figure(figsize=FIG_SIZE)
    plt.plot(
        local_times,
        local_values,
        linewidth=LINE_WIDTH,
        color=LINE_COLOR,
    )
    plt.xlabel(X_LABEL, fontsize=LABEL_FONT_SIZE)
    plt.ylabel(Y_LABEL or TARGET_COLUMN, fontsize=LABEL_FONT_SIZE)
    plt.title(
        PLOT_TITLE
        or f"{TARGET_COLUMN} ({LOCAL_START_TIME}s to {LOCAL_END_TIME}s)",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.xlim(LOCAL_START_TIME, LOCAL_END_TIME)
    set_readable_ylim(plt.gca(), local_values)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    if GRID_ENABLED:
        plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Column: {TARGET_COLUMN}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
