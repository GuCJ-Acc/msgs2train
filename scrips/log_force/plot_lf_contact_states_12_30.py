#!/usr/bin/env python3

from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from script_config import get_data_file


DATA_FILE = get_data_file()
LOCAL_START_TIME = 150.0
LOCAL_END_TIME = 169.0
TARGET_COLUMN = "LH_contact_states"


def resolve_label_csv_path(force_dir: Path):
    csv_path = force_dir / "data_label.csv"
    if csv_path.exists():
        return csv_path.resolve()
    raise FileNotFoundError(f"Unable to find data_label.csv: {csv_path}")


def load_contact_state_data(csv_path: Path, column_name: str):
    relative_times = []
    contact_states = []
    start_time = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if column_name not in fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found in {csv_path}. Available columns: {fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                contact_state = float(row[column_name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            contact_states.append(contact_state)

    if not relative_times:
        raise ValueError(f"No valid '{column_name}' data found in {csv_path}")

    return relative_times, contact_states


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


def main():
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    force_dir = save_dir / "Force"
    csv_path = resolve_label_csv_path(force_dir)
    fig_dir = force_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "lf_contact_states_12_30.png"

    relative_times, contact_states = load_contact_state_data(csv_path, TARGET_COLUMN)
    local_times, local_states = filter_time_range(
        relative_times,
        contact_states,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )

    plt.figure(figsize=(12, 4.5))
    plt.step(
        local_times,
        local_states,
        where="post",
        linewidth=1.8,
        color="#1f77b4",
    )
    plt.xlabel("time (s, start at 0)")
    plt.ylabel(TARGET_COLUMN)
    plt.title("LF contact states (12s to 30s)")
    plt.xlim(LOCAL_START_TIME, LOCAL_END_TIME)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Input CSV: {csv_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
