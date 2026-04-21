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
LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0


def resolve_data_csv_path(save_dir: Path):
    csv_path = save_dir / "data.csv"
    if csv_path.exists():
        return csv_path.resolve()
    raise FileNotFoundError(f"Unable to find data.csv: {csv_path}")


def load_lh_data(csv_path: Path):
    relative_times = []
    lh_values = []
    start_time = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                lh_value = float(row["LH"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            lh_values.append(lh_value)

    if not relative_times:
        raise ValueError(f"No valid LH data found in {csv_path}")

    return relative_times, lh_values


def main():
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    csv_path = resolve_data_csv_path(save_dir)
    fig_dir = save_dir / "LH" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "lh_12_20.png"

    relative_times, lh_values = load_lh_data(csv_path)

    local_times = []
    local_lh_values = []
    for time_value, lh_value in zip(relative_times, lh_values):
        if LOCAL_START_TIME <= time_value <= LOCAL_END_TIME:
            local_times.append(time_value)
            local_lh_values.append(lh_value)

    if not local_times:
        raise ValueError(
            f"No data found in the time range [{LOCAL_START_TIME}, {LOCAL_END_TIME}] s"
        )

    plt.figure(figsize=(12, 6))
    plt.plot(local_times, local_lh_values, linewidth=1.8, color="#1f77b4")
    plt.xlabel("time (s, start at 0)")
    plt.ylabel("LH")
    plt.title("LH (12s to 20s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
