#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from script_config import get_data_file


DATA_FILE = get_data_file()
DEFAULT_START_TIME = 16.0
DEFAULT_END_TIME = 155.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crop Force/data_label.csv by relative time and save the result under "
            "Gait_GRF with the time range in the file name."
        )
    )
    parser.add_argument(
        "--start",
        type=float,
        default=DEFAULT_START_TIME,
        help=f"Start time in seconds, default: {DEFAULT_START_TIME:g}",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=DEFAULT_END_TIME,
        help=f"End time in seconds, default: {DEFAULT_END_TIME:g}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional input CSV path, default: DATA_FILE/Force/data_label.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )

    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")

    return args


def require_file(path: Path):
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(f"Required file not found: {path}")


def format_range_token(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def make_default_output_path(output_dir: Path, start_time: float, end_time: float):
    start_token = format_range_token(start_time)
    end_token = format_range_token(end_time)
    return output_dir / f"data_label_with_{start_token}_{end_token}.csv"


def crop_rows_by_relative_time(input_path: Path, start_time: float, end_time: float):
    cropped_rows = []
    start_timestamp = None

    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []

        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse time on row {row_number} from {input_path}: {row}"
                ) from exc

            if start_timestamp is None:
                start_timestamp = time_value

            relative_time = time_value - start_timestamp
            if start_time <= relative_time <= end_time:
                cropped_rows.append(row)

    if start_timestamp is None:
        raise ValueError(f"No valid data found in {input_path}")

    if not cropped_rows:
        raise ValueError(f"No rows found in relative time range [{start_time}, {end_time}] s")

    return fieldnames, cropped_rows


def save_csv(output_path: Path, fieldnames, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    output_dir = save_dir / "Gait_GRF"

    input_path = require_file(
        args.input.resolve() if args.input is not None else save_dir / "Force" / "data_label.csv"
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else make_default_output_path(output_dir, args.start, args.end)
    )

    fieldnames, cropped_rows = crop_rows_by_relative_time(
        input_path,
        args.start,
        args.end,
    )
    save_csv(output_path, fieldnames, cropped_rows)

    print(f"Input CSV: {input_path}")
    print(f"Saved cropped CSV to: {output_path}")
    print(f"Relative time range: {args.start:g}-{args.end:g}s")
    print(f"Rows saved: {len(cropped_rows)}")


if __name__ == "__main__":
    main()
