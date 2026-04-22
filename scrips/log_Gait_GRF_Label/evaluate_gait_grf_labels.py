#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from script_config import get_data_file


DATA_FILE = get_data_file()
LEGS = ["LF", "LH", "RF", "RH"]


def require_file(path: Path):
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(f"Required file not found: {path}")


def parse_label(row, column_name: str, csv_path: Path, row_number: int):
    try:
        value = int(float(row[column_name]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to parse column '{column_name}' on row {row_number} from {csv_path}: {row}"
        ) from exc

    if value not in (0, 1):
        raise ValueError(
            f"Expected binary label 0/1 in column '{column_name}' on row {row_number} "
            f"from {csv_path}, got: {value}"
        )

    return value


def make_empty_counts():
    return {leg: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0} for leg in LEGS}


def update_counts(counts, leg: str, truth_value: int, pred_value: int):
    leg_counts = counts[leg]
    leg_counts["total"] += 1

    if truth_value == 1 and pred_value == 1:
        leg_counts["tp"] += 1
    elif truth_value == 0 and pred_value == 0:
        leg_counts["tn"] += 1
    elif truth_value == 0 and pred_value == 1:
        leg_counts["fp"] += 1
    elif truth_value == 1 and pred_value == 0:
        leg_counts["fn"] += 1


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def calc_metrics_from_counts(counts):
    total = counts["total"]
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    return {
        "accuracy": safe_divide(tp + tn, total),
        "fpr": safe_divide(fp, fp + tn),
        "fnr": safe_divide(fn, fn + tp),
        "counts": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total": total,
        },
    }


def average_metric(leg_metrics, metric_name: str):
    values = [
        metrics[metric_name]
        for metrics in leg_metrics.values()
        if metrics[metric_name] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_prediction_file(
    truth_csv_path: Path,
    pred_csv_path: Path,
    pred_column_suffix: str,
):
    counts = make_empty_counts()
    compared_rows = 0

    with truth_csv_path.open("r", encoding="utf-8", newline="") as truth_file, pred_csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as pred_file:
        truth_reader = csv.DictReader(truth_file)
        pred_reader = csv.DictReader(pred_file)

        for row_number, (truth_row, pred_row) in enumerate(
            zip(truth_reader, pred_reader), start=2
        ):
            compared_rows += 1

            truth_time = truth_row.get("time")
            pred_time = pred_row.get("time")
            if truth_time != pred_time:
                raise ValueError(
                    f"Time mismatch on row {row_number}: truth={truth_time}, prediction={pred_time}"
                )

            for leg in LEGS:
                truth_value = parse_label(
                    truth_row,
                    f"{leg}_contact_states",
                    truth_csv_path,
                    row_number,
                )
                pred_value = parse_label(
                    pred_row,
                    f"{leg}_{pred_column_suffix}",
                    pred_csv_path,
                    row_number,
                )
                update_counts(counts, leg, truth_value, pred_value)

        extra_truth = next(truth_reader, None)
        extra_pred = next(pred_reader, None)
        if extra_truth is not None or extra_pred is not None:
            raise ValueError(
                f"Row count mismatch between {truth_csv_path} and {pred_csv_path}"
            )

    if compared_rows == 0:
        raise ValueError(f"No rows compared for {pred_csv_path}")

    leg_metrics = {leg: calc_metrics_from_counts(counts[leg]) for leg in LEGS}
    average_metrics = {
        "accuracy": average_metric(leg_metrics, "accuracy"),
        "fpr": average_metric(leg_metrics, "fpr"),
        "fnr": average_metric(leg_metrics, "fnr"),
    }

    return {
        "csv": str(pred_csv_path),
        "compared_rows": compared_rows,
        "average": average_metrics,
        "legs": leg_metrics,
    }


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def print_metrics_summary(result):
    print("\nEvaluation summary:")
    for name, evaluation in result["evaluations"].items():
        average = evaluation["average"]
        print(f"\n{name}:")
        print(f"  rows: {evaluation['compared_rows']}")
        print(
            "  average: "
            f"accuracy={format_metric(average['accuracy'])}, "
            f"FPR={format_metric(average['fpr'])}, "
            f"FNR={format_metric(average['fnr'])}"
        )

        for leg in LEGS:
            metrics = evaluation["legs"][leg]
            counts = metrics["counts"]
            print(
                f"  {leg}: "
                f"accuracy={format_metric(metrics['accuracy'])}, "
                f"FPR={format_metric(metrics['fpr'])}, "
                f"FNR={format_metric(metrics['fnr'])}, "
                f"TP={counts['tp']}, TN={counts['tn']}, "
                f"FP={counts['fp']}, FN={counts['fn']}"
            )


def main():
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    force_dir = save_dir / "Force"
    gait_grf_dir = save_dir / "Gait_GRF"

    truth_csv_path = require_file(force_dir / "data_label.csv")
    force_label_csv_path = require_file(gait_grf_dir / "data_label_with_force.csv")
    gait_label_csv_path = require_file(gait_grf_dir / "data_label_with_gait.csv")
    output_json_path = gait_grf_dir / "label_evaluation_metrics.json"

    result = {
        "data_file": DATA_FILE,
        "truth_csv": str(truth_csv_path),
        "metrics_definition": {
            "accuracy": "(TP + TN) / total",
            "fpr": "FP / (FP + TN)",
            "fnr": "FN / (FN + TP)",
            "positive_label": 1,
            "negative_label": 0,
        },
        "evaluations": {
            "force": evaluate_prediction_file(
                truth_csv_path,
                force_label_csv_path,
                "contact_states",
            ),
            "gait": evaluate_prediction_file(
                truth_csv_path,
                gait_label_csv_path,
                "gait_contact_states",
            ),
        },
    }

    with output_json_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print(f"Truth CSV: {truth_csv_path}")
    print(f"Saved metrics JSON to: {output_json_path}")
    print_metrics_summary(result)


if __name__ == "__main__":
    main()
