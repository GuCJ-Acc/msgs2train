#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from script_config import get_data_file


DATA_FILE = get_data_file()
FOOT_POS_IMU_Z_THRESHOLD = -0.32

LEG_SETTINGS = {
    "LF": {"leftFoot": 1.0, "frontFoot": 1.0},
    "LH": {"leftFoot": 1.0, "frontFoot": -1.0},
    "RF": {"leftFoot": -1.0, "frontFoot": 1.0},
    "RH": {"leftFoot": -1.0, "frontFoot": -1.0},
}


def resolve_data_csv_path(save_dir: Path):
    csv_path = save_dir / "data.csv"
    if csv_path.exists():
        return csv_path.resolve()
    raise FileNotFoundError(f"Unable to find data.csv: {csv_path}")


def load_wl_config(config_path: Path):
    config = {}
    in_target_section = False

    with config_path.open("r", encoding="utf-8") as config_file:
        for raw_line in config_file:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue

            stripped = line.strip()
            if stripped == "LimxDynamic_WL:":
                in_target_section = True
                continue

            if not in_target_section:
                continue

            if not raw_line.startswith("  "):
                break

            key, value = stripped.split(":", 1)
            config[key.strip()] = float(value.strip())

    required_keys = [
        "L1",
        "L2",
        "L3",
        "offset_Angle_HAA",
        "offset_Angle_HFE",
        "offset_Angle_KFE",
        "oz",
        "body2imu_z",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing keys in {config_path}: {missing_keys}")

    return config


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"No valid data found in {csv_path}")

    return fieldnames, rows


def calc_leg_foot_pos_imu_z(row, leg: str, config, leftFoot, frontFoot):
    try:
        q1_raw = float(row[f"{leg}_HAA_position"])
        q2_raw = float(row[f"{leg}_HFE_position"])
        q3_raw = float(row[f"{leg}_KFE_position"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Failed to parse {leg} joint data from row: {row}") from exc

    offset_rad_haa = math.radians(config["offset_Angle_HAA"])
    offset_rad_hfe = math.radians(config["offset_Angle_HFE"])
    offset_rad_kfe = math.radians(config["offset_Angle_KFE"])

    l1 = config["L1"]
    l2 = config["L2"]
    l3 = config["L3"]
    oz = config["oz"]
    body2imu_z = config["body2imu_z"]

    q1 = q1_raw + offset_rad_haa
    q2 = q2_raw + offset_rad_hfe
    q3 = q3_raw + offset_rad_kfe

    c1 = math.cos(q1)
    c2 = math.cos(q2)
    c3 = math.cos(q3)
    s1 = math.sin(q1)
    s2 = math.sin(q2)
    s3 = math.sin(q3)
    c23 = c2 * c3 - s2 * s3

    # Matches calculateFootState_Body(): frontFoot only affects x, not z.
    _ = frontFoot
    foot_pos_body_z = leftFoot * l1 * s1 - l2 * c1 * c2 - l3 * c1 * c23 + oz
    return foot_pos_body_z - body2imu_z


def calc_gait_contact_state(foot_pos_imu_z):
    if not math.isfinite(foot_pos_imu_z):
        return 0
    return 1 if foot_pos_imu_z <= FOOT_POS_IMU_Z_THRESHOLD else 0


def build_labeled_rows(rows, config):
    labeled_rows = []

    for row in rows:
        labeled_row = dict(row)

        for leg, settings in LEG_SETTINGS.items():
            foot_pos_imu_z = calc_leg_foot_pos_imu_z(
                row,
                leg,
                config,
                leftFoot=settings["leftFoot"],
                frontFoot=settings["frontFoot"],
            )
            labeled_row[f"{leg}_foot_pos_imu_z"] = foot_pos_imu_z
            labeled_row[f"{leg}_gait_contact_states"] = calc_gait_contact_state(
                foot_pos_imu_z
            )

        labeled_rows.append(labeled_row)

    return labeled_rows


def save_labeled_csv(output_path: Path, fieldnames, labeled_rows):
    position_fields = [f"{leg}_foot_pos_imu_z" for leg in LEG_SETTINGS]
    contact_fields = [f"{leg}_gait_contact_states" for leg in LEG_SETTINGS]
    output_fields = list(fieldnames) + position_fields + contact_fields

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(labeled_rows)


def main():
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "log_data" / DATA_FILE
    csv_path = resolve_data_csv_path(save_dir)
    config_path = (script_dir / "../../config/LimxDynamic_WL_Config.yaml").resolve()
    output_dir = save_dir / "Gait_GRF"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "data_label_with_gait.csv"

    config = load_wl_config(config_path)
    fieldnames, rows = load_rows(csv_path)
    labeled_rows = build_labeled_rows(rows, config)
    save_labeled_csv(output_csv_path, fieldnames, labeled_rows)

    print(f"Input CSV: {csv_path}")
    print(f"Saved gait labeled CSV to: {output_csv_path}")
    print(f"foot_pos_imu_z contact threshold: {FOOT_POS_IMU_Z_THRESHOLD:g}")


if __name__ == "__main__":
    main()
