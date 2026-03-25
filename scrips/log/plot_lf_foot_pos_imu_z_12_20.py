#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0


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


def load_lf_joint_angles(csv_path: Path):
    relative_times = []
    lf_joint_angles = []
    start_time = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                q1 = float(row["LF_HAA_position"])
                q2 = float(row["LF_HFE_position"])
                q3 = float(row["LF_KFE_position"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            lf_joint_angles.append((q1, q2, q3))

    if not relative_times:
        raise ValueError(f"No valid LF joint data found in {csv_path}")

    return relative_times, lf_joint_angles


def calc_lf_foot_pos_imu_z(joint_angles, config):
    offset_rad_haa = math.radians(config["offset_Angle_HAA"])
    offset_rad_hfe = math.radians(config["offset_Angle_HFE"])
    offset_rad_kfe = math.radians(config["offset_Angle_KFE"])

    l1 = config["L1"]
    l2 = config["L2"]
    l3 = config["L3"]
    oz = config["oz"]
    body2imu_z = config["body2imu_z"]

    foot_pos_imu_z = []
    for q1_raw, q2_raw, q3_raw in joint_angles:
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

        foot_pos_body_z = l1 * s1 - l2 * c1 * c2 - l3 * c1 * c23 + oz
        foot_pos_imu_z.append(foot_pos_body_z - body2imu_z)

    return foot_pos_imu_z


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "data.csv"
    config_path = (script_dir / "../../config/LimxDynamic_WL_Config.yaml").resolve()
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = fig_dir / "lf_foot_pos_imu_z_12_20.png"

    config = load_wl_config(config_path)
    relative_times, lf_joint_angles = load_lf_joint_angles(csv_path)
    lf_foot_pos_imu_z = calc_lf_foot_pos_imu_z(lf_joint_angles, config)

    local_times = []
    local_foot_pos_imu_z = []
    for time_value, foot_z in zip(relative_times, lf_foot_pos_imu_z):
        if LOCAL_START_TIME <= time_value <= LOCAL_END_TIME:
            local_times.append(time_value)
            local_foot_pos_imu_z.append(foot_z)

    if not local_times:
        raise ValueError(
            f"No data found in the time range [{LOCAL_START_TIME}, {LOCAL_END_TIME}] s"
        )

    plt.figure(figsize=(12, 6))
    plt.plot(local_times, local_foot_pos_imu_z, linewidth=1.8, color="#1f77b4")
    plt.xlabel("time (s, start at 0)")
    plt.ylabel("LF foot_pos_imu z (m)")
    plt.title("LF foot_pos_imu z (12s to 20s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
