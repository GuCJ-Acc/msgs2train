#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


TIME_INDEX = 0
LF_HAA_INDEX = 1
LF_HFE_INDEX = 2
LF_KFE_INDEX = 3
LOCAL_START_TIME = 15.0
LOCAL_END_TIME = 35.0


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
        "ox",
        "oy",
        "oz",
        "offset_Angle_HAA",
        "offset_Angle_HFE",
        "offset_Angle_KFE",
        "body2imu_x",
        "body2imu_y",
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

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                time_motor_state = float(row[TIME_INDEX])
                q1 = float(row[LF_HAA_INDEX])
                q2 = float(row[LF_HFE_INDEX])
                q3 = float(row[LF_KFE_INDEX])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_motor_state

            relative_times.append(time_motor_state - start_time)
            lf_joint_angles.append((q1, q2, q3))

    if not relative_times:
        raise ValueError(f"No valid data found in {csv_path}")

    return relative_times, lf_joint_angles


def calc_lf_foot_pos_imu_xyz(joint_angles, config):
    offset_rad_haa = math.radians(config["offset_Angle_HAA"])
    offset_rad_hfe = math.radians(config["offset_Angle_HFE"])
    offset_rad_kfe = math.radians(config["offset_Angle_KFE"])

    l1 = config["L1"]
    l2 = config["L2"]
    l3 = config["L3"]
    ox = config["ox"]
    oy = config["oy"]
    oz = config["oz"]
    body2imu_x = config["body2imu_x"]
    body2imu_y = config["body2imu_y"]
    body2imu_z = config["body2imu_z"]

    foot_pos_imu_xyz = []
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
        s23 = s2 * c3 + c2 * s3

        # LF is leg index 0, so leftFoot = 1 and frontFoot = 1.
        foot_pos_body_x = -l2 * s2 - l3 * s23 + ox
        foot_pos_body_y = l1 * c1 + l2 * c2 * s1 + l3 * s1 * c23 + oy
        foot_pos_body_z = l1 * s1 - l2 * c1 * c2 - l3 * c1 * c23 + oz

        foot_pos_imu_xyz.append(
            (
                foot_pos_body_x - body2imu_x,
                foot_pos_body_y - body2imu_y,
                foot_pos_body_z - body2imu_z,
            )
        )

    return foot_pos_imu_xyz


def set_equal_3d_axes(ax, xs, ys, zs):
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / "../../log/data.csv").resolve()
    config_path = (script_dir / "../../config/LimxDynamic_WL_Config.yaml").resolve()
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = fig_dir / "lf_foot_pos_imu_xyz_15_35.png"

    config = load_wl_config(config_path)
    relative_times, lf_joint_angles = load_lf_joint_angles(csv_path)
    lf_foot_pos_imu_xyz = calc_lf_foot_pos_imu_xyz(lf_joint_angles, config)

    local_xyz = []
    for time_value, foot_xyz in zip(relative_times, lf_foot_pos_imu_xyz):
        if LOCAL_START_TIME <= time_value <= LOCAL_END_TIME:
            local_xyz.append(foot_xyz)

    if not local_xyz:
        raise ValueError(
            f"No data found in the time range [{LOCAL_START_TIME}, {LOCAL_END_TIME}] s"
        )

    xs = [point[0] for point in local_xyz]
    ys = [point[1] for point in local_xyz]
    zs = [point[2] for point in local_xyz]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, linewidth=1.8, color="#1f77b4")
    ax.scatter(xs[0], ys[0], zs[0], color="#2ca02c", s=30, label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="#d62728", s=30, label="End")
    ax.set_xlabel("LF foot_pos_imu x (m)")
    ax.set_ylabel("LF foot_pos_imu y (m)")
    ax.set_zlabel("LF foot_pos_imu z (m)")
    ax.set_title("LF foot_pos_imu xyz trajectory (15s to 35s)")
    set_equal_3d_axes(ax, xs, ys, zs)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
