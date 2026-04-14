#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = "data_09"
LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 30.0
FORCE_THRESHOLD = 113.0


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
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing keys in {config_path}: {missing_keys}")

    return config


def load_lf_joint_and_torque(csv_path: Path):
    relative_times = []
    lf_joint_angles = []
    lf_joint_torques = []
    start_time = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time"])
                q1 = float(row["LF_HAA_position"])
                q2 = float(row["LF_HFE_position"])
                q3 = float(row["LF_KFE_position"])
                tau1 = float(row["LF_HAA_effort"])
                tau2 = float(row["LF_HFE_effort"])
                tau3 = float(row["LF_KFE_effort"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_number} in {csv_path}: {row}"
                ) from exc

            if start_time is None:
                start_time = time_value

            relative_times.append(time_value - start_time)
            lf_joint_angles.append((q1, q2, q3))
            lf_joint_torques.append((tau1, tau2, tau3))

    if not relative_times:
        raise ValueError(f"No valid LF joint data found in {csv_path}")

    return relative_times, lf_joint_angles, lf_joint_torques


def calc_lf_foot_force_z(joint_angles, joint_torques, config):
    offset_rad_haa = math.radians(config["offset_Angle_HAA"])
    offset_rad_hfe = math.radians(config["offset_Angle_HFE"])
    offset_rad_kfe = math.radians(config["offset_Angle_KFE"])

    l1 = config["L1"]
    l2 = config["L2"]
    l3 = config["L3"]
    leftFoot = 1.0
    frontFoot = 1.0

    foot_force_z = []
    for (q1_raw, q2_raw, q3_raw), (tau1, tau2, tau3) in zip(
        joint_angles, joint_torques
    ):
        q1 = q1_raw + offset_rad_haa
        q2 = q2_raw + offset_rad_hfe
        q3 = q3_raw + offset_rad_kfe

        s1 = math.sin(q1)
        s2 = math.sin(q2)
        s3 = math.sin(q3)
        c1 = math.cos(q1)
        c2 = math.cos(q2)
        c3 = math.cos(q3)
        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        _ = frontFoot
        jacobian = np.array(
            [
                [0.0, -l3 * c23 - l2 * c2, -l3 * c23],
                [
                    l3 * c1 * c23 + l2 * c1 * c2 - leftFoot * l1 * s1,
                    -l3 * s1 * s23 - l2 * s1 * s2,
                    -l3 * s1 * s23,
                ],
                [
                    l3 * s1 * c23 + l2 * c2 * s1 + leftFoot * l1 * c1,
                    l3 * c1 * s23 + l2 * c1 * s2,
                    l3 * c1 * s23,
                ],
            ],
            dtype=float,
        )

        torque = np.array([tau1, tau2, tau3], dtype=float)

        try:
            force = np.linalg.solve(jacobian.T, torque)
            foot_force_z.append(-float(force[2]))
        except np.linalg.LinAlgError:
            foot_force_z.append(float("nan"))

    return foot_force_z


def save_lf_force_csv(output_path: Path, relative_times, foot_force_z):
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "LF_foot_force_z"])
        for time_value, force_z in zip(relative_times, foot_force_z):
            writer.writerow([time_value, force_z])


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
    csv_path = resolve_data_csv_path(save_dir)
    config_path = (script_dir / "../../config/LimxDynamic_WL_Config.yaml").resolve()
    force_dir = save_dir / "Force"
    output_csv_path = force_dir / "data_lf_foot_force_z.csv"
    fig_dir = force_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "lf_foot_force_12_30_threshold_113.png"

    config = load_wl_config(config_path)
    relative_times, lf_joint_angles, lf_joint_torques = load_lf_joint_and_torque(csv_path)
    lf_foot_force_z = calc_lf_foot_force_z(lf_joint_angles, lf_joint_torques, config)
    save_lf_force_csv(output_csv_path, relative_times, lf_foot_force_z)

    local_times, local_forces = filter_time_range(
        relative_times,
        lf_foot_force_z,
        LOCAL_START_TIME,
        LOCAL_END_TIME,
    )

    local_times_np = np.array(local_times, dtype=float)
    local_forces_np = np.array(local_forces, dtype=float)
    below_threshold = np.isfinite(local_forces_np) & (local_forces_np < FORCE_THRESHOLD)

    plt.figure(figsize=(12, 6))
    plt.plot(
        local_times_np,
        local_forces_np,
        linewidth=1.8,
        color="#1f77b4",
        label="LF_foot_force_z",
    )
    plt.axhline(
        FORCE_THRESHOLD,
        linewidth=1.5,
        color="#d62728",
        linestyle="--",
        label=f"threshold={FORCE_THRESHOLD:g}",
    )
    plt.fill_between(
        local_times_np,
        local_forces_np,
        FORCE_THRESHOLD,
        where=below_threshold,
        color="#ff9896",
        alpha=0.35,
        interpolate=True,
        label="LF_foot_force_z < 113",
    )
    plt.xlabel("time (s, start at 0)")
    plt.ylabel("LF foot_force_z")
    plt.title("LF foot_force_z with below-threshold highlight (12s to 30s)")
    plt.xlim(LOCAL_START_TIME, LOCAL_END_TIME)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved data to: {output_csv_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
