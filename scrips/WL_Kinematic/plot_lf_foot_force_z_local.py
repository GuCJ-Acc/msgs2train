#!/usr/bin/env python3

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_lf_foot_force_z import (
    calc_lf_foot_force_z,
    load_lf_joint_and_torque,
    load_wl_config,
)


LOCAL_START_TIME = 12.0
LOCAL_END_TIME = 20.0


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / "../../log/data.csv").resolve()
    config_path = (script_dir / "../../config/LimxDynamic_WL_Config.yaml").resolve()
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(exist_ok=True)
    output_path = fig_dir / "lf_foot_force_z_12_20.png"

    config = load_wl_config(config_path)
    relative_times, lf_joint_angles, lf_joint_torques = load_lf_joint_and_torque(csv_path)
    lf_foot_force_z = calc_lf_foot_force_z(lf_joint_angles, lf_joint_torques, config)

    local_times = []
    local_force_z = []
    for time_value, force_z in zip(relative_times, lf_foot_force_z):
        if LOCAL_START_TIME <= time_value <= LOCAL_END_TIME:
            local_times.append(time_value)
            local_force_z.append(force_z)

    if not local_times:
        raise ValueError(
            f"No data found in the time range [{LOCAL_START_TIME}, {LOCAL_END_TIME}] s"
        )

    plt.figure(figsize=(12, 6))
    plt.plot(local_times, local_force_z, linewidth=1.8, color="#2ca02c")
    plt.xlabel("time_MotorState (s, start at 0)")
    plt.ylabel("LF foot_force_z")
    plt.title("LF foot_force_z (12s to 20s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
