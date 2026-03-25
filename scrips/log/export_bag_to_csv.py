#!/usr/bin/env python3

from pathlib import Path
import csv
from dataclasses import dataclass

import rosbag


BAG_PATH = Path("/home/jackgu/Codex/contact_DataProcess/data/Limx-Wheel-Contact-09.bag")
OUTPUT_PATH = Path("/home/jackgu/ROS_WS/LimxDynamics/src/msgs2train/scrips/log/data.csv")

MOTOR_TOPIC = "/LimxMotorState"
IMU_TOPIC = "/LimxIMUState"
PRESSURE_TOPIC = "/totalValue"

EXPECTED_MOTOR_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LF_WHL",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "LH_WHL",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RF_WHL",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
    "RH_WHL",
]

PRESSURE_OUTPUT_ORDER = ["LF", "LH", "RF", "RH"]


@dataclass
class ImuSample:
    stamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass
class PressureSample:
    stamp: float
    values: dict


def build_header():
    header = ["time"]
    header.extend([f"{name}_position" for name in EXPECTED_MOTOR_NAMES])
    header.extend([f"{name}_velocity" for name in EXPECTED_MOTOR_NAMES])
    header.extend([f"{name}_effort" for name in EXPECTED_MOTOR_NAMES])
    header.extend(
        [
            "imu_Acc_x",
            "imu_Acc_y",
            "imu_Acc_z",
            "imu_Gyro_x",
            "imu_Gyro_y",
            "imu_Gyro_z",
            "LF",
            "LH",
            "RF",
            "RH",
        ]
    )
    return header


def build_pressure_map(pressure_msg):
    if pressure_msg.name and len(pressure_msg.name) == len(pressure_msg.pressureValue):
        return {
            name: pressure_msg.pressureValue[index]
            for index, name in enumerate(pressure_msg.name)
        }

    # Fallback to the order used in filter_pressureData.cpp.
    default_order = ["RF", "RH", "LF", "LH"]
    return {
        name: pressure_msg.pressureValue[index]
        for index, name in enumerate(default_order[: len(pressure_msg.pressureValue)])
    }


def validate_motor_names(msg):
    names = list(msg.name)
    if names and names != EXPECTED_MOTOR_NAMES:
        raise ValueError(
            "Unexpected /LimxMotorState joint order.\n"
            f"Expected: {EXPECTED_MOTOR_NAMES}\n"
            f"Actual:   {names}"
        )


def collect_reference_streams():
    imu_samples = []
    pressure_samples = []

    with rosbag.Bag(str(BAG_PATH), "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=[IMU_TOPIC, PRESSURE_TOPIC]):
            if topic == IMU_TOPIC:
                imu_samples.append(
                    ImuSample(
                        stamp=msg.header.stamp.to_sec(),
                        acc_x=msg.linear_acceleration.x,
                        acc_y=msg.linear_acceleration.y,
                        acc_z=msg.linear_acceleration.z,
                        gyro_x=msg.angular_velocity.x,
                        gyro_y=msg.angular_velocity.y,
                        gyro_z=msg.angular_velocity.z,
                    )
                )
            else:
                pressure_samples.append(
                    PressureSample(
                        stamp=msg.header.stamp.to_sec(),
                        values=build_pressure_map(msg),
                    )
                )

    if not imu_samples:
        raise ValueError(f"No messages found for topic {IMU_TOPIC}")
    if not pressure_samples:
        raise ValueError(f"No messages found for topic {PRESSURE_TOPIC}")

    return imu_samples, pressure_samples


def advance_to_latest_index(samples, current_index, target_stamp):
    # Forward-fill in chronological order: keep the latest sample whose
    # timestamp is not later than the current /LimxMotorState timestamp.
    while current_index + 1 < len(samples):
        next_sample = samples[current_index + 1]
        if next_sample.stamp <= target_stamp:
            current_index += 1
        else:
            break

    return current_index


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    imu_samples, pressure_samples = collect_reference_streams()

    imu_index = 0
    pressure_index = 0
    saved_rows = 0

    with rosbag.Bag(str(BAG_PATH), "r") as bag, OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_header())

        for _, msg, _ in bag.read_messages(topics=[MOTOR_TOPIC]):
            validate_motor_names(msg)

            motor_time = msg.header.stamp.to_sec()
            imu_index = advance_to_latest_index(imu_samples, imu_index, motor_time)
            # Forward-fill /totalValue in time order to match the latest value
            # already available at the current motor timestamp.
            pressure_index = advance_to_latest_index(
                pressure_samples, pressure_index, motor_time
            )

            matched_imu = imu_samples[imu_index]
            matched_pressure = pressure_samples[pressure_index]
            row = [f"{motor_time:.6f}"]
            row.extend(f"{value:.6f}" for value in msg.position[:16])
            row.extend(f"{value:.6f}" for value in msg.velocity[:16])
            row.extend(f"{value:.6f}" for value in msg.effort[:16])
            row.extend(
                [
                    f"{matched_imu.acc_x:.6f}",
                    f"{matched_imu.acc_y:.6f}",
                    f"{matched_imu.acc_z:.6f}",
                    f"{matched_imu.gyro_x:.6f}",
                    f"{matched_imu.gyro_y:.6f}",
                    f"{matched_imu.gyro_z:.6f}",
                ]
            )
            row.extend(
                str(matched_pressure.values.get(name, ""))
                for name in PRESSURE_OUTPUT_ORDER
            )
            writer.writerow(row)
            saved_rows += 1

    print(f"Saved {saved_rows} rows to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
