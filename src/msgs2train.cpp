#include <ros/ros.h>
#include <serial/serial.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>

#include <iostream>
#include <string>
#include <deque>
#include <mutex>
#include <sstream>
#include <vector>
#include <thread>
#include <yaml-cpp/yaml.h>
#include <fstream>

#include "pressure_tr_serial/pressure_info.h"


class msgs2train
{
public:
    std::string Topic_MotorState;
    std::string Topic_IMUState;
    std::string Topic_PressureValues;
    std::string saveData_FilePath;

    std::ofstream data2File;

    std::mutex mtx_buff;

    int Value_RF;
    int Value_RH;
    int Value_LF;
    int Value_LH;

    double imu_Acc_x, imu_Acc_y, imu_Acc_z;
    double imu_Gyro_x, imu_Gyro_y, imu_Gyro_z;
    double imu_Quat_x, imu_Quat_y, imu_Quat_z, imu_Quat_w;

    double motor_q[16];     // LF/LH/RF/RH;  HAA/HFE/KFE/WHL;
    double motor_dq[16];
    double motor_tau[16];

    ros::NodeHandle nh;
    ros::Subscriber sub_MotorState;
    ros::Subscriber sub_IMUState;
    ros::Subscriber sub_PressureValues;

    bool init_flag_MotorState = false;
    bool init_flag_PressureValue = false;

    std::deque<sensor_msgs::JointState::ConstPtr> buff_MotorState;
    std::deque<sensor_msgs::Imu::ConstPtr> buff_IMUState;
    std::deque<pressure_tr_serial::pressure_info::ConstPtr> buff_PressureValue;
    std::deque<double> buff_time_PressureValue;
    

    msgs2train()
    {
        YAML::Node sub_yaml;
        sub_yaml = YAML::LoadFile(std::string(std::string(ROOT_DIR)) + "config/msgs2train.yaml");
        Topic_MotorState = sub_yaml["Topic_MotorState"].as<std::string>();
        Topic_IMUState = sub_yaml["LimxIMUState"].as<std::string>();
        Topic_PressureValues = sub_yaml["Topic_PressureValues"].as<std::string>();
        saveData_FilePath = sub_yaml["saveData_FilePath"].as<std::string>();

        saveData_FilePath = std::string(std::string(ROOT_DIR)) + saveData_FilePath;
        int reflash_File = system((std::string("exec rm ") + saveData_FilePath).c_str());

        sub_MotorState = nh.subscribe<sensor_msgs::JointState>(Topic_MotorState, 1000, &msgs2train::Topic_MotorState_CallBack, this, ros::TransportHints().tcpNoDelay());
        sub_IMUState = nh.subscribe<sensor_msgs::Imu>(Topic_IMUState, 1000, &msgs2train::Topic_IMUState_CallBack, this, ros::TransportHints().tcpNoDelay());
        sub_PressureValues = nh.subscribe<pressure_tr_serial::pressure_info>(Topic_PressureValues, 1000, &msgs2train::Topic_PressureValues_CallBack, this, ros::TransportHints().tcpNoDelay());
    
    }

    ~msgs2train() {}

    void Topic_MotorState_CallBack(const sensor_msgs::JointState::ConstPtr& msg)
    {
        mtx_buff.lock();

        buff_MotorState.push_back(msg);

        mtx_buff.unlock();
    }

    void Topic_IMUState_CallBack(const sensor_msgs::Imu::ConstPtr& msg)
    {
        mtx_buff.lock();

        buff_IMUState.push_back(msg);

        mtx_buff.unlock();
    }

    void Topic_PressureValues_CallBack(const pressure_tr_serial::pressure_info::ConstPtr& msg)
    {
        mtx_buff.lock();

        buff_PressureValue.push_back(msg);

        mtx_buff.unlock();
    }


    /*****************************************************************
     * @brief 数据处理的时间同步函数
     * 
     ****************************************************************/
    void sync_DataProcess()
    {
        if (buff_MotorState.empty() || buff_PressureValue.empty() || buff_IMUState.empty()) {
            return;
        }

        Value_RF = buff_PressureValue.front()->pressureValue[0];
        Value_RH = buff_PressureValue.front()->pressureValue[1];
        Value_LF = buff_PressureValue.front()->pressureValue[2];
        Value_LH = buff_PressureValue.front()->pressureValue[3];

        imu_Acc_x = buff_IMUState.front()->linear_acceleration.x;
        imu_Acc_y = buff_IMUState.front()->linear_acceleration.y;
        imu_Acc_z = buff_IMUState.front()->linear_acceleration.z;
        imu_Gyro_x = buff_IMUState.front()->angular_velocity.x;
        imu_Gyro_y = buff_IMUState.front()->angular_velocity.y;
        imu_Gyro_z = buff_IMUState.front()->angular_velocity.z;
        imu_Quat_w = buff_IMUState.front()->orientation.w;
        imu_Quat_x = buff_IMUState.front()->orientation.x;
        imu_Quat_y = buff_IMUState.front()->orientation.y;
        imu_Quat_z = buff_IMUState.front()->orientation.z;


        double time_PressureValue = buff_PressureValue.front()->header.stamp.toSec();
        double time_MotorState = buff_MotorState.front()->header.stamp.toSec();
        double time_IMUState = buff_IMUState.front()->header.stamp.toSec();


        while ((!buff_MotorState.empty()) && (time_MotorState < time_PressureValue) && (time_MotorState < time_IMUState))
        {
            time_MotorState = buff_MotorState.front()->header.stamp.toSec();
            if (time_MotorState > time_PressureValue) {
                break;
            }

            if (time_MotorState > time_IMUState) {
                break;
            }

            // save MotorState data
            for (int i = 0; i < 16; i++)
            {
                motor_q[i] = buff_MotorState.front()->position[i];
                motor_dq[i] = buff_MotorState.front()->velocity[i];
                motor_tau[i] = buff_MotorState.front()->effort[i];
            }

            // save data to file
            // Time,motor_q[i],motor_dq[i],motor_tau[i],imu_Acc[i],imu_Gyro[i],imu_Quat[i],Value_LF,Value_LH,Value_RF,Value_RH
            data2File.open(saveData_FilePath, std::ios::app);
            if (data2File.is_open())
            {
                // Time
                data2File << std::fixed << time_MotorState << ",";

                // motor
                for (int i = 0; i < 16; i++) {
                    data2File << motor_q[i] << ",";
                }
                for (int i = 0; i < 16; i++) {
                    data2File << motor_dq[i] << ",";
                }
                for (int i = 0; i < 16; i++) {
                    data2File << motor_tau[i] << ",";
                }

                // imu
                data2File << imu_Acc_x << "," << imu_Acc_y << "," << imu_Acc_z << ",";
                data2File << imu_Gyro_x << "," << imu_Gyro_y << "," << imu_Gyro_z << ",";
                data2File << imu_Quat_w << "," << imu_Quat_x << "," << imu_Quat_y << "," << imu_Quat_z << ",";

                // pressure_value
                data2File << Value_LF << "," << Value_LH << "," << Value_RF << "," << Value_RH << std::endl;

                data2File.close();
            }

            buff_MotorState.pop_front();
        }

        if (time_MotorState > time_PressureValue) {
            buff_PressureValue.pop_front();
        }

        if (time_MotorState > time_IMUState) {
            buff_IMUState.pop_front();
        }

    }

    void loop_saveData2File()
    {
        ROS_INFO("\033[1;32m----> Enter Loop Save Msgs Data to .csv File.\033[0m");

        while (ros::ok())
        {
            sync_DataProcess();
        }
    }

};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "msgs2train");

    ROS_INFO("\033[1;32m----> Save Msgs Data to .csv File Process is Started.\033[0m");

    msgs2train MO;

    std::thread ProceeThread(&msgs2train::loop_saveData2File, &MO);

    ros::spin();

    ProceeThread.join();

    return 0;
}