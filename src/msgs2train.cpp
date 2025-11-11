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

    std::mutex mtx_buff;

    int Value_RF;
    int Value_RH;
    int Value_LF;
    int Value_LH;

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
        Topic_PressureValues = sub_yaml["Topic_PressureValues"].as<std::string>();

        sub_MotorState = nh.subscribe<sensor_msgs::JointState>(Topic_MotorState, 1000, &msgs2train::Topic_MotorState_CallBack, this, ros::TransportHints().tcpNoDelay());
        sub_PressureValues = nh.subscribe<pressure_tr_serial::pressure_info>(Topic_PressureValues, 1000, &msgs2train::Topic_PressureValues_CallBack, this, ros::TransportHints().tcpNoDelay());
    
    }

    ~msgs2train() {}

    void Topic_MotorState_CallBack(const sensor_msgs::JointState::ConstPtr& msg)
    {
        mtx_buff.lock();

        if (init_flag_MotorState == false){
            init_flag_MotorState = true;
        }

        buff_MotorState.push_back(msg);

        mtx_buff.unlock();
    }

    void Topic_PressureValues_CallBack(const pressure_tr_serial::pressure_info::ConstPtr& msg)
    {
        mtx_buff.lock();

        if (init_flag_PressureValue == false) {
            init_flag_PressureValue = true;
        }

        buff_PressureValue.push_back(msg);

        mtx_buff.unlock();
    }

    void sync_DataProcess()
    {
        if (buff_MotorState.empty() || buff_PressureValue.empty()) {
            return;
        }

        Value_RF = buff_PressureValue.front()->pressureValue[0];
        Value_RH = buff_PressureValue.front()->pressureValue[1];
        Value_LF = buff_PressureValue.front()->pressureValue[2];
        Value_LH = buff_PressureValue.front()->pressureValue[3];

        double time_PressureValue = buff_PressureValue.front()->header.stamp.toSec();
        double time_MotorState = buff_MotorState.front()->header.stamp.toSec();


        while ((!buff_MotorState.empty()) && (time_MotorState < time_PressureValue))
        {
            time_MotorState = buff_MotorState.front()->header.stamp.toSec();
            if (time_MotorState > time_PressureValue)
                break;

            // save MotorState data to file for MLP train


            buff_MotorState.pop_front();
        }

        buff_PressureValue.pop_front();

    }

};


int main(int argc, char** argv)
{


    return 0;
}