#include <ros/ros.h>
#include <serial/serial.h>
#include <std_msgs/String.h>

#include <iostream>
#include <string>
#include <mutex>
#include <sstream>
#include <vector>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "pressure_tr_serial/pressure_info.h"


class msgs2train
{
public:
    std::string Topic_MotorState;
    std::string Topic_PressureValues;

    std::mutex mtx_buff;

    int Value_RF;
    int Value_RH;
    int Value_LF;
    int Value_LH;

    ros::NodeHandle nh;
    ros::Subscriber sub_MotorState;
    ros::Subscriber sub_PressureValues;

    msgs2train()
    {
        YAML::Node sub_yaml;
        sub_yaml = YAML::LoadFile(std::string(std::string(ROOT_DIR)) + "config/pressureSerialPort.yaml");
        Topic_MotorState = sub_yaml["Topic_MotorState"].as<std::string>();
        Topic_PressureValues = sub_yaml["Topic_PressureValues"].as<std::string>();

    }

    ~msgs2train() {}
};


int main(int argc, char** argv)
{


    return 0;
}