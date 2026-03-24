#include <cmath>
#include <math.h>
#include <deque>
#include <thread>
#include <fstream>
#include <Eigen/Eigen>
#include <Eigen/Core>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

#include <yaml-cpp/yaml.h>

#define NUM_LEG 4

class LimxDynamics_WL_Kinematic
{
private:
    double L1;      // 腿平面与髋侧摆距离
    double L2;      // 大腿长度
    double L3;      // 小腿长度

    double Whip;    // 髋左右间距
    double Lhip;    // 髋前后间距

    double ox, oy, oz;  // body系到关节模组的外参
    double body2imu_x, body2imu_y, body2imu_z;  // imu系到body系的外参
    double force_threshold;

public:
    std::string root_dir = ROOT_DIR;

    Eigen::Matrix<double, 3, NUM_LEG>  foot_angle;      // 3*4 关节电机角度数据矩阵    [LF, RF, LB, RB]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_angle_vel;  // 3*4 关节电机角速度数据矩阵  [LF, RF, LB, RB]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_torque;     // 3*4 关节电机力矩数据矩阵    [LF, RF, LB, RB]

    Eigen::Matrix<double, 3, NUM_LEG>  foot_force;      // 3*4 足底接触力估计数据矩阵  [LF, RF, LB, RB]
    Eigen::Matrix<double, 1, NUM_LEG>  foot_force_z;    // 1*4 足底力z轴数据矩阵 [LF, RF, LB, RB]
    Eigen::Matrix<double, 1, NUM_LEG>  foot_contact;    // 1*4 足底接触数据矩阵  [LF, RF, LB, RB]

    Eigen::Matrix<double, 3, NUM_LEG>  foot_pos_body;   // 3*4 足底位置（body系）数据矩阵  [LF, RF, LB, RB]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_vel_body;   // 3*4 足底速度（body系）数据矩阵  [LF, RF, LB, RB]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_pos_imu;    // 3*4 足底速度（imu系）数据矩阵  [LF, RF, LB, RB]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_vel_imu;    // 3*4 足底速度（imu系）数据矩阵  [LF, RF, LB, RB]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_pos_rel;    // 3*4 足底位置（world系）数据矩阵  [LF, RF, LB, RB]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_vel_rel;    // 3*4 足底速度（world系）数据矩阵  [LF, RF, LB, RB]

};