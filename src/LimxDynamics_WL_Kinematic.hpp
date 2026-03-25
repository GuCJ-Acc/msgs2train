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
    double rW;      // 小腿长度

    double offset_deg_HAA;    // 髋关节偏移角度
    double offset_deg_HFE;    // 大腿关节偏移角度
    double offset_deg_KFE;    // 小腿关节偏移角度

    double offset_rad_HAA;    // 髋关节偏移弧度
    double offset_rad_HFE;    // 大腿关节偏移弧度
    double offset_rad_KFE;    // 小腿关节偏移弧度

    double ox, oy, oz;  // body系到关节模组的外参
    double body2imu_x, body2imu_y, body2imu_z;  // imu系到body系的外参
    double force_threshold;

public:
    std::string root_dir = ROOT_DIR;

    Eigen::Matrix<double, 3, NUM_LEG>  foot_angle;      // 3*4 关节电机角度数据矩阵    [LF, LH, RF, RH]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_angle_vel;  // 3*4 关节电机角速度数据矩阵  [LF, LH, RF, RH]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_torque;     // 3*4 关节电机力矩数据矩阵    [LF, LH, RF, RH]

    Eigen::Matrix<double, 3, NUM_LEG>  foot_force;      // 3*4 足底接触力估计数据矩阵  [LF, LH, RF, RH]
    Eigen::Matrix<double, 1, NUM_LEG>  foot_force_z;    // 1*4 足底力z轴数据矩阵 [LF, LH, RF, RH]
    Eigen::Matrix<double, 1, NUM_LEG>  foot_contact;    // 1*4 足底接触数据矩阵  [LF, LH, RF, RH]

    Eigen::Matrix<double, 3, NUM_LEG>  foot_pos_body;   // 3*4 足底位置（body系）数据矩阵  [LF, LH, RF, RH]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_vel_body;   // 3*4 足底速度（body系）数据矩阵  [LF, LH, RF, RH]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_pos_imu;    // 3*4 足底位置（imu系）数据矩阵  [LF, LH, RF, RH]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_vel_imu;    // 3*4 足底速度（imu系）数据矩阵  [LF, LH, RF, RH]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_pos_rel;    // 3*4 足底位置（world系）数据矩阵  [LF, LH, RF, RH]
    Eigen::Matrix<double, 3, NUM_LEG>  foot_vel_rel;    // 3*4 足底速度（world系）数据矩阵  [LF, LH, RF, RH]

    LimxDynamics_WL_Kinematic();
    ~LimxDynamics_WL_Kinematic();

    Eigen::Matrix3d calcJco(double q1, double q2, double q3);
    Eigen::Vector3d calcFoc(Eigen::Vector3d Angel, Eigen::Vector3d Torque);
    Eigen::Vector3d calcEe2H(Eigen::Vector3d Angel);
    Eigen::Vector3d calcVEe2H(Eigen::Vector3d Angel, Eigen::Vector3d Omega);

    void calculateFootState_Body();
    void get_LimxDynamic_WL_InfoFromMsg(const sensor_msgs::JointState::ConstPtr& msg);


    Eigen::Matrix3d skew(Eigen::Vector3d a);
    Eigen::Matrix3d Exp(const Eigen::Vector3d& r);
    Eigen::Vector3d Log(const Eigen::Matrix3d& R);
};


/***********************************************************
 * @brief Construct a new LimxDynamics_WL_Kinematic::LimxDynamics_WL_Kinematic object
 * 相关参数的赋值
 * ox: WL = 0.3148;
 * oy: WL = 0.0650;
 * L1: WL = 0.1614;
 * L2: WL = 0.280;
 * L3: WL = 0.280;
 * rW: WL = 0.128;
 **********************************************************/
LimxDynamics_WL_Kinematic::LimxDynamics_WL_Kinematic()
{
    YAML::Node yaml;
    yaml = YAML::LoadFile(std::string(std::string(ROOT_DIR)) + "config/LimxDynamic_WL_Config.yaml");

    L1 = yaml["LimxDynamic_WL"]["L1"].as<double>();       // 腿平面与髋侧摆距离
    L2 = yaml["LimxDynamic_WL"]["L2"].as<double>();       // 大腿长度
    L3 = yaml["LimxDynamic_WL"]["L3"].as<double>();       // 小腿长度
    rW = yaml["LimxDynamic_WL"]["rW"].as<double>();       // 轮胎半径

    // 关节偏移角度
    offset_deg_HAA = yaml["LimxDynamic_WL"]["offset_Angle_HAA"].as<double>();
    offset_deg_HFE = yaml["LimxDynamic_WL"]["offset_Angle_HFE"].as<double>();
    offset_deg_KFE = yaml["LimxDynamic_WL"]["offset_Angle_KFE"].as<double>();
    offset_rad_HAA = offset_deg_HAA * M_PI / 180.0;
    offset_rad_HFE = offset_deg_HFE * M_PI / 180.0;
    offset_rad_KFE = offset_deg_KFE * M_PI / 180.0;

    ox = yaml["LimxDynamic_WL"]["ox"].as<double>();
    oy = yaml["LimxDynamic_WL"]["oy"].as<double>();
    oz = yaml["LimxDynamic_WL"]["oz"].as<double>();

    body2imu_x = yaml["LimxDynamic_WL"]["body2imu_x"].as<double>();
    body2imu_y = yaml["LimxDynamic_WL"]["body2imu_y"].as<double>();
    body2imu_z = yaml["LimxDynamic_WL"]["body2imu_z"].as<double>();

    force_threshold = yaml["LimxDynamic_WL"]["force_threshold"].as<double>();  // 足端接触力的阈值

}

LimxDynamics_WL_Kinematic::~LimxDynamics_WL_Kinematic() {}





/***********************************************************
 * @brief 计算LimxDynamics_WL的单腿雅可比矩阵
 * 
 * @param q1 髋关节角度
 * @param q2 大腿关节角度
 * @param q3 小腿关节角度
 * @return Eigen::Matrix3d 返回计算得到的雅可比矩阵
 **********************************************************/
Eigen::Matrix3d LimxDynamics_WL_Kinematic::calcJco(double q1, double q2, double q3)
{
    Eigen::Matrix3d Jaco;

    double s1 = std::sin(q1);
    double s2 = std::sin(q2);
    double s3 = std::sin(q3);

    double c1 = std::cos(q1);
    double c2 = std::cos(q2);
    double c3 = std::cos(q3);

    double c23 = c2 * c3 - s2 * s3;
    double s23 = s2 * c3 + c2 * s3;

    Jaco(0, 0) = 0;
    Jaco(1, 0) = -L3 * c1 * c23 - L2 * c1 * c2 - L1 * s1;
    Jaco(2, 0) = -L3 * s1 * c23 - L2 * c2 * s1 + L1 * c1;
    Jaco(0, 1) = L3 * c23 + L2 * c2;
    Jaco(1, 1) = L3 * s1 * s23 + L2 * s1 * s2;
    Jaco(2, 1) = -L3 * c1 * s23 - L2 * c1 * s2;
    Jaco(0, 2) = L3 * c23;
    Jaco(1, 2) = L3 * s1 * s23;
    Jaco(2, 2) = -L3 * c1 * s23;

    return Jaco;
}


/***********************************************************
 * @brief 计算LimxDynamics_WL的单腿静力学的足端的力输出
 * 
 * @param Angel 关节角度
 * @param Torque 关节力矩
 * @return Eigen::Vector3d 返回足端的力
 **********************************************************/
Eigen::Vector3d LimxDynamics_WL_Kinematic::calcFoc(Eigen::Vector3d Angel, Eigen::Vector3d Torque)
{
    Eigen::Matrix3d Jaco;
    Eigen::Matrix3d Jaco_T;
    Eigen::Matrix3d Jaco_Tinv;

    Eigen::Vector3d force;

    Jaco = LimxDynamics_WL_Kinematic::calcJco(Angel(0), Angel(1), Angel(2));
    Jaco_T = Jaco.transpose();
    Jaco_Tinv = Jaco_T.inverse();

    // Jaco_Tinv = Jaco.transpose().inverse();

    force = Jaco_Tinv * Torque;

    return force;
}


/***********************************************************
 * @brief 计算X20的单腿正运动学的足端位置（髋关节坐标系下）
 * 
 * @param Angel 关节电机角度
 * @return Eigen::Vector3d 返回足端位置
 **********************************************************/
Eigen::Vector3d LimxDynamics_WL_Kinematic::calcEe2H(Eigen::Vector3d Angel)
{
    Eigen::Vector3d pEe2H;

    double s1 = std::sin(Angel(0));
    double s2 = std::sin(Angel(1));
    double s3 = std::sin(Angel(2));

    double c1 = std::cos(Angel(0));
    double c2 = std::cos(Angel(1));
    double c3 = std::cos(Angel(2));

    double c23 = c2 * c3 - s2 * s3;
    double s23 = s2 * c3 + c2 * s3;

    pEe2H(0) = L3 * s23 + L2 * s2;
    pEe2H(1) = -L3 * s1 * c23 + L1 * c1 - L2 * c2 * s1;
    pEe2H(2) =  L3 * c1 * c23 + L1 * s1 + L2 * c1 * c2;

    return pEe2H;
}


/***********************************************************
 * @brief 计算LimxDynamics_WL的单腿正运动学的足端速度（髋关节坐标系下）
 * 
 * @param Angel 关节电机角度
 * @param Omega 关节电机角速度
 * @return Eigen::Vector3d 
 **********************************************************/
Eigen::Vector3d LimxDynamics_WL_Kinematic::calcVEe2H(Eigen::Vector3d Angel, Eigen::Vector3d Omega)
{
    Eigen::Matrix3d Jaco;

    Jaco = LimxDynamics_WL_Kinematic::calcJco(Angel(0), Angel(1), Angel(2));
    return Jaco * Omega;
}


/***********************************************************
 * @brief 计算LimxDynamics_WL全身运动的函数
 * 
 * 计算包括：
 * LimxDynamics_WL的足端位置（body系）、LimxDynamics_WL的足端速度（髋关节系）、足端接触状态、足端力
 **********************************************************/
void LimxDynamics_WL_Kinematic::calculateFootState_Body()
{
    for (int i = 0; i < 4; ++i)
    {
        // 判断是左边还是右边，前面还是后面，用于修正正运动学计算
        // 左边腿：L1/oy > 0; 右边腿：L1/oy < 0, ;
        // 前面腿：ox > 0; 后面腿：ox < 0;
        int leftFoot = -1, frontFoot = -1;
        if (i < 2) leftFoot = 1;
        if (i == 0 || i == 2) frontFoot = 1;
        
        // 计算 sin 和 cos 的值
        double s1 = std::sin(foot_angle(0, i));
        double s2 = std::sin(foot_angle(1, i));
        double s3 = std::sin(foot_angle(2, i));

        double c1 = cos(foot_angle(0, i));
        double c2 = cos(foot_angle(1, i));
        double c3 = cos(foot_angle(2, i));

        double c23 = c2 * c3 - s2 * s3;
        double s23 = s2 * c3 + c2 * s3;

        // double s23 = sin(foot_angle(1, i) + foot_angle(2, i));  
        // double c23 = cos(foot_angle(1, i) + foot_angle(2, i));

        // 计算足端在body系下的坐标位置
        foot_pos_body(0, i) = (- L2 * s2) + (- L3 * s23) + frontFoot * ox;
        foot_pos_body(1, i) = (leftFoot * L1 * c1) + (L2 * c2 * s1) + (L3 * s1 * c23) + leftFoot * oy;
        foot_pos_body(2, i) = (leftFoot * L1 * s1) + (- L2 * c1 * c2) + (- L3 * c1 * c23) + oz;

        // 计算足端从body系转换到imu系下的坐标位置
        foot_pos_imu(0, i) = foot_pos_body(0, i) - body2imu_x;
        foot_pos_imu(1, i) = foot_pos_body(1, i) - body2imu_y;
        foot_pos_imu(2, i) = foot_pos_body(2, i) - body2imu_z;

        // 计算雅可比矩阵
        Eigen::Matrix3d J;
        J(0, 0) = 0;
        J(1, 0) = L3 * c1 * c23 + L2 * c1 * c2 - leftFoot * L1 * s1;
        J(2, 0) = L3 * s1 * c23 + L2 * c2 * s1 + leftFoot * L1 * c1;
        J(0, 1) = - L3 * c23 - L2 * c2;
        J(1, 1) = - L3 * s1 * s23 - L2 * s1 * s2;
        J(2, 1) = L3 * c1 * s23 + L2 * c1 * s2;
        J(0, 2) = - L3 * c23;
        J(1, 2) = - L3 * s1 * s23;
        J(2, 2) = L3 * c1 * s23;

        // 计算足端的运动速度 v = J * ω;
        foot_vel_body(0, i) =                                  J(0, 1) * foot_angle_vel(1, i) + J(0, 2) * foot_angle_vel(2, i);
        foot_vel_body(1, i) = J(1, 0) * foot_angle_vel(0, i) + J(1, 1) * foot_angle_vel(1, i) + J(1, 2) * foot_angle_vel(2, i);
        foot_vel_body(2, i) = J(2, 0) * foot_angle_vel(0, i) + J(2, 1) * foot_angle_vel(1, i) + J(2, 2) * foot_angle_vel(2, i);

        // 计算足端的力估计 F = J^_T * t;
        Eigen::Matrix3d J_T;
        Eigen::Matrix3d J_Tinv;
        Eigen::Vector3d force;
        Eigen::Vector3d Torque;

        J_T = J.transpose();
        J_Tinv = J_T.inverse();
        Torque(0) = foot_torque(0, i);
        Torque(1) = foot_torque(1, i);
        Torque(2) = foot_torque(2, i);
        force = J_Tinv * Torque;

        // 将足端估计的力存进相关数据缓存内
        foot_force(0, i) = - force(0);
        foot_force(1, i) = - force(1);
        foot_force(2, i) = - force(2);
        foot_force_z(i) = - force(2);
        // foot_force_z(i) = std::sqrt(force(0) * force(0) + force(1) * force(1) + force(2) * force(2));

        // force_threshold = 110;
        if (foot_force_z(i) > force_threshold)
        { foot_contact(i) = 1; }
        else
        { foot_contact(i) = 0; }

    }
}


/***********************************************************
 * @brief 接收LimxDynamics_WL关节数据话题的回调函数
 * 
 * @param msg 关节数据话题
 * 
 * 接收话题数据，进行相关数据的缓存填充并进行计算相关运动学数据
 **********************************************************/
void LimxDynamics_WL_Kinematic::get_LimxDynamic_WL_InfoFromMsg(const sensor_msgs::JointState::ConstPtr& msg)
{
    sensor_msgs::JointState curMsg;
    curMsg = *msg;

    /*************************************
     * @brief 矩阵结构
     *          | LF_HAA,  LH_HAA,  RF_HAA,  RH_HAA |
     * Matrix = | LF_HFE,  LH_HFE,  RF_HFE,  RH_HFE |
     *          | LF_KFE,  LH_KFE,  RF_KFE,  RH_KFE |
     *          | LF_WHL,  LH_WHL,  RF_WHL,  RH_WHL |
     *************************************/
    Eigen::Matrix<double, 3, 4> Joint_Angel;
    Eigen::Matrix<double, 3, 4> Joint_Angel_Vel;
    Eigen::Matrix<double, 3, 4> Joint_Torque;
    for (int i = 0; i < 4; i++)
    {
        Joint_Angel(0, i) = curMsg.position[i * 4]     + offset_rad_HAA;
        Joint_Angel(1, i) = curMsg.position[i * 4 + 1] + offset_rad_HFE;
        Joint_Angel(2, i) = curMsg.position[i * 4 + 2] + offset_rad_KFE;

        Joint_Angel_Vel(0, i) = curMsg.velocity[i * 4];
        Joint_Angel_Vel(1, i) = curMsg.velocity[i * 4 + 1];
        Joint_Angel_Vel(2, i) = curMsg.velocity[i * 4 + 2];

        Joint_Torque(0, i) = curMsg.effort[i * 4];
        Joint_Torque(1, i) = curMsg.effort[i * 4 + 1];
        Joint_Torque(2, i) = curMsg.effort[i * 4 + 2];
    }

    foot_angle = Joint_Angel;
    foot_angle_vel = Joint_Angel_Vel;
    foot_torque = Joint_Torque;

    calculateFootState_Body();  // 计算相关的运动学参数
}









/***********************************************************
 * @brief 求向量a的反对称矩阵
 * 
 * @param a 输入的向量a
 * @return Eigen::Matrix3d 求得的反对称矩阵
 **********************************************************/
Eigen::Matrix3d LimxDynamics_WL_Kinematic::skew(Eigen::Vector3d a)
{
    Eigen::Matrix3d skewresult;
    skewresult << 0, -a[2], a[1],
                  a[2], 0, -a[0],
                 -a[1], a[0], 0;
    return skewresult;
}


/***********************************************************
 * @brief 旋转向量转换为旋转矩阵
 * 
 * @param r 旋转向量
 * @return Eigen::Matrix3d 旋转矩阵
 * 
 * 使用罗德里格斯公式将旋转向量转换为旋转矩阵
 **********************************************************/
Eigen::Matrix3d LimxDynamics_WL_Kinematic::Exp(const Eigen::Vector3d& r)
{
    Eigen::Matrix3d expr;
    double theta = r.norm();
    if(theta < 1e-12){
        expr = Eigen::Matrix3d::Identity();
    }
    else{
        Eigen::Matrix3d skew_ = skew(r / theta);
        expr = Eigen::Matrix3d::Identity() + sin(theta) * skew_ + (1 - cos(theta)) * skew_ * skew_;
    }
    return expr;
}


/***********************************************************
 * @brief 旋转矩阵转换为旋转向量
 * 
 * @param R 旋转矩阵
 * @return Eigen::Vector3d 旋转向量
 **********************************************************/
Eigen::Vector3d LimxDynamics_WL_Kinematic::Log(const Eigen::Matrix3d& R)
{
    double theta = (R.trace() > 3 - 1e-6) ? 0 : acos((R.trace() - 1) / 2);
    //double theta =  acos((R.trace() - 1) / 2);

    Eigen::Vector3d r(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    return fabs(theta) < 0.001 ? (0.5 * r) : (0.5 * theta / sin(theta) * r);
    //return 0.5 * theta / sin(theta) * r;
}