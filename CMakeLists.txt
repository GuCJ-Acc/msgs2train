cmake_minimum_required(VERSION 3.0.2)
project(msgs2train)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  serial
  std_msgs
  sensor_msgs
  message_generation
  pressure_tr_serial
)

find_package(yaml-cpp REQUIRED)


catkin_package(
  
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

include_directories(${YAML_CPP_INCLUDE_DIRS})


add_executable(msgs2train src/msgs2train.cpp)
target_link_libraries(msgs2train ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})