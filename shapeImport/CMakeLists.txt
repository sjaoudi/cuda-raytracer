cmake_minimum_required(VERSION 2.8.11)
project(shpimport)

set(CMAKE_CXX_FLAGS "-g -Wall")

#required CMake options for QT Apps
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(demo demo.cpp)
target_link_libraries(demo -lassimp)
