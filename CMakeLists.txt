cmake_minimum_required(VERSION 2.15)
project(ORB_SLAM2)

add_definitions(-w)


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

add_executable(main src/main.cc)
target_link_libraries(main ${OpenCV_LIBS} ${EIGEN3_LIBS} ${Pangolin_LIBRARIES})