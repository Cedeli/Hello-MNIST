cmake_minimum_required(VERSION 3.28.3)
project(Hello-MNIST VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE INTERNAL "")

set(EXECUTABLE_NAME ${PROJECT_NAME})

add_executable(${EXECUTABLE_NAME})

target_sources(${EXECUTABLE_NAME}
    PRIVATE
    src/main.cpp
    src/data/mnist_parser.cpp
    src/utils/file_reader.cpp
    src/utils/data_utils.cpp
    src/network/neural_network.cpp
)

add_subdirectory(eigen)

target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${EXECUTABLE_NAME} 
    PUBLIC
    Eigen3::Eigen
)

