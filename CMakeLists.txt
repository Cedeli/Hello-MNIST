cmake_minimum_required(VERSION 3.28.3)
project(HelloMNIST VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(./fastor)

add_executable(HelloMNIST
    src/main.cpp
)

target_include_directories(HelloMNIST PRIVATE ${CMAKE_SOURCE_DIR}/src)
