cmake_minimum_required(VERSION 3.29.6)
project(Hello-MNIST VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Eigen setup
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(eigen)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Flags used by the C++ compiler during debug builds.")

set(DATA_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data")

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE INTERNAL "")

file(GLOB_RECURSE ALL_PROJECT_SRC_FILES "src/*.cpp")

set(MAIN_APPLICATION_SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

set(HMNIST_LIB_SRC_FILES "")
foreach(src_file ${ALL_PROJECT_SRC_FILES})
    if(NOT ("${src_file}" STREQUAL "${MAIN_APPLICATION_SRC_FILE}"))
        list(APPEND HMNIST_LIB_SRC_FILES ${src_file})
    endif()
endforeach()

# Sanity checks
if(NOT HMNIST_LIB_SRC_FILES)
    message(WARNING "No library source files were identified.")
endif()
if(NOT EXISTS ${MAIN_APPLICATION_SRC_FILE})
    message(FATAL_ERROR "Main application source file '${MAIN_APPLICATION_SRC_FILE}' not found.")
endif()

add_library(hmnist_lib STATIC ${HMNIST_LIB_SRC_FILES})
target_include_directories(hmnist_lib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(hmnist_lib
        PUBLIC
        Eigen3::Eigen
)

target_compile_definitions(hmnist_lib PRIVATE DATA_DIRECTORY_PATH="${DATA_DIR_PATH}")

set(EXECUTABLE_NAME ${PROJECT_NAME})
add_executable(${EXECUTABLE_NAME} ${MAIN_APPLICATION_SRC_FILE}) # Built ONLY from main.cpp
target_include_directories(${EXECUTABLE_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(${EXECUTABLE_NAME}
        PRIVATE
        hmnist_lib
)

# Google Test Setup
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Test Discovery and Setup
file(GLOB TEST_CPP_FILES "tests/*.cpp")

foreach(test_src_file ${TEST_CPP_FILES})
    get_filename_component(test_executable_name ${test_src_file} NAME_WE)

    add_executable(${test_executable_name} ${test_src_file})
    target_include_directories(${test_executable_name}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${test_executable_name}
            PRIVATE
            hmnist_lib
            GTest::gtest_main
    )
    gtest_discover_tests(${test_executable_name})
endforeach()