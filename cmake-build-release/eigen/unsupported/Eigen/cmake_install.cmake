# Install script for directory: D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/Programming/repos/Hello-MNIST/cmake-build-release")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/Program Files/JetBrains/CLion 2024.2.1/bin/mingw/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/AdolcForward"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/AlignedVector3"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/ArpackSupport"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/AutoDiff"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/BVH"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/EulerAngles"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/FFT"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/IterativeSolvers"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/KroneckerProduct"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/LevenbergMarquardt"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/MatrixFunctions"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/MPRealSupport"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/NNLS"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/NonLinearOptimization"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/NumericalDiff"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/OpenGLSupport"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/Polynomials"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/SparseExtra"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/SpecialFunctions"
    "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "D:/Programming/repos/Hello-MNIST/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/Programming/repos/Hello-MNIST/cmake-build-release/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

