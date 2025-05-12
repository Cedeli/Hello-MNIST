[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Version](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](#requirements)
[![Build Status](https://github.com/cedeli/hello-mnist/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/cedeli/hello-mnist/actions)

# MNIST Handwritten Digit Classifier

A C++ project demonstrating handwritten digit classification using the MNIST dataset. This implementation leverages the powerful [Eigen](http://eigen.tuxfamily.org/) C++ library for efficient linear algebra operations.

## Dependencies
*   A C++23-capable compiler
*   [Eigen Linear Algebra Library](http://eigen.tuxfamily.org/)
*   [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) (download non `.idx` files)

## Building
```bash
# Clone the repository
git clone https://github.com/Cedeli/Hello-MNIST.git
cd Hello-MNIST/

# Initialize eigen submodule
git submodule init
git submodule update

# Build the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
