#include "relu.h"

Eigen::MatrixXf hmnist::layer::activation::Relu::activate(const Eigen::MatrixXf &X) {
    return X.cwiseMax(0);
}

Eigen::MatrixXf hmnist::layer::activation::Relu::derivative(const Eigen::MatrixXf &X) {
    return (X.array() > 0).cast<float>();
}
