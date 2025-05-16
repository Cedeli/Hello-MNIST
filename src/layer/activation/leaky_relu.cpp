#include "leaky_relu.h"

Eigen::MatrixXf hmnist::layer::activation::LeakyRelu::activate(const Eigen::MatrixXf &X) {
    return X.unaryExpr([](float v){ return v > 0 ? v : 0.01f * v; });
}

Eigen::MatrixXf hmnist::layer::activation::LeakyRelu::derivative(const Eigen::MatrixXf &X) {
    return X.unaryExpr([](float v){ return v > 0 ? 1.0f : 0.01f; });
}