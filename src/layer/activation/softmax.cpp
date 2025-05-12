#include "softmax.h"

Eigen::MatrixXf hmnist::layer::activation::Softmax::activate(const Eigen::MatrixXf &X) {
    Eigen::MatrixXf Xc = X;
    // For each row, grab the largest number
    Eigen::VectorXf row_max = Xc.rowwise().maxCoeff();
    // Subtract each row's maximum from all elements in that corresponding row of 'Xc'
    Eigen::MatrixXf shifted = (Xc.array().colwise() - row_max.array()).matrix();
    // Get natural exponent of each coefficient
    Eigen::MatrixXf exps = shifted.array().exp().matrix();
    // Get the summation the exponent rows
    Eigen::VectorXf sums = exps.rowwise().sum();
    // Divide each element in 'exps' by the sum of its original row
    return (exps.array().colwise() / sums.array()).matrix();
}

Eigen::MatrixXf hmnist::layer::activation::Softmax::derivative(const Eigen::MatrixXf &X) {
    throw std::runtime_error("Not implemented, use CrossEntropyLoss.gradient() instead");
}