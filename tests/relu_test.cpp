#include <gtest/gtest.h>
#include <Eigen/Core>
#include "layer/activation/relu.h"

TEST(ReluTest, ActivatePositiveInput) {
    hmnist::layer::activation::Relu relu;
    Eigen::MatrixXf X(2, 2);
    X << 1.0f, 2.0f,
            3.0f, 4.0f;
    Eigen::MatrixXf expected = X;
    EXPECT_TRUE(relu.activate(X).isApprox(expected));
}

TEST(ReluTest, DerivativePositiveInput) {
    hmnist::layer::activation::Relu relu;
    Eigen::MatrixXf X(2, 2);
    X << 1.0f, 0.5f,
            3.0f, 0.01f;
    Eigen::MatrixXf expected(2, 2);
    expected << 1.0f, 1.0f,
            1.0f, 1.0f;
    EXPECT_TRUE(relu.derivative(X).isApprox(expected));
}
