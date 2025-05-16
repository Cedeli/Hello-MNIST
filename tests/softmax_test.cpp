#include <gtest/gtest.h>
#include <Eigen/Core>
#include "layer/activation/softmax.h"

TEST(SoftmaxTest, ActivateProbabilitiesSumToOne) {
    hmnist::layer::activation::Softmax softmax;
    Eigen::MatrixXf X(2, 3);
    X << 1.0f, 2.0f, 3.0f,
            -1.0f, 0.0f, 1.0f;
    Eigen::MatrixXf A = softmax.activate(X);
    for (int i = 0; i < A.rows(); ++i) {
        EXPECT_NEAR(A.row(i).sum(), 1.0f, 1e-6f);
    }
}

TEST(SoftmaxTest, DerivativeThrowsException) {
    hmnist::layer::activation::Softmax softmax;
    Eigen::MatrixXf X(1, 1);
    X << 0;
    EXPECT_THROW(softmax.derivative(X), std::runtime_error);
}
