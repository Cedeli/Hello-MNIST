#include <gtest/gtest.h>
#include <Eigen/Core>
#include "loss/cross_entropy_loss.h"

TEST(CrossEntropyLossTest, LossPerfectPrediction) {
    hmnist::loss::CrossEntropyLoss cel;
    Eigen::MatrixXf Y(2, 2);
    Y << 1.0f, 0.0f,
         0.0f, 1.0f;
    Eigen::MatrixXf Yp = Y;
    EXPECT_NEAR(cel.loss(Yp, Y), 0.0f, 1e-5f);
}

TEST(CrossEntropyLossTest, GradientCorrectness) {
    hmnist::loss::CrossEntropyLoss cel;
    Eigen::MatrixXf Y(1, 2);
    Y << 1.0f, 0.0f;
    Eigen::MatrixXf Yp(1, 2);
    Yp << 0.7f, 0.3f;
    Eigen::MatrixXf expected_grad(1,2);
    expected_grad << -0.3f, 0.3f;
    EXPECT_TRUE(cel.gradient(Yp, Y).isApprox(expected_grad));
}