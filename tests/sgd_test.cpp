#include <gtest/gtest.h>
#include <Eigen/Core>
#include "optimizer/sgd.h"

TEST(SgdTest, StepUpdatesParametersMatrix) {
    hmnist::optimizer::Sgd sgd(0.1f);
    Eigen::MatrixXf params(2, 2);
    params << 1.0f, 2.0f,
            3.0f, 4.0f;
    Eigen::MatrixXf grad(2, 2);
    grad << 0.5f, 1.0f,
            1.5f, 2.0f;
    Eigen::MatrixXf expected = params - 0.1f * grad;
    sgd.step(params, grad);
    EXPECT_TRUE(params.isApprox(expected));
}
