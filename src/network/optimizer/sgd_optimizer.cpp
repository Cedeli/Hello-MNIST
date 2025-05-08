#include "sgd_optimizer.h"

mnist::SgdOptimizer::SgdOptimizer(const float lr, const float momentum) : learning_rate(lr), beta(momentum) {
}

void mnist::SgdOptimizer::update(Eigen::MatrixXf &param, const Eigen::MatrixXf &grad) {
    const void *key = &param;
    Eigen::MatrixXf &v = velocity_map[key];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
        v = Eigen::MatrixXf::Zero(grad.rows(), grad.cols());
    }
    v = beta * v + (1.0f - beta) * grad;
    param -= learning_rate * v;
}

void mnist::SgdOptimizer::update(Eigen::RowVectorXf &param, const Eigen::RowVectorXf &grad) {
    const void *key = &param;
    Eigen::MatrixXf &v = velocity_map[key];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
        v = Eigen::MatrixXf::Zero(grad.rows(), grad.cols());
    }
    v = beta * v + (1.0f - beta) * grad;
    param -= learning_rate * v.row(0);
}
