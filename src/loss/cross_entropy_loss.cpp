#include "cross_entropy_loss.h"

float hmnist::loss::CrossEntropyLoss::loss(const Eigen::MatrixXf &Yp, const Eigen::MatrixXf &Y) {
    // Max to avoid log(0)
    const Eigen::ArrayXXf p = Yp.array().max(1e-9f);
    return -(Y.array() * p.log()).sum() / Y.rows();
}

Eigen::MatrixXf hmnist::loss::CrossEntropyLoss::gradient(const Eigen::MatrixXf &Yp, const Eigen::MatrixXf &Y) {
    return Yp - Y;
}
