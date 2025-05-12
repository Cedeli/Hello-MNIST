#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H
#include "loss.h"
#include <Eigen/Core>

namespace hmnist::loss {
    class CrossEntropyLoss final : public Loss {
    public:
        float loss(const Eigen::MatrixXf &Yp, const Eigen::MatrixXf &Y) override;

        Eigen::MatrixXf gradient(const Eigen::MatrixXf &Yp, const Eigen::MatrixXf &Y) override;
    };
}

#endif //CROSS_ENTROPY_LOSS_H
