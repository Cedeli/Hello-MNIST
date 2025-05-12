#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "activation.h"
#include <Eigen/Core>

namespace hmnist::layer::activation {
    class Softmax final : public Activation {
    public:
        Eigen::MatrixXf activate(const Eigen::MatrixXf &X) override;

        Eigen::MatrixXf derivative(const Eigen::MatrixXf &X) override;
    };
}

#endif //SOFTMAX_H
