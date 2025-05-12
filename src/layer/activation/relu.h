#ifndef RELU_H
#define RELU_H
#include "activation.h"
#include <Eigen/Core>

namespace hmnist::layer::activation {
    class Relu final : public Activation {
    public:
        Eigen::MatrixXf activate(const Eigen::MatrixXf &X) override;

        Eigen::MatrixXf derivative(const Eigen::MatrixXf &X) override;
    };
}

#endif //RELU_H
