#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H
#include "activation.h"
#include <Eigen/Core>

namespace hmnist::layer::activation {
    class LeakyRelu final : public Activation {
    public:
        Eigen::MatrixXf activate(const Eigen::MatrixXf &X) override;

        Eigen::MatrixXf derivative(const Eigen::MatrixXf &X) override;
    };
}


#endif //LEAKY_RELU_H
