#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <Eigen/Core>

namespace hmnist::layer::activation {
    class Activation {
    public:
        virtual ~Activation() = default;

        virtual Eigen::MatrixXf activate(const Eigen::MatrixXf &X) = 0;

        virtual Eigen::MatrixXf derivative(const Eigen::MatrixXf &X) = 0;
    };
}

#endif //ACTIVATION_H
