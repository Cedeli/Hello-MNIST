#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Core>
#include <optimizer/optimizer.h>

namespace hmnist::layer {
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Eigen::MatrixXf forward(const Eigen::MatrixXf &X) = 0;

        virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &grad) = 0;

        virtual void update(optimizer::Optimizer &opt) = 0;
    };
}

#endif //LAYER_H
