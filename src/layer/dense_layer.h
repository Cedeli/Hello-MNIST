#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "activation/activation.h"
#include "activation/softmax.h"
#include "optimizer/optimizer.h"
#include <Eigen/Core>
#include <memory>
#include <random>

namespace hmnist::layer {
    class DenseLayer final : public Layer {
    public:
        DenseLayer(int in_dim, int out_dim, std::unique_ptr<activation::Activation> act);

        Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

        Eigen::MatrixXf backward(const Eigen::MatrixXf &grad) override;

        void update(optimizer::Optimizer &opt) override;

        Eigen::MatrixXf W, Xc, Zc;
        Eigen::RowVectorXf b;
        Eigen::MatrixXf dW;
        Eigen::RowVectorXf db;
    private:
        std::unique_ptr<activation::Activation> m_activation;
    };
}


#endif //DENSE_LAYER_H
