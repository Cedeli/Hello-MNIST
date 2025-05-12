#ifndef TRAINER_H
#define TRAINER_H
#include "layer/layer.h"
#include "loss/loss.h"
#include "optimizer/optimizer.h"
#include "network.h"
#include <memory>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>

namespace hmnist::model {
    class Trainer {
    public:
        explicit Trainer(std::unique_ptr<Network> network);

        void train(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y, int epochs, int batch_size) const;

        float evaluate(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y) const;

        int predict(const Eigen::VectorXf &X) const;

    private:
        std::unique_ptr<Network> m_network;
    };
}

#endif //TRAINER_H
