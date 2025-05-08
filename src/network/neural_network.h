#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "optimizer/optimizer.h"
#include <Eigen/Core>

// TO DO: Refactor with OOP principles.
// Create abstract classes for Layers, Optimizers, and Loss functions.
namespace mnist {
    class NeuralNetwork {
    public:
        NeuralNetwork();

        ~NeuralNetwork() = default;

        void initialize();

        Eigen::MatrixXf forward(const Eigen::MatrixXf &input);

        void backward(const Eigen::MatrixXf &input, const Eigen::MatrixXf &labels, Optimizer &optimizer);

        static float calculate_loss(const Eigen::MatrixXf &prediction, const Eigen::MatrixXf &labels);

    private:
        Eigen::MatrixXf W1;
        Eigen::RowVectorXf b1;

        Eigen::MatrixXf W2;
        Eigen::RowVectorXf b2;

        Eigen::MatrixXf W3;
        Eigen::RowVectorXf b3;

        Eigen::MatrixXf Z1;
        Eigen::MatrixXf A1;
        Eigen::MatrixXf Z2;
        Eigen::MatrixXf A2;
        Eigen::MatrixXf Z3;
        Eigen::MatrixXf A3;

        static Eigen::MatrixXf relu(const Eigen::MatrixXf &input);

        static Eigen::MatrixXf softmax(const Eigen::MatrixXf &input);
    };
}; // namespace mnist

#endif // !NEURAL_NETWORK_H
