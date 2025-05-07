#ifndef MODEL_TRAINER_H
#define MODEL_TRAINER_H

#include "neural_network.h"
#include <Eigen/Core>
#include <iostream>
#include <memory>

namespace mnist {
    class ModelTrainer {
    public:
        static void train(const Eigen::MatrixXf &train_images, const Eigen::MatrixXf &train_labels, int epochs = 10,
                          float learning_rate = 0.01f, int batch_size = 32);

        static float evaluate(const Eigen::MatrixXf &test_images, const Eigen::MatrixXf &test_labels);

    private:
        inline static auto network = std::make_unique<NeuralNetwork>();
    };
} // namespace mnist

#endif
