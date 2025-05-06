#ifndef MODEL_TRAINER_H
#define MODEL_TRAINER_H

#include "neural_network.h"

namespace mnist {

class ModelTrainer {
public:
  ModelTrainer(NeuralNetwork &network) : m_network(network) {}

  void train(const Eigen::MatrixXf &train_images,
             const Eigen::MatrixXf &train_labels, const Eigen::MatrixXf &images,
             const Eigen::MatrixXf &labels, int epochs = 10,
             int batch_size = 32, float learning_rate = 0.01f);

  float evaluate(const Eigen::MatrixXf &test_images,
                 const Eigen::MatrixXf &test_labels);

private:
  NeuralNetwork &m_network;
};

} // namespace mnist

#endif
