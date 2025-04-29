#include "neural_network.h"

mnist::NeuralNetwork::NeuralNetwork() { initialize(); }

void mnist::NeuralNetwork::initialize() {
  size_t input_size = 784;
  size_t hidden1_size = 128;
  size_t hidden2_size = 64;
  size_t output_size = 10;
  float scale = 0.01f;

  W1.resize(input_size, hidden1_size);
  W1.setRandom();
  W1 *= scale;

  b1.resize(hidden1_size);
  b1.setZero();

  W2.resize(hidden1_size, hidden2_size);
  W2.setRandom();
  W2 *= scale;

  b2.resize(hidden2_size);
  b2.setZero();

  W3.resize(hidden2_size, output_size);
  W3.setRandom();
  W3 *= scale;

  b3.resize(output_size);
  b3.setRandom();
}

Eigen::MatrixXf mnist::NeuralNetwork::forward(const Eigen::MatrixXf &input) {
  return input;
}

Eigen::MatrixXf mnist::NeuralNetwork::relu(const Eigen::MatrixXf &input) {
  return input;
}

Eigen::MatrixXf mnist::NeuralNetwork::softmax(const Eigen::MatrixXf &input) {
  return input;
}
