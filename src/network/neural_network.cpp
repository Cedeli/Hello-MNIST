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
  b3.setZero();
}

Eigen::MatrixXf mnist::NeuralNetwork::forward(const Eigen::MatrixXf &input) {
  Eigen::MatrixXf Z1 = (input * W1).rowwise() + b1.transpose();
  Eigen::MatrixXf A1 = relu(Z1);

  Eigen::MatrixXf Z2 = (A1 * W2).rowwise() + b2.transpose();
  Eigen::MatrixXf A2 = relu(Z2);

  Eigen::MatrixXf Z3 = (A2 * W3).rowwise() + b3.transpose();
  Eigen::MatrixXf A3 = softmax(Z3);
  return A3;
}

Eigen::MatrixXf mnist::NeuralNetwork::relu(const Eigen::MatrixXf &input) {
  Eigen::MatrixXf result = input.array().cwiseMax(0);
  return result;
}

Eigen::MatrixXf mnist::NeuralNetwork::softmax(const Eigen::MatrixXf &input) {
  Eigen::VectorXf max_values = input.rowwise().maxCoeff();
  Eigen::ArrayXXf shifted = input.array().colwise() - max_values.array();
  Eigen::ArrayXf exps = shifted.exp();
  Eigen::ArrayXf sums = exps.rowwise().sum();
  return (exps.colwise() / sums).matrix();
}
