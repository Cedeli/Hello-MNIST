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
  Z1 = (input * W1).rowwise() + b1.transpose();
  A1 = relu(Z1);

  Z2 = (A1 * W2).rowwise() + b2.transpose();
  A2 = relu(Z2);

  Z3 = (A2 * W3).rowwise() + b3.transpose();
  A3 = softmax(Z3);
  return A3;
}

Eigen::MatrixXf mnist::NeuralNetwork::backward(const Eigen::MatrixXf &input) {
  size_t batch_size = input.rows();
  // 1. dZ3 = A3 - Y
  Eigen::MatrixXf dZ3 = A3 - input;

  // 2. dW3 = (A2.transpose() * dZ3) / N
  Eigen::MatrixXf dW3 = (A2.transpose() * dZ3) / batch_size;

  //    db3 = column-wise sum of dZ3, then / N
  Eigen::VectorXf db3 = dZ3.colwise().sum() / batch_size;

  // 3. dA2 = dZ3 * W3.transpose()
  Eigen::MatrixXf dA2 = dZ3 * W3.transpose();

  // 4. dZ2 = elementwise multiply by ReLU derivative mask
  Eigen::MatrixXf relu_mask = (Z2.array() > 0).cast<float>();
  Eigen::MatrixXf dZ2 = dA2.array() * relu_mask.array();

  Eigen::MatrixXf dW2 = (A1.transpose() * dZ2) / batch_size;
  Eigen::VectorXf db2 = dZ2.colwise().sum() / batch_size;

  Eigen::MatrixXf dA1 = dZ2 * W2.transpose();
  Eigen::MatrixXf relu_mask1 = (Z1.array() > 0).cast<float>();
  Eigen::MatrixXf dZ1 = dA1.array() * relu_mask1.array();
  Eigen::MatrixXf dW1 = (input.transpose() * dZ1) / batch_size;
  Eigen::VectorXf db1 = dZ1.colwise().sum() / batch_size;
  return input;
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

float mnist::NeuralNetwork::calculate_loss(const Eigen::MatrixXf &predictions,
                                           const Eigen::MatrixXf &labels) {
  float epsilon = 1e-9f;
  Eigen::MatrixXf log_predictions = (predictions.array() + epsilon).log();
  Eigen::MatrixXf masked_log_predictions =
      log_predictions.array() * labels.array();
  float log_sum = -masked_log_predictions.sum();
  return log_sum / labels.rows();
}
