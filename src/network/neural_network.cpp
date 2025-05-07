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
  // Z1 = X * W1 + b1
  Z1 = (input * W1).rowwise() + b1;
  A1 = relu(Z1);

  // Z2 = A1 * W2 + b2
  Z2 = (A1 * W2).rowwise() + b2;
  A2 = relu(Z2);

  // Z3 = A2 * W3 + b3
  Z3 = (A2 * W3).rowwise() + b3;
  A3 = softmax(Z3);
  return A3;
}

void mnist::NeuralNetwork::backward(const Eigen::MatrixXf &input,
                                    const Eigen::MatrixXf &label,
                                    float learning_rate) {
  int batch_size = input.rows();

  forward(input);

  // 1. Calculate output layer error
  Eigen::MatrixXf dZ3 = A3 - label;

  // 2. Calculate gradients for output layer
  //    dW3 = A2^T * dZ3
  Eigen::MatrixXf dW3 = A2.transpose() * dZ3 / batch_size;
  //    db3 = sum of dZ3 across all examples
  Eigen::VectorXf db3 = dZ3.colwise().sum() / batch_size;

  // 3. Backpropagate to hidden layer 2
  //    dA2 = dZ3 * W3^T
  Eigen::MatrixXf dA2 = dZ3 * W3.transpose();
  //    Apply ReLU derivative
  Eigen::MatrixXf dZ2 = dA2.array() * (Z2.array() > 0.0f).cast<float>();
  //    Calculate gradients: dW2 = A1^T * dZ2
  Eigen::MatrixXf dW2 = A1.transpose() * dZ2 / batch_size;
  //    db2 = sum of dZ2 across all examples
  Eigen::VectorXf db2 = dZ2.colwise().sum() / batch_size;

  // 6. Backpropagate to hidden layer 1
  //    dA1 = dZ2 * W2^T
  Eigen::MatrixXf dA1 = dZ2 * W2.transpose();
  //    Apply ReLU derivative
  Eigen::MatrixXf dZ1 = dA1.array() * (Z1.array() > 0.0f).cast<float>();
  //    Calculate gradients: dW1 = X^T * dZ1
  Eigen::MatrixXf dW1 = input.transpose() * dZ1 / batch_size;
  //    db1 = sum of dZ1 across all examples
  Eigen::VectorXf db1 = dZ1.colwise().sum() / batch_size;

  // 7. Update weights and biases
  W3 -= learning_rate * dW3;
  b3 -= learning_rate * db3;
  W2 -= learning_rate * dW2;
  b2 -= learning_rate * db2;
  W1 -= learning_rate * dW1;
  b1 -= learning_rate * db1;
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
