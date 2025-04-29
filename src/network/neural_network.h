#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

// TO DO: Refactor with OOP principles.
// Create abstract classes for Layers, Optimizers, and Loss functions.
namespace mnist {
class NeuralNetwork {
public:
  NeuralNetwork();
  ~NeuralNetwork() = default;

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input);

private:
  Eigen::MatrixXf W1;
  Eigen::RowVectorXf b1;

  Eigen::MatrixXf W2;
  Eigen::RowVectorXf b2;

  Eigen::MatrixXf W3;
  Eigen::RowVectorXf b3;

  Eigen::MatrixXf relu(const Eigen::MatrixXf &input);
  Eigen::MatrixXf softmax(const Eigen::MatrixXf &input);

  void initialize();
};
}; // namespace mnist

#endif // !NEURAL_NETWORK_H
