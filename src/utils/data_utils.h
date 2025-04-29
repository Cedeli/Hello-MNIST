#ifndef MNIST_DATA_UTILS_H
#define MNIST_DATA_UTILS_H

#include "../data/mnist_data.h"
#include <eigen3/Eigen/Core>

namespace mnist {
class DataUtils {
public:
  static Eigen::MatrixXf prepare_image_data(const MnistImages &raw_images);
  static Eigen::MatrixXf prepare_label_data(const MnistLabels &raw_labels);

private:
  DataUtils() = delete;
  ~DataUtils() = delete;
};
}; // namespace mnist

#endif // !MNIST_DATA_UTILS_H
