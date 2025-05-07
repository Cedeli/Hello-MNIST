#include "data_utils.h"

Eigen::MatrixXf
mnist::DataUtils::prepare_image_data(const MnistImages &raw_images) {
  size_t count = raw_images.num_images;
  size_t cols = 784;

  Eigen::MatrixXf image_matrix(count, 784);

  for (size_t i = 0; i < count; ++i) {
    for (size_t r = 0; r < raw_images.rows; ++r) {
      for (size_t c = 0; c < raw_images.columns; ++c) {
        // Calculates where the pixel (r, c) belongs in the flattened 784
        // feature vector. row index * row stride + offset
        uint32_t flat_col_index = r * raw_images.columns + c;
        image_matrix(i, flat_col_index) = raw_images.value[i][r][c] / 255.0f;
      }
    }
  }

  return image_matrix;
}

Eigen::MatrixXf
mnist::DataUtils::prepare_label_data(const MnistLabels &raw_labels) {
  size_t count = raw_labels.num_labels;
  size_t cols = 10;

  // One-hot encoding setup
  Eigen::MatrixXf label_matrix(count, cols);
  label_matrix.setZero();

  for (size_t i = 0; i < count; ++i) {
    size_t value = raw_labels.value[i];
    label_matrix(i, value) = 1.0f;
  }

  return label_matrix;
}
