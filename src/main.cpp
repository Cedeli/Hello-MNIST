#include "data/mnist_data.h"
#include "data/mnist_parser.h"
#include "utils/data_utils.h"
#include <Eigen/Core>
#include <iostream>

int main() {
  mnist::MnistParser parser;
  mnist::MnistImages raw_images;
  mnist::MnistLabels raw_labels;

  std::string path = "../data/t10k-images-idx3-ubyte";
  std::string label_path = "../data/t10k-labels-idx1-ubyte";

  parser.parse_images(path, raw_images);
  parser.parse_labels(label_path, raw_labels);

  Eigen::MatrixXf training_images =
      mnist::DataUtils::prepare_image_data(raw_images);
  Eigen::MatrixXf training_labels =
      mnist::DataUtils::prepare_label_data(raw_labels);

  std::cout << "\nMiddle 5x10 block (rows 0-4, cols 300-309) of image matrix:\n"
            << training_images.block(0, 300, 5, 10) << std::endl;

  std::cout << "\nFirst 5 rows of label matrix:\n"
            << training_labels.block(0, 0, 5, 10) << std::endl;

  std::cout << "\nLabel matrix row 0 (expected label 7):\n"
            << training_labels.row(0) << std::endl;
  std::cout << "Label matrix row 1 (expected label 2):\n"
            << training_labels.row(1) << std::endl;

  return 0;
}
