#include "mnist_data.h"
#include "mnist_parser.h"

int main() {
  mnist::MnistParser parser;
  MnistImages images;
  MnistLabels labels;

  std::string path = "../data/t10k-images-idx3-ubyte";
  std::string label_path = "../data/t10k-labels-idx1-ubyte";
  parser.parse_images(path, images);
  parser.parse_labels(label_path, labels);
  return 0;
}
