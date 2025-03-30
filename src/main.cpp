#include "mnist_data.h"
#include "mnist_parser.h"

int main() {
  mnist::MnistParser parser;
  MnistImages images;

  std::string path = "../data/t10k-images-idx3-ubyte";
  parser.parse_images(path, images);
  return 0;
}
