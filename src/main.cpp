#include "mnist_parser.h"

int main() {
  mnist::parse_image("../data/t10k-images-idx3-ubyte");
  return 0;
}
