#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <cstdint>
#include <vector>

struct MnistImages {
  uint32_t num_images;
  uint32_t rows;
  uint32_t columns;
  std::vector<std::vector<std::vector<unsigned char>>> images;
};

struct MnistLabels {
  uint32_t num_labels;
  std::vector<unsigned char> labels;
};

#endif // !MNIST_DATA_H
