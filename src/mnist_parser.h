#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H

#include "file_reader.h"
#include "mnist_data.h"
#include <string>

namespace mnist {
class MnistParser {
private:
  mnist::FileReader reader_;

public:
  MnistParser();
  ~MnistParser();
  bool parse_images(std::string &path, MnistImages &images);
  bool parse_labels(std::string &path, MnistLabels &labels);
};
} // namespace mnist

#endif // MNIST_PARSER
