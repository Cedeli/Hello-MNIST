#ifndef MNIST_FILE_READER_H
#define MNIST_FILE_READER_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace mnist {
class FileReader {
private:
  std::ifstream file_;
  uint32_t big_endian(char *buffer);

public:
  bool open(std::string &path);
  void close();
  bool read_uint32(uint32_t &value);
};
} // namespace mnist

#endif // !MNIST_FILE_READER_H
