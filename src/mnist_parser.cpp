#include "mnist_parser.h"
#include "file_reader.h"

namespace mnist {
MnistParser::MnistParser() : reader_() {}
MnistParser::~MnistParser() { reader_.close(); }

bool MnistParser::parse_images(std::string &path, MnistImages &images) {
  if (!reader_.open(path)) {
    std::cerr << "Failed to open file: " << path << '\n';
    return false;
  }

  uint32_t magic, amount, rows, cols;
  if (!reader_.read_uint32(magic) || magic != 2051 ||
      !reader_.read_uint32(amount) || !reader_.read_uint32(rows) ||
      !reader_.read_uint32(cols)) {
    std::cerr << "Failed to read file or invalid magic number" << '\n';
    reader_.close();
    return false;
  }

  std::cout << "Magic Number: " << magic << '\n';
  std::cout << "Image Amount: " << amount << '\n';
  std::cout << "Image Rows: " << rows << '\n';
  std::cout << "Image Columns: " << cols << '\n';

  reader_.close();
  return true;
}
} // namespace mnist
