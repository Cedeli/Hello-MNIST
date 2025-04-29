#include "mnist_parser.h"
#include "file_reader.h"
#include "mnist_data.h"

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

  images.num_images = amount;
  images.rows = rows;
  images.columns = cols;

  images.value.resize(amount);

  for (size_t i = 0; i < amount; ++i) {
    images.value[i].resize(rows);
    for (size_t r = 0; r < rows; ++r) {
      images.value[i][r].resize(cols);

      char buffer[cols];
      if (!reader_.file.read(buffer, cols)) {
        std::cerr << "Error reading pixel data for image: " << i << ", row" << r
                  << '\n';
        reader_.close();
        return false;
      }

      for (size_t c = 0; c < cols; ++c) {
        images.value[i][r][c] = static_cast<unsigned char>(buffer[c]);
      }
    }

    if (i % 1000 == 0) {
      std::cout << "Processed " << i << " images..." << '\n';
    }
  }

  std::cout << "Successfully parsed " << amount << " images" << '\n';
  reader_.close();
  return true;
}

bool MnistParser::parse_labels(std::string &path, MnistLabels &labels) {
  if (!reader_.open(path)) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return false;
  }

  uint32_t magic, amount;

  if (!reader_.read_uint32(magic) || magic != 2049 ||
      !reader_.read_uint32(amount)) {
    std::cerr << "Failed to read label file header or invalid magic number"
              << std::endl;
    reader_.close();
    return false;
  }

  std::cout << "Magic Number: " << magic << '\n';
  std::cout << "Label Amount: " << amount << '\n';

  labels.num_labels = amount;

  labels.value.resize(amount);

  std::vector<char> buffer(amount);
  if (!reader_.file.read(buffer.data(), amount)) {
    std::cerr << "Error reading label data" << std::endl;
    reader_.close();
    return false;
  }

  for (size_t i = 0; i < amount; ++i) {
    labels.value[i] = static_cast<unsigned char>(buffer[i]);
  }

  std::cout << "Successfully parsed " << amount << " labels" << std::endl;
  reader_.close();
  return true;
}

} // namespace mnist
