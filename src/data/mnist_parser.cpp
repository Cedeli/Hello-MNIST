#include "mnist_parser.h"

mnist::MnistParser::MnistParser() = default;

mnist::MnistParser::~MnistParser() { reader.close(); }

bool mnist::MnistParser::parse_images(const std::string &path,
                                      MnistImages &images) {
  if (!reader.open(const_cast<std::string &>(path))) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return false;
  }

  uint32_t magic_number;
  if (!reader.read_uint32(magic_number)) {
    std::cerr << "Failed to read magic number" << std::endl;
    reader.close();
    return false;
  }

  std::cout << "Magic Number: " << magic_number << std::endl;
  if (magic_number != 2051) {
    std::cerr << "Invalid magic number for images file" << std::endl;
    reader.close();
    return false;
  }

  uint32_t num_images;
  if (!reader.read_uint32(num_images)) {
    std::cerr << "Failed to read number of images" << std::endl;
    reader.close();
    return false;
  }

  uint32_t num_rows;
  if (!reader.read_uint32(num_rows)) {
    std::cerr << "Failed to read number of rows" << std::endl;
    reader.close();
    return false;
  }

  uint32_t num_columns;
  if (!reader.read_uint32(num_columns)) {
    std::cerr << "Failed to read number of columns" << std::endl;
    reader.close();
    return false;
  }

  std::cout << "Image Amount: " << num_images << std::endl;
  std::cout << "Image Rows: " << num_rows << std::endl;
  std::cout << "Image Columns: " << num_columns << std::endl;

  images.num_images = num_images;
  images.rows = num_rows;
  images.columns = num_columns;

  // Allocate memory for image data
  images.value = new uint8_t **[num_images];
  for (uint32_t i = 0; i < num_images; ++i) {
    images.value[i] = new uint8_t *[num_rows];
    for (uint32_t r = 0; r < num_rows; ++r) {
      images.value[i][r] = new uint8_t[num_columns];
    }
  }

  char pixel;
  for (uint32_t i = 0; i < num_images; ++i) {
    if (i % 1000 == 0) {
      std::cout << "Processed " << i << " images..." << std::endl;
    }

    for (uint32_t r = 0; r < num_rows; ++r) {
      for (uint32_t c = 0; c < num_columns; ++c) {
        if (!reader.file.read(&pixel, 1)) {
          std::cerr << "Failed to read pixel data at image " << i << ", row "
                    << r << ", column " << c << std::endl;
          reader.close();
          return false;
        }
        images.value[i][r][c] = static_cast<uint8_t>(pixel);
      }
    }
  }

  reader.close();
  std::cout << "Successfully parsed " << num_images << " images" << std::endl;
  return true;
}

bool mnist::MnistParser::parse_labels(const std::string &path,
                                      MnistLabels &labels) {
  if (!reader.open(const_cast<std::string &>(path))) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return false;
  }

  uint32_t magic_number;
  if (!reader.read_uint32(magic_number)) {
    std::cerr << "Failed to read magic number" << std::endl;
    reader.close();
    return false;
  }

  std::cout << "Magic Number: " << magic_number << std::endl;
  if (magic_number != 2049) {
    std::cerr << "Invalid magic number for labels file" << std::endl;
    reader.close();
    return false;
  }

  uint32_t num_labels;
  if (!reader.read_uint32(num_labels)) {
    std::cerr << "Failed to read number of labels" << std::endl;
    reader.close();
    return false;
  }

  std::cout << "Label Amount: " << num_labels << std::endl;

  labels.num_labels = num_labels;

  // Allocate memory for label data
  labels.value = new uint8_t[num_labels];

  char label;
  for (uint32_t i = 0; i < num_labels; ++i) {
    if (!reader.file.read(&label, 1)) {
      std::cerr << "Failed to read label data at index " << i << std::endl;
      reader.close();
      return false;
    }
    labels.value[i] = static_cast<uint8_t>(label);
  }

  reader.close();
  std::cout << "Successfully parsed " << num_labels << " labels" << std::endl;
  return true;
}
