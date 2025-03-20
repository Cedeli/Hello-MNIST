#include "mnist_parser.h"

namespace {
std::string file_path;
std::ifstream current_file;
int magic_number;
char magic_buffer[4];

void open_file(std::string path) {
  file_path = path;
  if (current_file.is_open()) {
    current_file.close();
  }

  current_file.open(file_path, std::ios::binary);
  if (!current_file.is_open()) {
    std::cerr << "Error opening file: " << file_path << '\n';
    exit(1);
  }
}
} // namespace

namespace mnist {
void parse_image(std::string path) {
  open_file(path);
  current_file.read(magic_buffer, 4);

  uint32_t *magic_ptr = reinterpret_cast<uint32_t *>(magic_buffer);
  magic_number = ntohl(*magic_ptr);
  std::cout << "Magic Number: " << magic_number << '\n';

  if (magic_number != 2051) {
    std::cerr
        << "Error: Invalid magic number for image file. Expected 2051, got "
        << magic_number << '\n';
    current_file.close();
    return;
  }
}
} // namespace mnist
