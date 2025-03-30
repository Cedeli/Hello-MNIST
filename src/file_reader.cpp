#include "file_reader.h"

namespace mnist {
bool FileReader::open(std::string &path) {
  if (file_.is_open()) {
    file_.close();
  }

  file_.open(path, std::ios::binary);
  return file_.is_open();
}

void FileReader::close() {
  if (file_.is_open()) {
    file_.close();
  }
}

bool FileReader::read_uint32(uint32_t &value) {
  char buffer[4];
  if (!file_.read(buffer, 4)) {
    return false;
  }

  value = big_endian(buffer);
  return true;
}

uint32_t FileReader::big_endian(char *buffer) {
  return (static_cast<unsigned char>(buffer[0]) << 24) |
         (static_cast<unsigned char>(buffer[1]) << 16) |
         (static_cast<unsigned char>(buffer[2]) << 8) |
         (static_cast<unsigned char>(buffer[3]));
}
} // namespace mnist
