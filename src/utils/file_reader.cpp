#include "file_reader.h"

bool mnist::FileReader::open(std::string &path) {
    if (file.is_open()) {
        file.close();
    }

    file.open(path, std::ios::binary);
    return file.is_open();
}

void mnist::FileReader::close() {
    if (file.is_open()) {
        file.close();
    }
}

bool mnist::FileReader::read_uint32(uint32_t &value) {
    char buffer[4];
    if (!file.read(buffer, 4)) {
        return false;
    }

    value = big_endian(buffer);
    return true;
}

uint32_t mnist::FileReader::big_endian(const char *buffer) {
    return (static_cast<unsigned char>(buffer[0]) << 24) |
           (static_cast<unsigned char>(buffer[1]) << 16) |
           (static_cast<unsigned char>(buffer[2]) << 8) |
           (static_cast<unsigned char>(buffer[3]));
}
