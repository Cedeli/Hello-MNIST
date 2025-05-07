#ifndef MNIST_FILE_READER_H
#define MNIST_FILE_READER_H

#include <cstdint>
#include <fstream>
#include <string>

namespace mnist {
    class FileReader {
    public:
        std::ifstream file;

        bool open(std::string &path);

        void close();

        bool read_uint32(uint32_t &value);

    private:
        static uint32_t big_endian(const char *buffer);
    };
} // namespace mnist

#endif // !MNIST_FILE_READER_H
