#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H

#include "../utils/file_reader.h"
#include "mnist_data.h"
#include <iostream>
#include <memory>
#include <string>

namespace mnist {

class MnistParser {
public:
    MnistParser();
    ~MnistParser();

    static bool parse_images(const std::string &path, MnistImages &images);
    static bool parse_labels(const std::string &path, MnistLabels &labels);

private:
    inline static auto reader = std::make_unique<FileReader>();
};

} // namespace mnist

#endif //MNIST_PARSER_H
