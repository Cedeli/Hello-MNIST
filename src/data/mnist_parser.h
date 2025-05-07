#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H
#include "../utils/file_reader.h"
#include "mnist_data.h"
#include <string>

namespace mnist {

class MnistParser {
public:
    MnistParser();
    ~MnistParser();

    bool parse_images(const std::string &path, MnistImages &images);
    bool parse_labels(const std::string &path, MnistLabels &labels);

private:
    FileReader reader;
};

}

#endif //MNIST_PARSER_H
