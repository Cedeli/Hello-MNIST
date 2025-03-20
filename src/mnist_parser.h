#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H

#include <arpa/inet.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace mnist {
void parse_image(std::string path);
void parse_label(std::string path);
} // namespace mnist
#endif // MNIST_PARSER
