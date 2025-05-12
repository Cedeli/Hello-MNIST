#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H
#include "data_loader.h"
#include <fstream>

namespace hmnist::data {
    class MnistLoader final : public DataLoader {
    public:
        DataSet load(const std::string &image_path, const std::string &label_path) override;

    private:
        static uint32_t readUInt(std::ifstream &stream);
    };
}

#endif //MNIST_LOADER_H
