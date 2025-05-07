#ifndef MNIST_DATA_H
#define MNIST_DATA_H
#include <cstdint>

namespace mnist {

struct MnistImages {
    uint32_t num_images;
    uint32_t rows;
    uint32_t columns;
    uint8_t ***value;

    MnistImages() : num_images(0), rows(0), columns(0), value(nullptr) {}

    ~MnistImages() {
        if (value) {
            for (uint32_t i = 0; i < num_images; ++i) {
                if (value[i]) {
                    for (uint32_t r = 0; r < rows; ++r) {
                        delete[] value[i][r];
                    }
                    delete[] value[i];
                }
            }
            delete[] value;
            value = nullptr;
        }
    }

    MnistImages(const MnistImages&) = delete;
    MnistImages& operator=(const MnistImages&) = delete;
};

struct MnistLabels {
    uint32_t num_labels;
    uint8_t *value;

    MnistLabels() : num_labels(0), value(nullptr) {}

    ~MnistLabels() {
        delete[] value;
        value = nullptr;
    }

    MnistLabels(const MnistLabels&) = delete;
    MnistLabels& operator=(const MnistLabels&) = delete;
};

} // namespace mnist

#endif //MNIST_DATA_H
