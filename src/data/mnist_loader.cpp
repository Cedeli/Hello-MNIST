#include "mnist_loader.h"

#if defined(_MSC_VER)
  #include <stdlib.h>
  #define bswap32(x) _byteswap_ulong(x)
#else
  #define bswap32(x) __builtin_bswap32(x)
#endif

hmnist::data::DataSet hmnist::data::MnistLoader::load(const std::string &image_path, const std::string &label_path) {
    std::ifstream fi(image_path, std::ios::binary);
    std::ifstream fj(label_path, std::ios::binary);

    if (!fi.is_open()) throw std::runtime_error("Cannot open MNIST image file: " + image_path);
    if (!fj.is_open()) throw std::runtime_error("Cannot open MNIST label file: " + label_path);

    if (readUInt(fi) != 2051) throw std::runtime_error("Invalid MNIST image file: " + image_path);
    uint32_t n_images = readUInt(fi);
    uint32_t rows = readUInt(fi);
    uint32_t cols = readUInt(fi);
    uint32_t image_res = rows * cols;

    if (readUInt(fj) != 2049) throw std::runtime_error("Invalid MNIST label file: " + label_path);
    if (uint32_t n_labels = readUInt(fj); n_images != n_labels) {
        throw std::runtime_error(
            "The number of images (" + std::to_string(n_images) +
            ") must be equal to the number of labels (" + std::to_string(n_labels) + ")");
    }

    DataSet ds;
    ds.images.resize(n_images, image_res);
    ds.labels.setZero(n_images, 10);
    std::vector<uint8_t> image_buffer(image_res);

    for (uint32_t i = 0; i < n_images; ++i) {
        // Read raw pixel data for one image into image_buffer.
        fi.read(reinterpret_cast<char *>(image_buffer.data()), image_buffer.size());
        if (!fi) {
            throw std::runtime_error("Error reading image data for image index " + std::to_string(i));
        }

        uint8_t label;
        fj.read(reinterpret_cast<char *>(&label), 1);
        if (!fj) {
            throw std::runtime_error("Error reading label for image index " + std::to_string(i));
        }

        // Normalize and set each pixel value in our image matrix.
        for (uint32_t j = 0; j < image_res; ++j) {
            ds.images(i, j) = static_cast<float>(image_buffer[j]) / 255.0f;
        }

        // One hot encoding.
        ds.labels(i, label) = 1.0f;
    }

    return ds;
}

uint32_t hmnist::data::MnistLoader::readUInt(std::ifstream &stream) {
    uint32_t x = 0;
    stream.read(reinterpret_cast<char *>(&x), 4);
    return bswap32(x);
}
