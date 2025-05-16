#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "data/mnist_loader.h"

namespace {
    void write_idx_images(const std::filesystem::path &p, const uint32_t count, const uint32_t rows,
                          const uint32_t cols) {
        std::ofstream out(p, std::ios::binary);
        auto write32 = [&](uint32_t x) {
            uint32_t v = __builtin_bswap32(x);
            out.write(reinterpret_cast<char *>(&v), 4);
        };
        write32(2051);
        write32(count);
        write32(rows);
        write32(cols);
        std::vector<uint8_t> buf(rows * cols, 42);
        for (uint32_t i = 0; i < count; ++i) out.write(reinterpret_cast<char *>(buf.data()), buf.size());
    }

    void write_idx_labels(const std::filesystem::path &p, const uint32_t count) {
        std::ofstream out(p, std::ios::binary);
        auto write32 = [&](uint32_t x) {
            uint32_t v = __builtin_bswap32(x);
            out.write(reinterpret_cast<char *>(&v), 4);
        };
        write32(2049);
        write32(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto lbl = static_cast<uint8_t>(i % 10);
            out.write(reinterpret_cast<char *>(&lbl), 1);
        }
    }
}

TEST(MnistLoaderTest, LoadsCorrectDimensions) {
    auto tmp = std::filesystem::temp_directory_path();
    auto img = tmp / "t-images-idx3-ubyte";
    auto lbl = tmp / "t-labels-idx1-ubyte";
    write_idx_images(img, 5, 28, 28);
    write_idx_labels(lbl, 5);

    hmnist::data::MnistLoader loader;
    auto ds = loader.load(img.string(), lbl.string());
    EXPECT_EQ(ds.images.rows(), 5);
    EXPECT_EQ(ds.images.cols(), 28 * 28);
    EXPECT_EQ(ds.labels.rows(), 5);
    EXPECT_EQ(ds.labels.cols(), 10);
    EXPECT_FLOAT_EQ(ds.images(0,0), 42.0f / 255.0f);
    EXPECT_EQ(ds.labels(0, 0), 1.0f);
}

TEST(MnistLoaderTest, ThrowsOnBadMagic) {
    auto tmp = std::filesystem::temp_directory_path();
    auto img = tmp / "bad-images-idx3-ubyte";
    std::ofstream out(img, std::ios::binary);
    uint32_t bad = __builtin_bswap32(0xDEADBEEF);
    out.write(reinterpret_cast<char *>(&bad), 4);
    hmnist::data::MnistLoader loader;
    EXPECT_THROW(loader.load(img.string(), img.string()), std::runtime_error);
}
