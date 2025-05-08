#include "data/mnist_data.h"
#include "data/mnist_parser.h"
#include "utils/data_utils.h"
#include <Eigen/Core>
#include <iostream>
#include <network/model_trainer.h>

int main() {
    const std::string train_images_path = "../data/train-images-idx3-ubyte";
    const std::string train_labels_path = "../data/train-labels-idx1-ubyte";
    const std::string test_images_path = "../data/t10k-images-idx3-ubyte";
    const std::string test_labels_path = "../data/t10k-labels-idx1-ubyte";

    mnist::MnistImages raw_train_images;
    mnist::MnistLabels raw_train_labels;
    if (!mnist::MnistParser::parse_images(train_images_path, raw_train_images) ||
        !mnist::MnistParser::parse_labels(train_labels_path, raw_train_labels)) {
        std::cerr << "Failed to parse train images" << "\n";
        return 1;
    }

    mnist::MnistImages raw_test_images;
    mnist::MnistLabels raw_test_labels;
    if (!mnist::MnistParser::parse_images(test_images_path, raw_test_images) ||
        !mnist::MnistParser::parse_labels(test_labels_path, raw_test_labels)) {
        std::cerr << "Failed to parse test images" << "\n";
        return 1;
    }

    const Eigen::MatrixXf train_images = mnist::DataUtils::prepare_image_data(raw_train_images);
    const Eigen::MatrixXf train_labels = mnist::DataUtils::prepare_label_data(raw_train_labels);
    const Eigen::MatrixXf test_images = mnist::DataUtils::prepare_image_data(raw_test_images);
    const Eigen::MatrixXf test_labels = mnist::DataUtils::prepare_label_data(raw_test_labels);

    constexpr int epochs = 10;
    constexpr float learning_rate = 0.01f;
    constexpr int batch_size = 64;

    mnist::ModelTrainer::train(train_images, train_labels, epochs, learning_rate, batch_size);

    const float accuracy = mnist::ModelTrainer::evaluate(test_images, test_labels);
    std::cout << "Test Accuracy: " << accuracy * 100.0f << "%" << "\n";

    return 0;
}
