#include <Eigen/Core>
#include <filesystem>
#include <iostream>
#include <memory>
#include <data/mnist_loader.h>
#include <layer/dense_layer.h>
#include <layer/layer.h>
#include <layer/activation/leaky_relu.h>
#include <layer/activation/softmax.h>
#include <loss/cross_entropy_loss.h>
#include <optimizer/sgd.h>
#include <model/trainer.h>

int main(int argc, char* argv[]) {
    constexpr float learning_rate = 0.01f;
    constexpr int epochs = 10;
    constexpr int batch_size = 64;

    std::filesystem::path executable_path(argv[0]);
    std::filesystem::path executable_dir = executable_path.parent_path();

    std::filesystem::path data_path = executable_dir / "../data";
    data_path = weakly_canonical(data_path);

    const std::string train_images_path = (data_path / "train-images-idx3-ubyte").string();
    const std::string train_labels_path = (data_path / "train-labels-idx1-ubyte").string();
    const std::string test_images_path = (data_path / "t10k-images-idx3-ubyte").string();
    const std::string test_labels_path = (data_path / "t10k-labels-idx1-ubyte").string();

    auto loader = hmnist::data::MnistLoader{};
    const auto train = loader.load(train_images_path, train_labels_path);
    const auto test = loader.load(test_images_path, test_labels_path);

    std::vector<std::unique_ptr<hmnist::layer::Layer> > layers;
    layers.emplace_back(
        std::make_unique<hmnist::layer::DenseLayer>(784, 128, std::make_unique<hmnist::layer::activation::LeakyRelu>()));
    layers.emplace_back(
        std::make_unique<hmnist::layer::DenseLayer>(128, 64, std::make_unique<hmnist::layer::activation::LeakyRelu>()));
    layers.emplace_back(
        std::make_unique<hmnist::layer::DenseLayer>(64, 10, std::make_unique<hmnist::layer::activation::Softmax>()));

    auto loss = std::make_unique<hmnist::loss::CrossEntropyLoss>();
    auto opt = std::make_unique<hmnist::optimizer::Sgd>(learning_rate);
    auto network = std::make_unique<hmnist::model::Network>(std::move(layers), std::move(loss), std::move(opt));
    hmnist::model::Trainer trainer(std::move(network));

    trainer.train(train.images, train.labels, epochs, batch_size);
    const float acc = trainer.evaluate(test.images, test.labels);
    std::cout << "Test Accuracy: " << acc * 100.0f << "%\n";

    std::cout << "\nRunning image classification:\n";
    int correct = 0;
    constexpr int tests = 250;

    std::vector<int> indices(test.images.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::ranges::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < tests; ++i) {
        Eigen::VectorXf X = test.images.row(indices[i]);
        int true_label;
        test.labels.row(indices[i]).maxCoeff(&true_label);

        const int pred = trainer.predict(X);
        if (pred == true_label) correct++;
        std::cout << "[" << i << "] prediction: " << pred << ", expected: " << true_label << "\n";
    }

    std::cout << "Prediction Accuracy: " << static_cast<float>(correct) * 100.0f / static_cast<float>(tests) << "%\n";

    return 0;
}
