#include <gtest/gtest.h>
#include <Eigen/Core>
#include "model/trainer.h"
#include "model/network.h"
#include "layer/dense_layer.h"
#include "layer/activation/softmax.h"
#include "layer/activation/leaky_relu.h"
#include "loss/cross_entropy_loss.h"
#include "optimizer/sgd.h"

TEST(TrainerTest, OverfitsTinyDataset) {
    Eigen::MatrixXf X(4, 2);
    X << 0, 0, 0, 1, 1, 0, 1, 1;
    Eigen::MatrixXf Y(4, 2);
    Y << 1, 0, 0, 1, 0, 1, 1, 0;
    std::vector<std::unique_ptr<hmnist::layer::Layer> > layers;
    layers.emplace_back(
        std::make_unique<hmnist::layer::DenseLayer>(2, 4, std::make_unique<hmnist::layer::activation::LeakyRelu>()));
    layers.emplace_back(
        std::make_unique<hmnist::layer::DenseLayer>(4, 2, std::make_unique<hmnist::layer::activation::Softmax>()));
    auto net = std::make_unique<hmnist::model::Network>(std::move(layers),
                                                        std::make_unique<hmnist::loss::CrossEntropyLoss>(),
                                                        std::make_unique<hmnist::optimizer::Sgd>(0.5f));
    const hmnist::model::Trainer trainer(std::move(net));
    trainer.train(X, Y, 500, 4);
    const float acc = trainer.evaluate(X, Y);
    EXPECT_GT(acc, 0.99f);
}
