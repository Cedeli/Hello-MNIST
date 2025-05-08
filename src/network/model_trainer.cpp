#include "model_trainer.h"

#include <numeric>
#include <random>

#include "optimizer/sgd_optimizer.h"

void mnist::ModelTrainer::train(const Eigen::MatrixXf &train_images, const Eigen::MatrixXf &train_labels, int epochs,
                                const float learning_rate, const int batch_size) {
    const int n_samples = static_cast<int>(train_images.rows());
    const int n_batches = (n_samples + batch_size - 1) / batch_size;

    SgdOptimizer sgd(learning_rate, 0.9f);

    std::vector<int> idx(n_samples);
    std::iota(idx.begin(), idx.end(), 0);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::ranges::shuffle(idx, std::mt19937{std::random_device{}()});

        float epoch_loss = 0.0f;
        for (int b = 0; b < n_batches; ++b) {
            const int start = b * batch_size;
            const int count = std::min(batch_size, n_samples - start);

            Eigen::MatrixXf X(count, train_images.cols());
            Eigen::MatrixXf Y(count, train_labels.cols());
            for (int i = 0; i < count; ++i) {
                X.row(i) = train_images.row(idx[start + i]);
                Y.row(i) = train_labels.row(idx[start + i]);
            }

            network->backward(X, Y, sgd);

            const float batch_loss = NeuralNetwork::calculate_loss(network->forward(X), Y);
            epoch_loss += batch_loss;
        }
        epoch_loss /= static_cast<float>(n_batches);
        std::cout << "Epoch " << epoch << "/" << epochs << " Loss: " << epoch_loss << "\n";
    }
}

float mnist::ModelTrainer::evaluate(const Eigen::MatrixXf &test_images, const Eigen::MatrixXf &test_labels) {
    Eigen::MatrixXf preds = network->forward(test_images);

    const int rows = static_cast<int>(preds.rows());
    Eigen::VectorXi pred_labels(rows);
    Eigen::VectorXi true_labels(rows);

    for (int i = 0; i < rows; ++i) {
        preds.row(i).maxCoeff(&pred_labels(i));
        test_labels.row(i).maxCoeff(&true_labels(i));
    }

    const int correct = static_cast<int>((pred_labels.array() == true_labels.array()).count());
    return static_cast<float>(correct) / static_cast<float>(rows);
}
