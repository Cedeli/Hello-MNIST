#include "trainer.h"

hmnist::model::Trainer::Trainer(std::unique_ptr<Network> network) : m_network(std::move(network)) {
}

void hmnist::model::Trainer::train(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y, const int epochs,
                                     const int batch_size) const {
    const int N = static_cast<int>(X.rows());
    float current_loss_value = 0.0f;

    std::vector<int> indices(N);
    // std::iota(indices.begin(), indices.end(), 0);
    // std::random_device rd;
    // std::mt19937 rng(rd());
    // TEMPORARILY TEST
    std::mt19937 rng(42);
    for (int e = 0; e < epochs; ++e) {
        std::cout << "Starting Epoch " << e + 1 << "/" << epochs << std::endl;
        std::ranges::shuffle(indices, rng);

        Eigen::MatrixXf X_shuffled(N, X.cols());
        Eigen::MatrixXf Y_shuffled(N, Y.cols());

        for (int j = 0; j < N; ++j) {
            X_shuffled.row(j) = X.row(indices[j]);
            Y_shuffled.row(j) = Y.row(indices[j]);
        }

        for (int i = 0; i < N; i += batch_size) {
            // Determining actual batch size as the last batch size might be smaller
            const int current_batch_size = std::min(batch_size, N - i);
            if (current_batch_size <= 0) continue;

            // Take a batch from the shuffled matrix
            Eigen::MatrixXf xb = X_shuffled.middleRows(i, current_batch_size);
            Eigen::MatrixXf yb = Y_shuffled.middleRows(i, current_batch_size);

            // Start forward pass with input batch
            Eigen::MatrixXf current_activation = xb;
            // Propagate to each layer
            for (const auto &layer_ptr: m_network->layers) {
                current_activation = layer_ptr->forward(current_activation);
            }
            // Output probabilities
            Eigen::MatrixXf final_output_Yp = current_activation;

            // Compute loss with the batch
            current_loss_value = m_network->loss->loss(final_output_Yp, yb);

            // Start backward pass by calculating gradient of loss w.r.t. the output
            Eigen::MatrixXf grad_chain = m_network->loss->gradient(final_output_Yp, yb);

            // Propagate gradients to previous layers
            for (const auto &layer_ptr: std::ranges::reverse_view(m_network->layers)) {
                grad_chain = layer_ptr->backward(grad_chain);
            }

            // Update weights and biases
            for (const auto &layer_ptr: m_network->layers) {
                layer_ptr->update(*m_network->optimizer);
            }
        }
        std::cout << "Epoch " << e + 1 << " complete. Last batch loss: " << current_loss_value << "\n";
    }
}

float hmnist::model::Trainer::evaluate(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y) const {
    // Forward pass on all samples
    Eigen::MatrixXf current_activation = X;
    for (const auto &layer_ptr: m_network->layers) {
        current_activation = layer_ptr->forward(current_activation);
    }
    Eigen::MatrixXf final_output_Yp = current_activation;
    const int num_samples = static_cast<int>(final_output_Yp.rows());

    // Convert one-hot vectors to class indices
    Eigen::VectorXi pred_indices(num_samples);
    Eigen::VectorXi true_indices(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        final_output_Yp.row(i).maxCoeff(&pred_indices(i));
        Y.row(i).maxCoeff(&true_indices(i));
    }

    const int correct_predictions = static_cast<int>((pred_indices.array() == true_indices.array()).count());
    return static_cast<float>(correct_predictions) / static_cast<float>(num_samples);
}

int hmnist::model::Trainer::predict(const Eigen::VectorXf &X) const {
    Eigen::MatrixXf activation = X.transpose();
    for (const auto &layer: m_network->layers) {
        activation = layer->forward(activation);
    }
    int pred;
    activation.row(0).maxCoeff(&pred);
    return pred;
}
