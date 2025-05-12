#include "dense_layer.h"

hmnist::layer::DenseLayer::DenseLayer(const int in_dim, const int out_dim, std::unique_ptr<activation::Activation> activation)
    : W(in_dim, out_dim), b(1, out_dim), m_activation(std::move(activation)) {
    // Kaiming He Initialization
    std::random_device dev;
    std::mt19937 rng(dev());
    std::normal_distribution<float> d(0, std::sqrt(2.0f / static_cast<float>(in_dim)));
    for (int i = 0; i < W.rows(); ++i) {
        for (int j = 0; j < W.cols(); ++j) {
            W(i, j) = d(rng);
        }
    }
    b.setZero();
}

// Forward pass
// X [batch_size x in_dim]
// Z = Z * W + b
// activation(Z)
Eigen::MatrixXf hmnist::layer::DenseLayer::forward(const Eigen::MatrixXf &input) {
    Xc = input;
    // Multiply each row by weight then add bias
    Zc = (input * W).rowwise() + b;
    return m_activation->activate(Zc);
}

// Backward pass
Eigen::MatrixXf hmnist::layer::DenseLayer::backward(const Eigen::MatrixXf &grad) {
    Eigen::MatrixXf dL_dZ;

    // TO DO: find a way to handle this differently, currently coupled with the softmax activation
    // Softmax + Cross Entropy Loss passes the gradient directly
    if (dynamic_cast<activation::Softmax*>(m_activation.get())) {
        dL_dZ = grad;
    } else {
        dL_dZ = grad.array() * m_activation->derivative(Zc).array();
    }

    // Gradient w.r.t. weights
    // Xc^T [in_dim x batch] * dL_dZ [batch x out_dim] / batch_size
    dW = (Xc.transpose() * dL_dZ) / static_cast<float>(Xc.rows());
    // Gradient w.r.t. biases
    // sum dL-DZ over batch rows, then average
    db = dL_dZ.colwise().sum() / static_cast<float>(Xc.rows());

    // Compute gradient w.r.t. input
    // dL-DZ [batch x out_dim] * W^T [out_dim x in_dim)
    return dL_dZ * W.transpose();
}

void hmnist::layer::DenseLayer::update(optimizer::Optimizer &opt) {
    opt.step(W, dW);
    opt.step(b, db);
}
