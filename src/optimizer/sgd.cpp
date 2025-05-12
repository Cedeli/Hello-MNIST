#include "sgd.h"

void hmnist::optimizer::Sgd::step(Eigen::MatrixXf &p, const Eigen::MatrixXf &g) {
    p.noalias() -= lr * g;
}

void hmnist::optimizer::Sgd::step(Eigen::RowVectorXf &p, const Eigen::RowVectorXf &g) {
    p.noalias() -= lr * g;
}
