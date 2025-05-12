#ifndef SGD_H
#define SGD_H
#include "optimizer.h"
#include <Eigen/Core>

namespace hmnist::optimizer {
    class Sgd final : public Optimizer {
    public:
        explicit Sgd(const float lr = 0.01f) : lr(lr) {
        }

        void step(Eigen::MatrixXf &p, const Eigen::MatrixXf &g) override;
        void step(Eigen::RowVectorXf &p, const Eigen::RowVectorXf &g) override;

    private:
        float lr;
    };
}

#endif //SGD_H
