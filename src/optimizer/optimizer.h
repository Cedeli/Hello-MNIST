#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Core>

namespace hmnist::optimizer {
    class Optimizer {
    public:
        virtual ~Optimizer() = default;

        virtual void step(Eigen::MatrixXf &param, const Eigen::MatrixXf &grad) = 0;
        // TO DO: Violates DRY, should refactor into using templates?
        virtual void step(Eigen::RowVectorXf& param, const Eigen::RowVectorXf& grad) = 0;
    };
}

#endif //OPTIMIZER_H
