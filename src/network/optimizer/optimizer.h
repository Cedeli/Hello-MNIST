#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <Eigen/Core>

namespace mnist {
    class Optimizer {
    public:
        virtual void update(Eigen::MatrixXf &param, const Eigen::MatrixXf &grad) = 0;

        virtual void update(Eigen::RowVectorXf &param, const Eigen::RowVectorXf &grad) = 0;

        virtual ~Optimizer() = default;
    };
} // namespace mnist

#endif //OPTIMIZER_H
