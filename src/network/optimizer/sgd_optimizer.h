#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include "optimizer.h"
#include <Eigen/Core>

namespace mnist {
    class SgdOptimizer : public Optimizer {
    public:
        explicit SgdOptimizer(float lr, float momentum = 0.0f);

        void update(Eigen::MatrixXf &param, const Eigen::MatrixXf &grad) override;
        void update(Eigen::RowVectorXf &param, const Eigen::RowVectorXf &grad) override;

    private:
        float learning_rate;
        float beta;
        std::unordered_map<const void*, Eigen::MatrixXf> velocity_map;
    };
} // namespace mnist

#endif //SGD_OPTIMIZER_H
