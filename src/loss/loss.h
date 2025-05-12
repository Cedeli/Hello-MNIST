#ifndef LOSS_H
#define LOSS_H
#include <Eigen/Core>

namespace hmnist::loss {

class Loss {
public:
    virtual ~Loss() = default;

    virtual float loss(const Eigen::MatrixXf &Yp, const Eigen::MatrixXf &Y) = 0;
    virtual Eigen::MatrixXf gradient(const Eigen::MatrixXf &Yp, const Eigen::MatrixXf &Y) = 0;
};

}

#endif //LOSS_H
