#ifndef NETWORK_H
#define NETWORK_H
#include "layer/layer.h"
#include "loss/loss.h"
#include "optimizer/optimizer.h"
#include <fstream>
#include <vector>
#include <memory>

namespace hmnist::model {
    struct Network {
        std::vector<std::unique_ptr<layer::Layer>> layers;
        std::unique_ptr<loss::Loss> loss;
        std::unique_ptr<optimizer::Optimizer> optimizer;

        Network(std::vector<std::unique_ptr<layer::Layer>>&& L,
                std::unique_ptr<loss::Loss>&& lo,
                std::unique_ptr<optimizer::Optimizer>&& opt)
          : layers(std::move(L))
          , loss(std::move(lo))
          , optimizer(std::move(opt))
        {}
    };
}

#endif //NETWORK_H
