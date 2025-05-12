#ifndef DATA_LOADER_H
#define DATA_LOADER_H
#include <Eigen/Core>

namespace hmnist::data {
    struct DataSet {
        Eigen::MatrixXf images; // [Nx784]
        Eigen::MatrixXf labels; // [Nx10]
    };

    class DataLoader {
    public:
        virtual ~DataLoader() = default;

        virtual DataSet load(const std::string &image_path, const std::string &label_path) = 0;
    };
}


#endif //DATA_LOADER_H
