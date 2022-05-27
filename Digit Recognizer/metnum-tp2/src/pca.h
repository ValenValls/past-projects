#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);
private:
    unsigned int _alpha_components;
    Matrix _V_Matrix;
};
