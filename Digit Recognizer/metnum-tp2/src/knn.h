#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

    unsigned int predict_row(Vector v);
private:
    unsigned int _neighbors;
    Matrix _train_data;
    Matrix _train_tags;
};
