#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
    //Asigno la cantidad de neighbors
    _neighbors = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y) {
    //Asigno la matriz de data y el vector columna de tags
    _train_data = X;
    _train_tags = y;

}

unsigned int KNNClassifier::predict_row(Vector v) {
    //Convierto la imagen a predecir a matriz para poder hacer la resta  con las imagenes del training
    Matrix v_matrix(_train_data.rows(), _train_data.cols());
    for (unsigned int i = 0; i < v_matrix.rows(); i++) {
        v_matrix.row(i) = v;
    }
    Matrix substraction(_train_data.rows(), _train_data.cols());
    substraction = _train_data - v_matrix;

    //Guardo la distancia euclidea de la imagen a predecir con cada imagen del training

    Vector distance(_train_data.rows());
    for (unsigned int i = 0; i < _train_data.rows(); i++) {
        distance(i) = substraction.row(i).squaredNorm();
    }

    //Ordeno las distancias ascententemente guardando su respectivo indice, para que queden primero los indices de los vecinos mas cercanos
    vector<pair<int, int>> sorted_distance(distance.size());
    for (unsigned int i = 0; i < distance.size(); i++) {
        sorted_distance[i] = make_pair(distance(i), i);
    }
    sort(sorted_distance.begin(), sorted_distance.end());

    //Cuento los tags de los vecinos mas cercanos y devuelvo el tag que más se repitio
    vector<int> tags_count(10,0);
    for (unsigned int i = 0; i < _neighbors; i++) {
        tags_count[ _train_tags(get<1>(sorted_distance[i]), 0)]++;
    }
    int max = -1;
    int max_tag = -1;
    for (unsigned int i = 0; i < 10; i++) {
        if (tags_count[i] > max) {
            max = tags_count[i];
            max_tag = i;
        }
    }
    return max_tag;
}


Vector KNNClassifier::predict(Matrix X) {
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());
    for (unsigned k = 0; k < X.rows(); k++) {
        ret(k) = predict_row(X.row(k));
    }
    return ret;
}
