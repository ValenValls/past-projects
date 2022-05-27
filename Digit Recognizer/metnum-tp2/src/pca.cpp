#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components)
{
    _alpha_components = n_components;
}

void PCA::fit(Matrix X)
{
    int n = X.rows();

    // Creamos un vector donde cada elemento i tenga el promedio de la columna i de X
    Vector u_vector(n);
    u_vector.fill((double)1 / (double)n);
    u_vector = X.transpose() * u_vector;

    Vector aux;
    // Modificamos la matriz X para que sea matriz de covarianza
    for (int i = 0; i < n; i++) {
        aux =X.row(i);
        X.row(i) = (aux - u_vector) / sqrt(n - 1);
    }
    X = X.transpose() * X;

    //Llamamos a la funcion para obtener la matriz de los primeros auto valores que diagonaliza a X y se guarda como V
    _V_Matrix = get<1>(get_first_eigenvalues(X, _alpha_components));
}


MatrixXd PCA::transform(Matrix X)
{
    //Aplicamos la transformacion guardada por el fit
    return X * _V_Matrix;
}
