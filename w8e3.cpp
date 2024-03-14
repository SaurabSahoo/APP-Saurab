#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

const int N = 10;

void print_matrix(std::vector<std::vector<int> >& matrix) { // Corrected >> to > >
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_vector(std::vector<int>& vec) {
    for (int i = 0; i < N; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<std::vector<int> > matrix(N, std::vector<int>(N)); // Corrected >> to > >
    std::vector<int> vec(N), result(N);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = i + j;
        }
        vec[i] = i;
    }

    print_matrix(matrix);
    print_vector(vec);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        result[i] = 0;
        for (int j = 0; j < N; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }

    std::cout << "12345678910" << std::endl;
    std::cout << "Result vector y:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
