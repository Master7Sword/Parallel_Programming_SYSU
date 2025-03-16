#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mkl.h> 

int main() {
    const int n = 1024; 
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> C(n * n, 0.0);

    std::srand(std::time(0));
    for (int i = 0; i < n * n; ++i) {
        A[i] = static_cast<double>(std::rand()) / RAND_MAX;
        B[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }

    std::clock_t start = std::clock();

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), n);

    std::clock_t end = std::clock();

    double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    printf("MKL matrix multiplication: %.6lf seconds\n", elapsed_time);

    return 0;
}