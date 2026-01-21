// src/prepare_data.cpp
#include "gemm_utils.hpp"

int main(int argc, char** argv) {
    // 默认矩阵大小，可以用命令行改
    int M = 1024;
    int K = 1024;
    int N = 1024;

    if (argc == 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }

    std::printf("Preparing data for GEMM: M=%d, K=%d, N=%d\n", M, K, N);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_gold(M * N);

    init_matrix(A.data(), M * K);
    init_matrix(B.data(), K * N);

    std::printf("Running CPU GEMM to generate C_gold...\n");
    cpu_gemm(A.data(), B.data(), C_gold.data(), M, K, N);

    save_matrix_bin("A.bin", A.data(), M, K);
    save_matrix_bin("B.bin", B.data(), K, N);
    save_matrix_bin("C_gold.bin", C_gold.data(), M, N);

    std::printf("Saved A.bin, B.bin, C_gold.bin\n");
    return 0;
}
