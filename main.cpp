#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>
using namespace std;

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << endl;      \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

#define CHECK_CUBLAS(call)                                                   \
    {                                                                        \
        cublasStatus_t status = call;                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            cerr << "CUBLAS error in " << __FILE__ << ":" << __LINE__   \
                      << " - " << status << endl;                       \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

enum class DataType {
    FP32,
    FP16,
    FP16_FP32_MIXED
};

double calculate_median(vector<double> &data) {
    if (data.empty()) return 0.0;
    sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0)
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    else
        return data[n / 2];
}

template <DataType dtype>
void matmul_gpu(cublasHandle_t handle, const int size, const float alpha, const float beta, const void *d_A, const void *d_B, void * d_C) {
    if constexpr (dtype == DataType::FP32) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha,
            (const float*)d_A, size,
            (const float*)d_B, size,
            &beta,
            (float*)d_C, size));
    }
    else if (dtype == DataType::FP16) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            (__half*)&alpha,
            (const half*)d_A, size,
            (const half*)d_B, size,
            (__half*)&beta,
            (half*)d_C, size));
    }
    else { 
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha,
            d_A, CUDA_R_16F, size,
            d_B, CUDA_R_16F, size,
            &beta,
            d_C, CUDA_R_32F, size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

template <DataType dtyep>
void matmul_cpu(const int size, const float alpha, const float beta, const float *h_A, const float *h_B, float *h_C) {

}

int main(int argc, char* argv[]) {
    int loop_count = 100;
    if (argc > 1) {
        loop_count = atoi(argv[1]);
    }

    vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};
    vector<DataType> data_types = {DataType::FP32, DataType::FP16, DataType::FP16_FP32_MIXED};
    ofstream outfile("../results.csv");
    outfile << "DataType,MatrixSize,Max_GFLOPS,Min_GFLOPS,Median_GFLOPS\n";

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> rand(0, 1e9);

    for (const auto& dtype : data_types) {
        string dtype_str;
        switch(dtype) {
            case DataType::FP32: dtype_str = "FP32"; break;
            case DataType::FP16: dtype_str = "FP16"; break;
            case DataType::FP16_FP32_MIXED: dtype_str = "FP16_FP32_MIXED"; break;
        }
        for (const auto& size : sizes) {
            size_t bytes_A, bytes_B, bytes_C;
            if (dtype == DataType::FP32) {
                bytes_A = bytes_B = bytes_C = size * size * sizeof(float);
            } else {
                bytes_A = bytes_B = size * size * sizeof(half);
                bytes_C = size * size * ((dtype == DataType::FP16) ? sizeof(half) : sizeof(float));
            }
            void* d_A;
            void* d_B;
            void* d_C;
            CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
            CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
            CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
            if (dtype == DataType::FP32) {
                vector<float> h_A(size * size);
                vector<float> h_B(size * size);
                for (int i = 0; i < size * size; i++) {
                    h_A[i] = rand(gen);
                    h_B[i] = rand(gen);
                }
                CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
            } else {  
                vector<half> h_A(size * size);
                vector<half> h_B(size * size);
                for (int i = 0; i < size * size; i++) {
                    h_A[i] = half(rand(gen));
                    h_B[i] = half(rand(gen));
                }
                CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
            }
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            vector<double> gflops_list;
            for(int i = 0; i < 10; ++i){
                if (dtype == DataType::FP32) {
                    matmul_gpu<DataType::FP32>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                else if (dtype == DataType::FP16) {
                    matmul_gpu<DataType::FP16>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                else { 
                    matmul_gpu<DataType::FP16_FP32_MIXED>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
            }
            for (int i = 0; i < loop_count; ++i) {
                if (dtype == DataType::FP32) {
                    matmul_gpu<DataType::FP32>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                else if (dtype == DataType::FP16) {
                    matmul_gpu<DataType::FP16>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                else { 
                    matmul_gpu<DataType::FP16_FP32_MIXED>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                // double seconds = milliseconds;
                // double gflops = (2.0 * size * size * size) / (seconds * 1e6);
                // gflops_list.push_back(gflops);
            }
            double max_gflops = *max_element(gflops_list.begin(), gflops_list.end());
            double min_gflops = *min_element(gflops_list.begin(), gflops_list.end());
            double median_gflops = calculate_median(gflops_list);
            outfile << dtype_str << "," << size << "," << max_gflops << "," << min_gflops << "," << median_gflops << "\n";
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_B));
            CHECK_CUDA(cudaFree(d_C));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            std::cout << "Completed: " << dtype_str << " with size " << size << "x" << size << endl;
        }
    }
    CHECK_CUBLAS(cublasDestroy(handle));
    outfile.close();
    std::cout << "Performance results saved to results.csv" << endl;
    return 0;
}