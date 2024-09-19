#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
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

int main(int argc, char* argv[]) {
    int loop_count = 100;
    if (argc > 1) {
        loop_count = atoi(argv[1]);
    }

    vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048};
    vector<DataType> data_types = {DataType::FP32, DataType::FP16, DataType::FP16_FP32_MIXED};
    ofstream outfile("../results.csv");
    outfile << "DataType,MatrixSize,Max_GFLOPS,Min_GFLOPS,Median_GFLOPS\n";

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    for (const auto& dtype : data_types) {
        string dtype_str;
        switch(dtype) {
            case DataType::FP32: dtype_str = "FP32"; break;
            case DataType::FP16: dtype_str = "FP16"; break;
            case DataType::FP16_FP32_MIXED: dtype_str = "FP16_FP32_MIXED"; break;
        }
        for (const auto& size : sizes) {
            int N = size;
            size_t bytes_A, bytes_B, bytes_C;
            if (dtype == DataType::FP32) {
                bytes_A = bytes_B = bytes_C = N * N * sizeof(float);
            } else {
                bytes_A = bytes_B = N * N * sizeof(half);
                bytes_C = N * N * ((dtype == DataType::FP16) ? sizeof(half) : sizeof(float));
            }
            void* d_A;
            void* d_B;
            void* d_C;
            CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
            CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
            CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
            if (dtype == DataType::FP32) {
                vector<float> h_A(N * N, 1.0f);
                vector<float> h_B(N * N, 1.0f);
                CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
            } else {  
                vector<half> h_A(N * N, half(1.0f));
                vector<half> h_B(N * N, half(1.0f));
                CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
            }
            float alpha = 1.0f;
            float beta = 0.0f;
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            vector<double> gflops_list;
            for(int i = 0; i < 10; ++i){
                if(dtype == DataType::FP32){
                    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        &alpha,
                        (const float*)d_A, N,
                        (const float*)d_B, N,
                        &beta,
                        (float*)d_C, N));
                }
                else if(dtype == DataType::FP16){
                    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        (__half*)&alpha,
                        (const half*)d_A, N,
                        (const half*)d_B, N,
                        (__half*)&beta,
                        (half*)d_C, N));
                }
                else { 
                    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        &alpha,
                        d_A, CUDA_R_16F, N,
                        d_B, CUDA_R_16F, N,
                        &beta,
                        d_C, CUDA_R_32F, N,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }
            for(int i = 0; i < loop_count; ++i){
                CHECK_CUDA(cudaEventRecord(start, 0));
                if(dtype == DataType::FP32){
                    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        &alpha,
                        (const float*)d_A, N,
                        (const float*)d_B, N,
                        &beta,
                        (float*)d_C, N));
                }
                else if(dtype == DataType::FP16){
                    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        (__half*)&alpha,
                        (const half*)d_A, N,
                        (const half*)d_B, N,
                        (__half*)&beta,
                        (half*)d_C, N));
                }
                else { 
                    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        &alpha,
                        d_A, CUDA_R_16F, N,
                        d_B, CUDA_R_16F, N,
                        &beta,
                        d_C, CUDA_R_32F, N,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
                CHECK_CUDA(cudaEventRecord(stop, 0));
                CHECK_CUDA(cudaEventSynchronize(stop));
                float milliseconds = 0;
                CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
                double seconds = milliseconds / 1000.0;
                double gflops = (2.0 * N * N * N) / (seconds * 1e9);
                gflops_list.push_back(gflops);
            }
            double max_gflops = *max_element(gflops_list.begin(), gflops_list.end());
            double min_gflops = *min_element(gflops_list.begin(), gflops_list.end());
            double median_gflops = calculate_median(gflops_list);
            outfile << dtype_str << "," << N << "," << max_gflops << "," << min_gflops << "," << median_gflops << "\n";
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_B));
            CHECK_CUDA(cudaFree(d_C));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            cout << "Completed: " << dtype_str << " with size " << N << "x" << N << endl;
        }
    }
    CHECK_CUBLAS(cublasDestroy(handle));
    outfile.close();
    cout << "Performance results saved to results.csv" << endl;
    return 0;
}