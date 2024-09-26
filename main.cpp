#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>
#include <immintrin.h> 
#include <avxintrin.h>
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
                      << " - " << cublasGetErrorString(status) << endl;                       \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

enum class DataType {
    FP32,
    FP16,
    FP16_FP32_MIXED
};

string cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS : return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED : return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED : return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE : return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH : return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR : return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED : return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR : return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED : return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR : return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return nullptr;
}

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
float matmul_gpu(cublasHandle_t handle, const int size, const float alpha, const float beta, const void *d_A, const void *d_B, void * d_C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    half alpha_h = half(alpha);
    half beta_h = half(beta);
    cudaEventRecord(start, 0);
    if constexpr (dtype == DataType::FP32) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha,
            d_A, CUDA_R_32F, size,
            d_B, CUDA_R_32F, size,
            &beta,
            d_C, CUDA_R_32F, size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }
    else if (dtype == DataType::FP16) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha_h,
            d_A, CUDA_R_16F, size,
            d_B, CUDA_R_16F, size,
            &beta_h,
            d_C, CUDA_R_16F, size,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT
        ));
    }
    else { 
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha,
            d_A, CUDA_R_16F, size,
            d_B, CUDA_R_16F, size,
            &beta,
            d_C, CUDA_R_32F, size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

void matmul_cpu(const int size, const float alpha, const float beta, const float *src_A, const float *src_B, float *dst) {
    std::memset(dst, 0, sizeof(float) * size * size);
    #pragma omp parallel for
    for (int x = 0; x < size; x++) {
        for (int k = 0; k < size; k++) {
            __m256 b_val = _mm256_set1_ps(src_B[x * size + k]);
            int y = 0;
            for (; y < size - 8; y+=8) {
                __m256 a_vals = _mm256_loadu_ps(&src_A[k * size + y]);
                __m256 d_vals = _mm256_loadu_ps(&dst[x * size + y]);
                __m256 mul = _mm256_mul_ps(a_vals, b_val);
                __m256 result = _mm256_add_ps(d_vals, mul);
                _mm256_storeu_ps(&dst[x * size + y], result);
            }
            for (; y < size; y++) {
                dst[x * size + y] += src_A[k * size + y] * src_B[x * size + k];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int loop_count = 10000;
    if (argc > 1) {
        loop_count = atoi(argv[1]);
    }

    vector<DataType> data_types = {DataType::FP32, DataType::FP16, DataType::FP16_FP32_MIXED};
    ofstream outfile("../results.csv");
    outfile << "DataType,MatrixSize,Max_TIME,Min_TIME,Median_TIME\n";
    ofstream timedata("../time_lists.txt");
    cudaSetDevice(0);
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> rand(-1, 1);

    for (const auto& dtype : data_types) 
    {
        // auto dtype = DataType::FP16;
        string dtype_str;
        switch(dtype) {
            case DataType::FP32: dtype_str = "FP32"; break;
            case DataType::FP16: dtype_str = "FP16"; break;
            case DataType::FP16_FP32_MIXED: dtype_str = "FP16_FP32_MIXED"; break;
        }
        for (int size = 1 << 7; size <= 1 << 11; size <<= 1) {
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
            vector<float> src_A(size * size);
            vector<float> src_B(size * size);
            vector<float> src_C(size * size);
            vector<half> src_A_fp16(size * size);
            vector<half> src_B_fp16(size * size);

            vector<float> dst_C(size * size);
            vector<half> dst_A_fp16(size * size);
            vector<half> dst_B_fp16(size * size);
            vector<half> dst_C_fp16(size * size);
            for (int i = 0; i < size * size; i++) {
                src_A[i] = rand(gen);
                src_B[i] = rand(gen);
                src_A_fp16[i] = half(src_A[i]);
                src_B_fp16[i] = half(src_B[i]);
            }
            if (dtype == DataType::FP32) {
                CHECK_CUDA(cudaMemcpy(d_A, src_A.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, src_B.data(), bytes_B, cudaMemcpyHostToDevice));
            } else {  
                CHECK_CUDA(cudaMemcpy(d_A, src_A_fp16.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, src_B_fp16.data(), bytes_B, cudaMemcpyHostToDevice));
            }
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;
            vector<double> time_list;
            for(int i = 0; i < 10; ++i) {
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
            matmul_cpu(size, alpha, beta, src_A.data(), src_B.data(), src_C.data());
            for (int i = 0; i < loop_count; ++i) {
                double milliseconds;
                if (dtype == DataType::FP32) {
                    CHECK_CUDA(cudaMemset(d_C, 0.f, bytes_C));
                    milliseconds = matmul_gpu<DataType::FP32>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                else if (dtype == DataType::FP16) {
                    CHECK_CUDA(cudaMemset(d_C, half(0.f), bytes_C));
                    milliseconds = matmul_gpu<DataType::FP16>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                else { 
                    CHECK_CUDA(cudaMemset(d_C, 0.f, bytes_C));
                    milliseconds = matmul_gpu<DataType::FP16_FP32_MIXED>(handle, size, alpha, beta, d_A, d_B, d_C);
                }
                timedata << i << ": " << milliseconds * 1000 << "micro seconds" << endl;
                time_list.push_back(milliseconds * 1e3);
            }
            double max_time = *max_element(time_list.begin(), time_list.end());
            double min_time = *min_element(time_list.begin(), time_list.end());
            double median_time = calculate_median(time_list);
            outfile << dtype_str << "," << size << "," << max_time << "," << min_time << "," << median_time << "\n";
            if (dtype == DataType::FP16) {
                CHECK_CUDA(cudaMemcpy(dst_C_fp16.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
            } else {
                CHECK_CUDA(cudaMemcpy(dst_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
            }
            for(int i = 0; i < size * size; i++) {
                // 0.1以上の誤差はFP16FP16FP16乱数だと絶対でる。
                if (dtype == DataType::FP16) {
                    if (abs(src_C[i] - __half2float(dst_C_fp16[i])) > 1) {
                        cout << src_C[i] << ' ' << __half2float(dst_C_fp16[i]) << ' ' << abs(src_C[i] - __half2float(dst_C_fp16[i])) << endl;
                    }
                } else {
                    if (abs(src_C[i] - dst_C[i]) > 1) {
                        cout << src_C[i] << ' ' << dst_C[i] << ' ' << abs(src_C[i] - dst_C[i]) << endl;
                    }
                }
            }
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_B));
            CHECK_CUDA(cudaFree(d_C));
            std::cout << "Completed: " << dtype_str << " with size " << size << "x" << size << endl;
        }
    }
    CHECK_CUBLAS(cublasDestroy(handle));
    outfile.close();
    std::cout << "Performance results saved to results.csv" << endl;
    return 0;
}