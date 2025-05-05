#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <cublas_v2.h>
#include <cublasLt.h>
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

#define TRY_CUBLAS(call)                                                    \
    {                                                                        \
        cublasStatus_t status = call;                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            cerr << "CUBLAS warning in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cublasGetErrorString(status) << endl;  \
            return status;                                                   \
        }                                                                    \
    }

enum class DataType {
    FP32,               // 32-bit floating point
    FP16_CUDA,          // 16-bit floating point with regular CUDA cores
    FP16_TENSOR,        // 16-bit floating point with tensor cores
    FP16_FP32_MIXED_CUDA, // Mixed precision: FP16 input, FP32 output, CUDA cores
    FP16_FP32_MIXED_TENSOR, // Mixed precision: FP16 input, FP32 output, tensor cores
    FP8_TENSOR,         // 8-bit floating point with tensor cores (Blackwell)
};

const char* get_dtype_name(DataType dtype) {
    switch(dtype) {
        case DataType::FP32: return "FP32_CUDA";
        case DataType::FP16_CUDA: return "FP16_CUDA";
        case DataType::FP16_TENSOR: return "FP16_TENSOR";
        case DataType::FP16_FP32_MIXED_CUDA: return "FP16_FP32_MIXED_CUDA";
        case DataType::FP16_FP32_MIXED_TENSOR: return "FP16_FP32_MIXED_TENSOR";
        case DataType::FP8_TENSOR: return "FP8_TENSOR";
        default: return "UNKNOWN";
    }
}

string cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "UNKNOWN_ERROR";
    }
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

// Helper function to align dimensions to optimize for tensor cores
inline int align_up(int v, int alignment) {
    return (v + alignment - 1) / alignment * alignment;
}

// Legacy matrix multiplication using cublasGemmEx
template <DataType dtype>
float matmul_gpu_legacy(cublasHandle_t handle, const int size, const float alpha, const float beta, 
                       const void *d_A, const void *d_B, void *d_C) {
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
    } else if (dtype == DataType::FP16_CUDA) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha_h,
            d_A, CUDA_R_16F, size,
            d_B, CUDA_R_16F, size,
            &beta_h,
            d_C, CUDA_R_16F, size,
            CUBLAS_COMPUTE_16F_PEDANTIC,
            CUBLAS_GEMM_DEFAULT
        );
    } else if (dtype == DataType::FP16_TENSOR) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha_h,
            d_A, CUDA_R_16F, size,
            d_B, CUDA_R_16F, size,
            &beta_h,
            d_C, CUDA_R_16F, size,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT
        );
    } else if (dtype == DataType::FP16_FP32_MIXED_CUDA) { 
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            size, size, size,
            &alpha,
            d_A, CUDA_R_16F, size,
            d_B, CUDA_R_16F, size,
            &beta,
            d_C, CUDA_R_32F, size,
            CUBLAS_COMPUTE_32F_PEDANTIC,
            CUBLAS_GEMM_DEFAULT
        );
    } else if (dtype == DataType::FP16_FP32_MIXED_TENSOR) { 
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

// Try to execute a cublasLt operation, return status to allow fallback
template <DataType dtype>
cublasStatus_t try_matmul_gpu_lt(cublasLtHandle_t ltHandle, const int size, const float alpha, const float beta, 
                            const void *d_A, const void *d_B, void *d_C, void* workspace, size_t workspaceSize,
                            float* milliseconds) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    half alpha_h = half(alpha);
    half beta_h = half(beta);
    
    // This will automatically be cleaned up if any step fails due to RAII pattern
    struct Cleanup {
        cublasLtMatmulDesc_t* op;
        cublasLtMatrixLayout_t* a;
        cublasLtMatrixLayout_t* b;
        cublasLtMatrixLayout_t* c;
        cublasLtMatmulPreference_t* pref;
        cudaEvent_t start, stop;
        
        ~Cleanup() {
            if (*op) cublasLtMatmulDescDestroy(*op);
            if (*a) cublasLtMatrixLayoutDestroy(*a);
            if (*b) cublasLtMatrixLayoutDestroy(*b);
            if (*c) cublasLtMatrixLayoutDestroy(*c);
            if (*pref) cublasLtMatmulPreferenceDestroy(*pref);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    } cleanup{&operationDesc, &Adesc, &Bdesc, &Cdesc, &preference, start, stop};
    
    // Determine compute type based on data type
    cudaDataType_t computeType;
    cudaDataType_t scaleType;
    cudaDataType_t inputType;
    cudaDataType_t outputType;
    
    if constexpr (dtype == DataType::FP32) {
        computeType = CUDA_R_32F;
        scaleType = CUDA_R_32F;
        inputType = CUDA_R_32F;
        outputType = CUDA_R_32F;
    } else if constexpr (dtype == DataType::FP16_CUDA) {
        computeType = CUDA_R_16F;
        scaleType = CUDA_R_16F;
        inputType = CUDA_R_16F;
        outputType = CUDA_R_16F;
    } else if constexpr (dtype == DataType::FP16_TENSOR) {
        computeType = CUDA_R_16F;
        scaleType = CUDA_R_16F;
        inputType = CUDA_R_16F;
        outputType = CUDA_R_16F;
    } else if constexpr (dtype == DataType::FP16_FP32_MIXED_CUDA || dtype == DataType::FP16_FP32_MIXED_TENSOR) {
        computeType = CUDA_R_32F;
        scaleType = CUDA_R_32F;
        inputType = CUDA_R_16F;
        outputType = CUDA_R_32F;
    } else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    
    // Set compute type based on operation type and GPU arch
    cublasComputeType_t cublasComputeType;
    
    if constexpr (dtype == DataType::FP32) {
        cublasComputeType = CUBLAS_COMPUTE_32F;
    } else if constexpr (dtype == DataType::FP16_CUDA) {
        cublasComputeType = CUBLAS_COMPUTE_16F_PEDANTIC;
    } else if constexpr (dtype == DataType::FP16_TENSOR) {
        cublasComputeType = CUBLAS_COMPUTE_16F;
    } else if constexpr (dtype == DataType::FP16_FP32_MIXED_CUDA) {
        cublasComputeType = CUBLAS_COMPUTE_32F_PEDANTIC;
    } else if constexpr (dtype == DataType::FP16_FP32_MIXED_TENSOR) {
        cublasComputeType = CUBLAS_COMPUTE_32F;
    } else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    
    // Create the operation descriptor
    TRY_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, cublasComputeType, scaleType));
    
    // Set matrix layouts with optimized strides
    int ld_a = align_up(size, 16);  // Align to 16 elements for optimal performance
    int ld_b = align_up(size, 16);
    int ld_c = align_up(size, 16);
    
    TRY_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, inputType, size, size, ld_a));
    TRY_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, inputType, size, size, ld_b));
    TRY_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, outputType, size, size, ld_c));
    
    // Create preference with workspace
    TRY_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    TRY_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    // Find the best algorithm for this operation
    int returnedAlgoCount = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult[5] = {0};
    TRY_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc,
        preference, 5, heuristicResult, &returnedAlgoCount));
    
    if (returnedAlgoCount == 0) {
        cerr << "Warning: No algorithms returned by heuristic for " << get_dtype_name(dtype) << endl;
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    
    // Start timing
    cudaEventRecord(start, 0);
    
    // Execute matrix multiplication using selected algorithm
    const void* alpha_ptr = (dtype == DataType::FP16_CUDA || dtype == DataType::FP16_TENSOR) ? 
                           static_cast<const void*>(&alpha_h) : 
                           static_cast<const void*>(&alpha);
    
    const void* beta_ptr = (dtype == DataType::FP16_CUDA || dtype == DataType::FP16_TENSOR) ? 
                          static_cast<const void*>(&beta_h) : 
                          static_cast<const void*>(&beta);
    
    TRY_CUBLAS(cublasLtMatmul(
        ltHandle, operationDesc,
        alpha_ptr,
        d_A, Adesc,
        d_B, Bdesc,
        beta_ptr,
        d_C, Cdesc,
        d_C, Cdesc,
        &heuristicResult[0].algo,
        workspace, workspaceSize,
        0));  // CUDA stream
    
    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(milliseconds, start, stop);
    
    return CUBLAS_STATUS_SUCCESS;
}

// Optimized matrix multiplication function with fallback
template <DataType dtype>
float matmul_gpu_optimized(cublasHandle_t handle, cublasLtHandle_t ltHandle, const int size, const float alpha, const float beta, 
                         const void *d_A, const void *d_B, void *d_C, void* workspace, size_t workspaceSize,
                         bool use_lt = true) {
    float milliseconds = 0.0f;
    
    if (use_lt) {
        // Try the LT implementation first
        cublasStatus_t status = try_matmul_gpu_lt<dtype>(
            ltHandle, size, alpha, beta, d_A, d_B, d_C, workspace, workspaceSize, &milliseconds);
        
        if (status == CUBLAS_STATUS_SUCCESS) {
            return milliseconds;
        }
        
        // If LT fails, fall back to legacy implementation
        cout << "cuBLASLt execution failed with status " << cublasGetErrorString(status) 
             << ", falling back to legacy implementation for " << get_dtype_name(dtype) << endl;
    }
    
    // Legacy fallback
    return matmul_gpu_legacy<dtype>(handle, size, alpha, beta, d_A, d_B, d_C);
}

// CPU implementation for validation
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
    int loop_count = 1000;
    bool use_cublas_lt = true;
    size_t workspace_size = 32 * 1024 * 1024; // 32MB default workspace
    int start_size = 7;  // 2^7 = 128
    int end_size = 12;   // 2^11 = 2048
    bool verify_results = true;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--loops" && i+1 < argc) {
            loop_count = atoi(argv[++i]);
        } else if (arg == "--no-lt") {
            use_cublas_lt = false;
        } else if (arg == "--workspace" && i+1 < argc) {
            workspace_size = atoi(argv[++i]) * 1024 * 1024; // Convert MB to bytes
        } else if (arg == "--start-size" && i+1 < argc) {
            start_size = atoi(argv[++i]);
        } else if (arg == "--end-size" && i+1 < argc) {
            end_size = atoi(argv[++i]);
        } else if (arg == "--no-verify") {
            verify_results = false;
        } else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --loops N                Number of benchmark iterations (default: 1000)" << endl;
            cout << "  --no-lt                  Disable cuBLASLt, use legacy cuBLAS only" << endl;
            cout << "  --workspace SIZE         Workspace size in MB (default: 32)" << endl;
            cout << "  --start-size POWER       Start matrix size as 2^POWER (default: 7 = 128x128)" << endl;
            cout << "  --end-size POWER         End matrix size as 2^POWER (default: 11 = 2048x2048)" << endl;
            cout << "  --no-verify              Skip result verification" << endl;
            cout << "  --help                   Display this help and exit" << endl;
            return 0;
        }
    }
    
    cout << "Configuration: " << endl;
    cout << " - Loop count: " << loop_count << endl;
    cout << " - Using cuBLASLt: " << (use_cublas_lt ? "Yes" : "No") << endl;
    cout << " - Workspace size: " << (workspace_size / (1024 * 1024)) << "MB" << endl;
    cout << " - Matrix sizes: 2^" << start_size << " to 2^" << end_size << endl;
    cout << " - Verify results: " << (verify_results ? "Yes" : "No") << endl;

    // Available data types
    vector<DataType> data_types = {
        DataType::FP32, 
        DataType::FP16_CUDA, 
        DataType::FP16_TENSOR, 
        DataType::FP16_FP32_MIXED_CUDA, 
        DataType::FP16_FP32_MIXED_TENSOR
    };
    
    // Open result file
    ofstream outfile("matmul_results.csv");
    outfile << "DataType,MatrixSize,Min_TIME_us,Max_TIME_us,Median_TIME_us,GFLOPS,Success" << endl;
    
    // CUDA setup
    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    if (use_cublas_lt) {
        CHECK_CUBLAS(cublasLtCreate(&ltHandle));
    }
    
    // Get GPU device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    cout << "Device: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    
    // Allocate workspace memory
    void* workspace = nullptr;
    if (use_cublas_lt) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> rand(-1, 1);

    // Test each data type
    for (const auto& dtype : data_types) {
        string dtype_str = get_dtype_name(dtype);
        
        // Test different matrix sizes
        for (int power = start_size; power <= end_size; power++) {
            int size = 1 << power;
            int aligned_size = align_up(size, 16);  // Align for tensor cores
            
            cout << "Testing " << dtype_str << " with size " << size << " (aligned to " << aligned_size << ")" << endl;
            
            // Calculate memory requirements
            size_t bytes_A, bytes_B, bytes_C;
            if (dtype == DataType::FP32) {
                bytes_A = bytes_B = bytes_C = aligned_size * aligned_size * sizeof(float);
            } else {
                bytes_A = bytes_B = aligned_size * aligned_size * sizeof(half);
                bytes_C = aligned_size * aligned_size * 
                         ((dtype == DataType::FP16_CUDA || dtype == DataType::FP16_TENSOR) 
                          ? sizeof(half) : sizeof(float));
            }
            
            // Allocate GPU memory
            void* d_A = nullptr;
            void* d_B = nullptr;
            void* d_C = nullptr;
            CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
            CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
            CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
            
            // Allocate and initialize host memory
            vector<float> src_A(aligned_size * aligned_size);
            vector<float> src_B(aligned_size * aligned_size);
            vector<float> src_C(aligned_size * aligned_size, 0.0f);
            vector<half> src_A_fp16(aligned_size * aligned_size);
            vector<half> src_B_fp16(aligned_size * aligned_size);
            
            vector<float> cpu_C;
            vector<float> gpu_C(aligned_size * aligned_size);
            vector<half> gpu_C_fp16(aligned_size * aligned_size);
            
            if (verify_results) {
                cpu_C.resize(aligned_size * aligned_size, 0.0f);
            }
            
            // Initialize input data
            for (int i = 0; i < aligned_size * aligned_size; i++) {
                if (i < size * size) {
                    src_A[i] = rand(gen);
                    src_B[i] = rand(gen);
                    src_A_fp16[i] = half(src_A[i]);
                    src_B_fp16[i] = half(src_B[i]);
                } else {
                    src_A[i] = 0.0f;
                    src_B[i] = 0.0f;
                    src_A_fp16[i] = half(0.0f);
                    src_B_fp16[i] = half(0.0f);
                }
            }
            
            // Copy data to GPU
            if (dtype == DataType::FP32) {
                CHECK_CUDA(cudaMemcpy(d_A, src_A.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, src_B.data(), bytes_B, cudaMemcpyHostToDevice));
            } else {
                CHECK_CUDA(cudaMemcpy(d_A, src_A_fp16.data(), bytes_A, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, src_B_fp16.data(), bytes_B, cudaMemcpyHostToDevice));
            }
            
            // Matrix multiplication parameters
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;
            
            // Warmup runs
            for (int i = 0; i < 5; i++) {
                CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));
                try {
                    switch (dtype) {
                        case DataType::FP32:
                            matmul_gpu_optimized<DataType::FP32>(handle, ltHandle, aligned_size, 
                                               alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_CUDA:
                            matmul_gpu_optimized<DataType::FP16_CUDA>(handle, ltHandle, aligned_size, 
                                                alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_TENSOR:
                            matmul_gpu_optimized<DataType::FP16_TENSOR>(handle, ltHandle, aligned_size, 
                                                alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_FP32_MIXED_CUDA:
                            matmul_gpu_optimized<DataType::FP16_FP32_MIXED_CUDA>(handle, ltHandle, aligned_size, 
                                                     alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_FP32_MIXED_TENSOR:
                            matmul_gpu_optimized<DataType::FP16_FP32_MIXED_TENSOR>(handle, ltHandle, aligned_size, 
                                                      alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        default:
                            cerr << "Unsupported data type: " << get_dtype_name(dtype) << endl;
                            break;
                    }
                } catch (const exception& e) {
                    cerr << "Exception during warmup: " << e.what() << endl;
                }
            }
            
            // CPU calculation for verification
            if (verify_results) {
                matmul_cpu(size, alpha, beta, src_A.data(), src_B.data(), cpu_C.data());
            }
            
            // Benchmark runs
            vector<double> time_list;
            bool success = true;
            
            for (int i = 0; i < loop_count; i++) {
                CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));
                
                try {
                    float milliseconds;
                    
                    switch (dtype) {
                        case DataType::FP32:
                            milliseconds = matmul_gpu_optimized<DataType::FP32>(handle, ltHandle, aligned_size, 
                                                          alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_CUDA:
                            milliseconds = matmul_gpu_optimized<DataType::FP16_CUDA>(handle, ltHandle, aligned_size, 
                                                           alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_TENSOR:
                            milliseconds = matmul_gpu_optimized<DataType::FP16_TENSOR>(handle, ltHandle, aligned_size, 
                                                           alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_FP32_MIXED_CUDA:
                            milliseconds = matmul_gpu_optimized<DataType::FP16_FP32_MIXED_CUDA>(handle, ltHandle, aligned_size, 
                                                                alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        case DataType::FP16_FP32_MIXED_TENSOR:
                            milliseconds = matmul_gpu_optimized<DataType::FP16_FP32_MIXED_TENSOR>(handle, ltHandle, aligned_size, 
                                                                 alpha, beta, d_A, d_B, d_C, workspace, workspace_size, use_cublas_lt);
                            break;
                        default:
                            cerr << "Unsupported data type: " << get_dtype_name(dtype) << endl;
                            milliseconds = -1.0f;
                            success = false;
                            break;
                    }
                    
                    if (milliseconds > 0.0f) {
                        time_list.push_back(milliseconds * 1e3);  // Convert to microseconds
                    }
                } 
                catch (const exception& e) {
                    cerr << "Exception during benchmark: " << e.what() << endl;
                    success = false;
                    break;
                }
            }
            
            // Get benchmark results
            double min_time = time_list.empty() ? -1.0 : *min_element(time_list.begin(), time_list.end());
            double max_time = time_list.empty() ? -1.0 : *max_element(time_list.begin(), time_list.end());
            double median_time = time_list.empty() ? -1.0 : calculate_median(time_list);
            
            // Calculate GFLOPS: (2*N^3 - N^2) / time
            double operations = 2.0 * size * size * size - size * size;
            double gflops = (operations / (median_time / 1e6)) / 1e9;
            
            // Save results
            outfile << dtype_str << "," << size << "," 
                    << min_time << "," << max_time << "," << median_time << "," 
                    << gflops << "," << (success ? "Yes" : "No") << endl;
            
            // Verify results if requested
            if (verify_results && success) {
                // Copy results back from GPU
                if (dtype == DataType::FP16_CUDA || dtype == DataType::FP16_TENSOR) {
                    CHECK_CUDA(cudaMemcpy(gpu_C_fp16.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
                    
                    // Convert and check for errors
                    bool has_large_error = false;
                    for (int i = 0; i < size * size; i++) {
                        float gpu_val = __half2float(gpu_C_fp16[i]);
                        if (abs(cpu_C[i] - gpu_val) > 1.0f) {
                            has_large_error = true;
                            if (i < 10) {  // Show only first 10 errors
                                cout << "Error at [" << i / size << "," << i % size << "]: "
                                     << "CPU=" << cpu_C[i] << " GPU=" << gpu_val 
                                     << " diff=" << abs(cpu_C[i] - gpu_val) << endl;
                            }
                        }
                    }
                    
                    if (!has_large_error) {
                        cout << "Verification passed" << endl;
                    }
                } 
                else {
                    CHECK_CUDA(cudaMemcpy(gpu_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
                    
                    // Check for errors
                    bool has_large_error = false;
                    for (int i = 0; i < size * size; i++) {
                        if (abs(cpu_C[i] - gpu_C[i]) > 1.0f) {
                            has_large_error = true;
                            if (i < 10) {  // Show only first 10 errors
                                cout << "Error at [" << i / size << "," << i % size << "]: "
                                     << "CPU=" << cpu_C[i] << " GPU=" << gpu_C[i]
                                     << " diff=" << abs(cpu_C[i] - gpu_C[i]) << endl;
                            }
                        }
                    }
                    
                    if (!has_large_error) {
                        cout << "Verification passed" << endl;
                    }
                }
            }
            
            // Free GPU memory
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_B));
            CHECK_CUDA(cudaFree(d_C));
            
            // Report results
            cout << "Results for " << dtype_str << " with size " << size << "x" << size 
                 << " - GFLOPS: " << gflops << endl;
            cout << "  Times (us): min=" << min_time << " max=" << max_time << " median=" << median_time << endl;
        }
    }
    
    // Clean up
    if (use_cublas_lt) {
        CHECK_CUDA(cudaFree(workspace));
        CHECK_CUBLAS(cublasLtDestroy(ltHandle));
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    outfile.close();
    cout << "Performance results saved to matmul_results.csv" << endl;
    
    return 0;
}
