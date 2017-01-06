#ifndef _MTC3_CUDA_H
#define _MTC3_CUDA_H
#include<cuda_runtime.h>
#include"mtc3_cuda.h"
#include"CUDAProcessor.h"
#include"mtc3_common.h"

#define CUDA_DEFAULT_DEVICE 0
#define CUDA_DEFAULT_THREADS 512
#define CUDA_DEFAULT_BLOCKS 16
#define CUDA_DEFAULT_ITERATIONS 32768

extern const unsigned int host_invMixCols[];
extern const unsigned int host_sbox[];
extern const unsigned int host_invSbox[];

void cudaPrintDeviceInfo(int device);
void cudaPrintError(cudaError_t error);

cudaError_t mtc3_cuda_run_kernel(int blocks, int threadsPerBlock,
                                struct CudaState *cudaState,
                                struct CudaThreadContext *threadContext);
#endif
