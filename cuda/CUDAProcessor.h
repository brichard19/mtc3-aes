#ifndef _CUDA_PROCESSOR_H
#define _CUDA_PROCESSOR_H

#include<cuda_runtime.h>
#include"mtc3_common.h"

struct CudaState {

    // Pointers to AES tables in device memory. They get copied from the device
    // main memory to constant memory at the start of each kernel
    unsigned int *invMixCols;
    unsigned int *sbox;
    unsigned int *invSbox;

    // in
    unsigned int ciphertext[4];
    unsigned long long start;
    unsigned long long current;
    unsigned int iterations;

    // Out
    int foundKey;
    unsigned char key[16];
    char plaintext[16];

    struct CudaState *devPtr;
};

/**
 *Holds context for a CUDA thread
 */
struct CudaThreadContext {
    unsigned long long current;
    unsigned long long start;
};


class CUDAProcessor : public MTC3Processor {

public:
    CUDAProcessor( int blocks, int threads, int iterations);

    ~CUDAProcessor();
    
    /**
     *Processes keys and updates the work unit state
     */
    virtual bool run(struct WorkUnit *workUnit);
    virtual void getWorkUnit(struct WorkUnit *workUnit);
    virtual bool setWorkUnit(struct WorkUnit *workUnit);

private:

    struct WorkUnit _workUnit;
    struct CudaState _cudaState;
    struct CudaState *_cudaStateDevicePtr;
    struct CudaThreadContext *_deviceThreadContextPtr;

    int _threads;
    int _blocks;
    int _iterations;

    cudaError_t copyTables();
    int initThreadContext();
    int initCudaState();
};

#endif
