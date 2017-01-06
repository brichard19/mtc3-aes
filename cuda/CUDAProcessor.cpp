#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<string.h>
#include<getopt.h>
#include"mtc3_platform.h"
#include"mtc3_common.h"
#include"mtc3_cuda.h"
#include"CUDAProcessor.h"


/**
 * Deletes contents of a CudaState structure
 */
static void cleanup_state(struct CudaState *cudaState)
{
    cudaFree( cudaState->sbox );
    cudaFree( cudaState->invSbox );
    cudaFree( cudaState->invMixCols );

    cudaState->sbox = NULL;
    cudaState->invSbox = NULL;
    cudaState->invMixCols = NULL;
}

CUDAProcessor::CUDAProcessor( int blocks, int threads, int iterations )
{
    _threads = threads;
    _blocks = blocks;
    _iterations = iterations;
}

CUDAProcessor::~CUDAProcessor()
{
    cleanup_state(&_cudaState);
}

bool CUDAProcessor::setWorkUnit(struct WorkUnit *workUnit)
{
    memcpy( &_workUnit, workUnit, sizeof(struct WorkUnit) );

    // Determine how many times we need to call the kernel: 2^40 divided by work done per kernel call
    unsigned long long work_per_call = _threads * _blocks * _iterations;
    unsigned long long diff = ((unsigned long long)1 << 40) - _workUnit.current;
    
    // Round the current position so that the amount of keys to check is a multiple
    // of the number of keys checked per kernel call. 
    if( _workUnit.current < work_per_call ) {
        _workUnit.current = 0;
    } else if(diff % work_per_call != 0) {
        _workUnit.current = ((unsigned long long)1 << 40) - (diff + ((diff + work_per_call) % work_per_call));
    }

    if( initCudaState() ) {
        return false;
    }

    if( initThreadContext() ) {
        return false;
    }
    
    return true;
}

void CUDAProcessor::getWorkUnit(struct WorkUnit *workUnit)
{
    memcpy(workUnit, &_workUnit, sizeof(struct WorkUnit));
}

bool CUDAProcessor::run(struct WorkUnit *workUnitPtr)
{
    bool success = true;

    while(1) {

        // Run the kernel
        cudaError_t cudaError = mtc3_cuda_run_kernel(_blocks, _threads, _cudaStateDevicePtr, _deviceThreadContextPtr);

        // Check for error
        if(cudaError != cudaSuccess) {
            cudaPrintError(cudaError);
            success = false;
            break;
        }

        // Get state
        cudaError = cudaMemcpy(&_cudaState, _cudaStateDevicePtr, sizeof(struct CudaState), cudaMemcpyDeviceToHost);
        if(cudaError != cudaSuccess) {
            cudaPrintError(cudaError);
            success = false;
            break;
        }

        // Check for result
        if(_cudaState.foundKey) {
        
            _workUnit.result = 1;  
            memcpy(_workUnit.key, _cudaState.key, 16);
            memcpy(_workUnit.plaintext, _cudaState.plaintext, 16);
    
            _workUnit.status = STATUS_COMPLETE;
            _workUnit.current = _cudaState.current;

            break;
        }

        // Check if finished
        if(_cudaState.current == 0x10000000000) {
            _workUnit.status = STATUS_COMPLETE;
            _workUnit.current = _cudaState.current;
            break; 
        } else if(_cudaState.current > 0x10000000000 ) {
            printf("Error: Out of range. current = %llx\n", _cudaState.current);
            _workUnit.status = STATUS_COMPLETE;
            success = false;
            break;
        } else if(((_cudaState.current-1) & 0x3ffffff) == 0x3ffffff) {
            _workUnit.status = STATUS_IN_PROGRESS;
            _workUnit.current = _cudaState.current;
            _workUnit.result = 0;
            break;
        }
    }

    memcpy(workUnitPtr, &_workUnit, sizeof(struct WorkUnit));

    return success; 
}


int CUDAProcessor::initThreadContext()
{
    cudaError_t cudaError = cudaSuccess;
    struct CudaThreadContext *hostThreadContext;
    int numThreads = _threads * _blocks;
    int err = 0;

    // Allocate context on host
    hostThreadContext = new CudaThreadContext[ numThreads ];

    if( hostThreadContext == NULL ) {
        err = 1;
        goto end;
    }

    // Initialize context on host
    for(int i = 0; i < numThreads; i++) {
        hostThreadContext[i].start = _workUnit.start;
        hostThreadContext[i].current = _workUnit.current + i;
    }

    // Allocate memory on device
    cudaError = cudaMalloc(&_deviceThreadContextPtr, sizeof(struct CudaThreadContext) * numThreads);

    if(cudaError != cudaSuccess) {
        err = 1;
        goto end;
    }

    // Initialize context on device
    cudaError = cudaMemcpy( _deviceThreadContextPtr, hostThreadContext,
                           numThreads * sizeof(struct CudaThreadContext),
                           cudaMemcpyHostToDevice );

    if( cudaError != cudaSuccess ) {
        err = 1;
        goto end;
    }


end:
    // No longer need thread context on device side
    delete[] hostThreadContext;

     if(err) {
        cudaFree(_deviceThreadContextPtr);
    }

    return err;

}

int CUDAProcessor::initCudaState()
{
    int err = 0;
    cudaError_t cudaError = cudaSuccess;

    memset(&_cudaState, 0, sizeof(struct CudaState));

   
    // Convert the ciphertext from a byte array to an array of words 
    for(int i = 0; i < 4; i++) {
        _cudaState.ciphertext[i] = (_workUnit.ciphertext[i*4] << 24)
                                  | (_workUnit.ciphertext[(i*4)+1]<<16)
                                  | (_workUnit.ciphertext[(i*4)+2]<<8)
                                  | _workUnit.ciphertext[(i*4)+3];
    }

    // Set starting position
    _cudaState.start = _workUnit.start;

    // Set number of GPU iterations
    _cudaState.iterations = _iterations;

    // Allocate and copy tables to the device
    cudaError = copyTables();

    if(cudaError != cudaSuccess) {
        cudaPrintError(cudaError);
        err = 1;
        goto end;
    }


    // Allocate memory on device to hold state info
    cudaError = cudaMalloc(&_cudaStateDevicePtr, sizeof(struct CudaState));
    if(cudaError != cudaSuccess) {
        cudaPrintError(cudaError);
        err = 1;
        goto end;
    }

    // Copy state to the device
    cudaError = cudaMemcpy(_cudaStateDevicePtr, &_cudaState, sizeof(struct CudaState), cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        cudaPrintError(cudaError);
        err = 1;
        goto end;
    }


end:
    if(err) {
        cleanup_state(&_cudaState);
    }

    return err;

}

cudaError_t CUDAProcessor::copyTables()
{
    cudaError_t cudaError = cudaSuccess;

    _cudaState.invMixCols = NULL;
    _cudaState.invSbox = NULL;
    _cudaState.sbox = NULL;

    // Allocate nverse mix columns table
    cudaError = cudaMalloc(&_cudaState.invMixCols, 1024);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    // Allocate s-box table
    cudaError = cudaMalloc(&_cudaState.sbox, 1024);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    // Allocate inverse s-box table
    cudaError = cudaMalloc(&_cudaState.invSbox, 1024);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    // Copy inverse mix columns table
    cudaError = cudaMemcpy(_cudaState.invMixCols, host_invMixCols, 1024, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    // Copy s-box table
    cudaError = cudaMemcpy(_cudaState.sbox, host_sbox, 1024, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    // Copy inverse s-box table
    cudaError = cudaMemcpy(_cudaState.invSbox, host_invSbox, 1024, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

end:
    if( cudaError != cudaSuccess ) {
        cudaFree(_cudaState.invMixCols);
        cudaFree(_cudaState.invSbox);
        cudaFree(_cudaState.sbox);
    }

    return cudaError;
}

