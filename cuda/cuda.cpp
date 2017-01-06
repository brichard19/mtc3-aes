#include<stdio.h>
#include"mtc3_cuda.h"

void cudaPrintError( cudaError_t err )
{
    printf( "CUDA error: %s\n", cudaGetErrorString( err ) );
}
