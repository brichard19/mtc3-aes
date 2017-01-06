#include<cuda.h>
#include<cuda_runtime.h>
#include"mtc3_cuda.h"

#define ROTL(x,n) ( ((x)<<(n)) | ((x)>>(32-(n))) )
#define ROTR(x,n) ( ((x)>>(n)) | ((x)<<(32-(n))) )

/**
 * Shared memory for AES sbox
 */
__device__ __shared__ unsigned int sbox1[256];
__device__ __shared__ unsigned int sbox2[256];
__device__ __shared__ unsigned int sbox3[256];
__device__ __shared__ unsigned int sbox4[256];

/**
 * We have 4 inverse sbox tables, one for each byte position. This saves
 * us from having to do any shifting when doing lookups
 */
__device__ __shared__ unsigned int invSbox1[256];
__device__ __shared__ unsigned int invSbox2[256];
__device__ __shared__ unsigned int invSbox3[256];
__device__ __shared__ unsigned int invSbox4[256];

__device__ __shared__ unsigned int invMixCols1[256];
__device__ __shared__ unsigned int invMixCols2[256];
__device__ __shared__ unsigned int invMixCols3[256];
__device__ __shared__ unsigned int invMixCols4[256];

/**
 * Shared memory for the ciphertext value
 */
__device__ __shared__ unsigned int mtc3Ciphertext[4];


/**
 * Decrypts a single block of ciphertext using AES and a given key.
 * ct: Ciphertext
 * pt: Plaintext
 * key: The key
 */
__device__ void cuda_aes128_decrypt(unsigned int ct[4], unsigned int pt[4], unsigned int key[4])
{
    unsigned int s0,s1,s2,s3;
    unsigned int tmp0,tmp1,tmp2,tmp3;
    unsigned int subkeys[44];
    unsigned int t = 0;

    subkeys[0] = key[0];
    subkeys[1] = key[1];
    subkeys[2] = key[2];
    subkeys[3] = key[3];

    // Macro for key expansion
#define SUBKEY_EXPAND(i, rcon)\
        t = subkeys[(((i)-1)*4)+3];\
        t = sbox1[(t>>16)&0xff] | sbox2[(t>>8)&0xff] | sbox3[t&0xff] | sbox4[(t>>24)&0xff];\
        t ^= (rcon);\
        t ^= subkeys[((i)-1)*4];\
        subkeys[i*4] = t;\
        t ^= subkeys[(((i)-1)*4)+1];\
        subkeys[((i)*4)+1] = t;\
        t ^= subkeys[(((i)-1)*4)+2];\
        subkeys[((i)*4)+2] = t;\
        t ^= subkeys[(((i)-1)*4)+3];\
        subkeys[((i)*4)+3] = t;\

    // Macro for AES round
#define AES_ROUND(i)\
        tmp0 = (invSbox1[(s0>>24)]) | (invSbox2[(s3>>16)&0xff]) | (invSbox3[(s2>>8)&0xff]) | (invSbox4[(s1)&0xff]);\
        tmp1 = (invSbox1[(s1>>24)]) | (invSbox2[(s0>>16)&0xff]) | (invSbox3[(s3>>8)&0xff]) | (invSbox4[(s2)&0xff]);\
        tmp2 = (invSbox1[(s2>>24)]) | (invSbox2[(s1>>16)&0xff]) | (invSbox3[(s0>>8)&0xff]) | (invSbox4[(s3)&0xff]);\
        tmp3 = (invSbox1[(s3>>24)]) | (invSbox2[(s2>>16)&0xff]) | (invSbox3[(s1>>8)&0xff]) | (invSbox4[(s0)&0xff]);\
        s0 = tmp0 ^ subkeys[(i)*4];\
        s1 = tmp1 ^ subkeys[((i)*4)+1];\
        s2 = tmp2 ^ subkeys[((i)*4)+2];\
        s3 = tmp3 ^ subkeys[((i)*4)+3];\
        s0 = invMixCols1[(s0>>24)] ^ invMixCols2[(s0>>16)&0xff] ^ invMixCols3[(s0>>8)&0xff] ^ invMixCols4[s0&0xff];\
        s1 = invMixCols1[(s1>>24)] ^ invMixCols2[(s1>>16)&0xff] ^ invMixCols3[(s1>>8)&0xff] ^ invMixCols4[s1&0xff];\
        s2 = invMixCols1[(s2>>24)] ^ invMixCols2[(s2>>16)&0xff] ^ invMixCols3[(s2>>8)&0xff] ^ invMixCols4[s2&0xff];\
        s3 = invMixCols1[(s3>>24)] ^ invMixCols2[(s3>>16)&0xff] ^ invMixCols3[(s3>>8)&0xff] ^ invMixCols4[s3&0xff];\

    // Unroll key expansion

    SUBKEY_EXPAND(1, 0x01000000)
    SUBKEY_EXPAND(2, 0x02000000)
    SUBKEY_EXPAND(3, 0x04000000)
    SUBKEY_EXPAND(4, 0x08000000)
    SUBKEY_EXPAND(5, 0x10000000)
    SUBKEY_EXPAND(6, 0x20000000)
    SUBKEY_EXPAND(7, 0x40000000)
    SUBKEY_EXPAND(8, 0x80000000)
    SUBKEY_EXPAND(9, 0x1b000000)
    SUBKEY_EXPAND(10, 0x36000000)

    // Initialize the cipher state
    s0 = ct[0];
    s1 = ct[1];
    s2 = ct[2];
    s3 = ct[3];
    
    s0 ^= subkeys[40];
    s1 ^= subkeys[41];
    s2 ^= subkeys[42];
    s3 ^= subkeys[43];

    // 9 rounds
    AES_ROUND(9)
    AES_ROUND(8)
    AES_ROUND(7)
    AES_ROUND(6)
    AES_ROUND(5)
    AES_ROUND(4)
    AES_ROUND(3)
    AES_ROUND(2)
    AES_ROUND(1)
 
    // Final InvSubBytes + InvShiftRows
    tmp0 = (invSbox1[(s0>>24)&0xff]) | (invSbox2[(s3>>16)&0xff]) | (invSbox3[(s2>>8)&0xff]) | invSbox4[s1&0xff];
    tmp1 = (invSbox1[(s1>>24)&0xff]) | (invSbox2[(s0>>16)&0xff]) | (invSbox3[(s3>>8)&0xff]) | invSbox4[s2&0xff];
    tmp2 = (invSbox1[(s2>>24)&0xff]) | (invSbox2[(s1>>16)&0xff]) | (invSbox3[(s0>>8)&0xff]) | invSbox4[s3&0xff];
    tmp3 = (invSbox1[(s3>>24)&0xff]) | (invSbox2[(s2>>16)&0xff]) | (invSbox3[(s1>>8)&0xff]) | invSbox4[s0&0xff];

    // Final AddRoundKey
    pt[0] = tmp0 ^ subkeys[0];
    pt[1] = tmp1 ^ subkeys[1];
    pt[2] = tmp2 ^ subkeys[2];
    pt[3] = tmp3 ^ subkeys[3];
}

__device__ void setupTables(struct CudaState *cudaState)
{
    // Set up lookup tables. Use one thread per element to do the copy
    if(threadIdx.x < 256) {
        sbox1[threadIdx.x] = cudaState->sbox[threadIdx.x] << 24;
        sbox2[threadIdx.x] = cudaState->sbox[threadIdx.x] << 16;
        sbox3[threadIdx.x] = cudaState->sbox[threadIdx.x] << 8;
        sbox4[threadIdx.x] = cudaState->sbox[threadIdx.x];
        
        invSbox1[threadIdx.x] = cudaState->invSbox[threadIdx.x] << 24;
        invSbox2[threadIdx.x] = cudaState->invSbox[threadIdx.x] << 16; 
        invSbox3[threadIdx.x] = cudaState->invSbox[threadIdx.x] << 8;
        invSbox4[threadIdx.x] = cudaState->invSbox[threadIdx.x];

        invMixCols1[threadIdx.x] = cudaState->invMixCols[threadIdx.x];
        invMixCols2[threadIdx.x] = ROTR(cudaState->invMixCols[threadIdx.x], 8);
        invMixCols3[threadIdx.x] = ROTR(cudaState->invMixCols[threadIdx.x], 16);
        invMixCols4[threadIdx.x] = ROTR(cudaState->invMixCols[threadIdx.x], 24);
    }
}

/**
 * The MTC3 CUDA kernel
 *
 * cudaState: Pointer to location containing current state
 * threadContext: Pointer to memory containing context for each thread
 */
__global__ void mtc3_cuda_kernel(struct CudaState *cudaState, struct CudaThreadContext *threadContext)
{
    const int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int start_mask = threadContext[threadId].start << 7;
    unsigned long long current = threadContext[threadId].current;   
    unsigned int key[4];
    unsigned int plaintext[4];
    
    if(threadIdx.x == 0) {
        mtc3Ciphertext[0] = cudaState->ciphertext[0];
        mtc3Ciphertext[1] = cudaState->ciphertext[1];
        mtc3Ciphertext[2] = cudaState->ciphertext[2];
        mtc3Ciphertext[3] = cudaState->ciphertext[3];
    }

    setupTables(cudaState);

    //Run through the keyspace
    for(int i = 0; i < cudaState->iterations; i++) {
        // key bits 0 to 31
        key[3] = 0xffffffff;

        // key bits 32 to 63
        key[2] = ((unsigned int)current << 31) | 0x7fffffff;

        // key bits 64 to 95
        key[1] = (unsigned int)(current >> 1);

        // key bits 96 to 127
        key[0] = start_mask | (unsigned int)(current >> 33);

        cuda_aes128_decrypt(mtc3Ciphertext, plaintext, key);
        
        // If the first 8 characters are capital ASCII letters and
        // the last 8 characters read "CrypTool" then it might be the right plaintext
        if(plaintext[2] == 0x43727970 && plaintext[3] == 0x546f6f6c ) {
            
            if( plaintext[0] >= 0x41414141 && plaintext[0] <= 0x5a5a5a5a
                && plaintext[1] >= 0x41414141 && plaintext[1] <= 0x5a5a5a5a) {

                cudaState->foundKey = 1;
                cudaState->current = current;
                cudaState->key[0] = key[0];
                cudaState->key[1] = key[1];
                cudaState->key[2] = key[2];
                cudaState->key[3] = key[3];

                // Convret the 4 word key into byte array
                for(int i = 0; i < 4; i++) {
                    cudaState->key[i*4] = (unsigned char)(key[i]>>24);
                    cudaState->key[(i*4)+1] = (unsigned char)(key[i]>>16);
                    cudaState->key[(i*4)+2] = (unsigned char)(key[i]>>8);
                    cudaState->key[(i*4)+3] = (unsigned char)key[i];
                }
            
                // Convert the 4 word plaintext to byte array
                for(int i = 0; i < 4; i++) {
                    cudaState->plaintext[i*4] = (unsigned char)(plaintext[i]>>24);
                    cudaState->plaintext[(i*4)+1] = (unsigned char)(plaintext[i]>>16);
                    cudaState->plaintext[(i*4)+2] = (unsigned char)(plaintext[i]>>8);
                    cudaState->plaintext[(i*4)+3] = (unsigned char)plaintext[i];
                }
            }
        }

        // Advance to next key
        current += gridDim.x * blockDim.x;
    }

    // Save current position in key space
    threadContext[threadId].current = current;

    // Update position in the state
    if( blockIdx.x == 0 && threadIdx.x == 0 ) {
        cudaState->current = current;
    }
}


/**
 * Runs the MTC3 CUDA kernel
 *
 * blocks: The number of thread blocks
 * threadsperBlock: The number of threads per block
 * cudaState: Pointer to device memory containing the current state
 * threadContext: Pointer to device memory containing the context data for each thread
 */
cudaError_t mtc3_cuda_run_kernel(int blocks, int threadsPerBlock, struct CudaState *cudaState, struct CudaThreadContext *threadContext)
{
    // Call kernel
    mtc3_cuda_kernel<<<blocks, threadsPerBlock>>>(cudaState, threadContext);

    return cudaGetLastError();
}
