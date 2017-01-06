#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<getopt.h>
#include"aes.h"
#include"mtc3_common.h"
#include"mtc3_platform.h"
#include"X86Processor.h"

static void (*aes128_decrypt)(unsigned int*, unsigned int *, const unsigned int*);

static void wordArrayToByteArray(unsigned char *bytes, unsigned int *words, int count)
{
    for(int i = 0; i < count; i++) {
        bytes[(i*4)] = (unsigned char)(words[i]>>24);
        bytes[(i*4)+1] = (unsigned char)(words[i]>>16);
        bytes[(i*4)+2] = (unsigned char)(words[i]>>8);
        bytes[(i*4)+3] = (unsigned char)words[i];
    }
}

static void byteArrayToWordArray(unsigned int *words, unsigned char *bytes, int count)
{
    for(int i = 0; i < count/4; i++) {
        words[i] = (bytes[i*4]<<24) | (bytes[i*4+1]<<16) | (bytes[i*4+2]<<8) | bytes[i*4+3]; 
    }
}

THREAD_FUNCTION threadFunction(void *p)
{
    struct CpuThreadContext *ctx = (struct CpuThreadContext *)p;

    uint64_t start;
    uint64_t current;
    unsigned int key[4];
    unsigned int _align(16) plaintext[4] = {0};
    unsigned int _align(16) ciphertext[4] = {0};
    unsigned int start_mask = 0;

    start = ctx->start;
    current = ctx->current +  ctx->threadId;

    ciphertext[0] = ctx->ciphertext[0];
    ciphertext[1] = ctx->ciphertext[1];
    ciphertext[2] = ctx->ciphertext[2];
    ciphertext[3] = ctx->ciphertext[3];

    start_mask = (unsigned int)(start << 7);

    for(unsigned int i = ctx->threadId; i < ctx->keysPerRun; i+=ctx->numThreads) {
        // key bits 0 to 31
        key[3] = 0xffffffff;

        // key bits 32 to 63
        key[2] = ((unsigned int)current << 31) | 0x7fffffff;

        // key bits 64 to 95
        key[1] = (unsigned int)(current >> 1);

        // key bits 96 to 127
        key[0] = start_mask | (unsigned int)(current >> 33);

        // Perform decryption
        aes128_decrypt(ciphertext, plaintext, key);
        
        // If the first 8 characters are capital ASCII characters and the
        // last 8 characters read "CrypTool" then it might be the right plaintext
        if( plaintext[0] >= 0x41414141 && plaintext[0] <= 0x5a5a5a5a
            && plaintext[1] >= 0x41414141 && plaintext[1] <= 0x5a5a5a5a
            && plaintext[2] == 0x43727970 && plaintext[3] == 0x546f6f6c ) {

            wordArrayToByteArray((unsigned char *)ctx->key, key, 4);
            wordArrayToByteArray((unsigned char *)ctx->plaintext, plaintext, 4);
            ctx->foundKey = 1;
        }

        current += ctx->numThreads; 
    }

    ctx->current += ctx->keysPerRun;

    return NULL;
}


X86Processor::X86Processor( int threads, bool useNI )
{
    _keysPerRun = 64*1024*1024;
    _numThreads = threads;

    // Set the decryption function 
    if(useNI && aesni_supported()) {
        aes128_decrypt = aes_ni_aes128_decrypt;
    } else {
        aes128_decrypt = aes_cpu_aes128_decrypt;
    }

}

X86Processor::~X86Processor()
{
    free(_threadContext);
    free(_threads);
}

bool X86Processor::setWorkUnit(struct WorkUnit *workUnit)
{
    memcpy(&_workUnit, workUnit, sizeof(struct WorkUnit));
    
    // Allocate thread contexts 
    _threadContext = (struct CpuThreadContext *)malloc(sizeof(struct CpuThreadContext) * _numThreads);
    if(_threadContext == NULL) {
        return false;
    } 

    // Allocate thread handles
    _threads = (thread_t *)malloc(sizeof(thread_t) * _numThreads);
    if(_threads == NULL) {
        free(_threadContext);
        return false;
    } 

    // Initialize some values in the thread context 
    memset(_threadContext, 0, sizeof(struct CpuThreadContext)*_numThreads);
    for(int i = 0; i < _numThreads; i++) {
        _threadContext[i].threadId = i;
        _threadContext[i].numThreads = _numThreads;
        _threadContext[i].start = _workUnit.start;
        _threadContext[i].keysPerRun = _keysPerRun;

        byteArrayToWordArray(_threadContext[i].ciphertext, _workUnit.ciphertext, 16);          
    }

    return true;
}


bool X86Processor::run(struct WorkUnit *workUnit)
{
    bool success = true;     

    while(1) {
        runThreads();

        if(_workUnit.result) {
            break;
        }

        if(_workUnit.status == STATUS_COMPLETE) {
            break; 
        }

        if(((_workUnit.current - 1) & 0x3fffff) == 0x3fffff) {
            break; 
        }
    }

    memcpy(workUnit, &_workUnit, sizeof(struct WorkUnit));

    return success;
}

void X86Processor::getWorkUnit(struct WorkUnit *workUnit)
{
    memcpy(workUnit, &_workUnit, sizeof(struct WorkUnit));
}

void X86Processor::runThreads()
{
    if(_workUnit.current == 0x10000000000) {
        _workUnit.status = STATUS_COMPLETE;
        return;
    } else if(_workUnit.current > 0x10000000000) {
        printf("Error: Out of range\n");
        _workUnit.status = STATUS_COMPLETE;
        return;
    }
    
    for(int i = 0; i < _numThreads; i++) {
        _threadContext[i].current = _workUnit.current;
    }

    // Start the threads
    for(int i = 0; i < _numThreads; i++) {
        thread_create(&_threads[i], threadFunction, &_threadContext[i]);
    }

    // Wait for threads to finish running
    thread_wait(_threads, _numThreads);

    _workUnit.current += _keysPerRun;

    // Check if any of the threads found the key
    for(int i = 0; i < _numThreads; i++) {
        if(_threadContext[i].foundKey) {
            _workUnit.status = STATUS_COMPLETE;
            _workUnit.result = 1;
            memcpy(_workUnit.plaintext, _threadContext[i].plaintext, 16);
            memcpy(_workUnit.key, _threadContext[i].key, 16);
            return;
        } 
    }

    if(_workUnit.current == 0x10000000000) {
        _workUnit.status = STATUS_COMPLETE; 
    } else if(_workUnit.current > 0x10000000000) {
        printf("Error: Out of range\n");
        _workUnit.status = STATUS_COMPLETE;  
    }

}
