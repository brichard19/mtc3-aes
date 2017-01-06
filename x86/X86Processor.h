#ifndef _X86_PROCESSOR_H
#define _X86_PROCESSOR_H

#include"mtc3_common.h"
#include"threads.h"

struct CpuThreadContext {

    // In
    int threadId;
    unsigned int ciphertext[4];
    unsigned long long start;
    unsigned long long current;
    unsigned long long keysPerRun;
    
    unsigned int numThreads;
 
    // Out 
    int foundKey;
    unsigned char key[16];
    char plaintext[16];
};

class X86Processor : public MTC3Processor {

public:
    X86Processor( int numThreads, bool useNI );
    ~X86Processor();

    bool run(struct WorkUnit *workUnit);
    void getWorkUnit(struct WorkUnit *workUnit);
    bool setWorkUnit(struct WorkUnit *workUnit);
    
private:
    unsigned int _keysPerRun;

    int _numThreads;
    struct CpuThreadContext *_threadContext;
    struct WorkUnit _workUnit;
    thread_t *_threads;
    void runThreads();
};

#endif
