#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<getopt.h>
#include"mtc3_common.h"
#include"mtc3_platform.h"

#if defined(_USE_X86)
#include"X86Processor.h"
#elif defined(_USE_CUDA)
#include"CUDAProcessor.h"
#elif defined(_USE_GENERIC)
#include"GenericProcessor.h"
#endif

/**
 Prints usage information to stdout
 */
static void usage()
{
    printf("Usage: mtc3-aes [FILE] [OPTIONS]\n");

#if defined(_USE_X86)
    X86Processor::usage();
#elif defined(_USE_CUDA)
    CUDAProcessor::usage(); 
#elif defined(_USE_GENERIC)
    GenericProcessor::usage();
#endif
}


int main(int argc, char **argv)
{
    struct WorkUnit state;
    char *filename = NULL;


    MTC3Processor *p = NULL;

#if defined(_USE_X86)
    p = new X86Processor();
#elif defined(_USE_CUDA)
    p = new CUDAProcessor();
#elif defined(_USE_GENERIC)
    p = new GenericProcessor();
#endif

    if(!p->init(argc, argv)) {
        return 0;
    }
   
    if(argv[optind] == NULL) {
        usage();
    }
    filename = argv[optind];


    if(load_WorkUnit(&state, filename)) {
        printf("Error loading work unit %s\n", filename);
        return 0;
    }
    
    if(state.status == STATUS_COMPLETE) {
        return 0;
    }

    if(!p->setWorkUnit(&state)) {
        delete p;
        return 0;
    }

    mtc3_do_work(p);
    delete p;
    
    return 0;
}
