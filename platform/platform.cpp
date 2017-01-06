#include"mtc3_platform.h"

#include<stdio.h>

#ifdef _WIN32
#include<Windows.h>
#include<intrin.h>
#else
#include<sys/time.h>
#endif

/**
 *Get the number of ticks. One tick is a millisecond. Useful for measuring
 *time between events.
 */
unsigned int get_tick_count()
{
#ifdef WIN32
    return GetTickCount();
#elif defined(__GNUC__)
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (tv.tv_sec*1000) + (tv.tv_usec/1000);
#endif
}

void cpuid(int code, int cpuinfo[4])
{
#ifdef WIN32
    __cpuid(cpuinfo, code);
#elif defined(__X86_64__) || defined(__i386__)
    asm volatile("cpuid":"=a"(*cpuinfo),"=b"(*(cpuinfo+1)),"=c"(*(cpuinfo+2)),"=d"(*(cpuinfo+3)):"a"(code));
#else
    (void)code;
    (void)cpuinfo;
#endif
}
