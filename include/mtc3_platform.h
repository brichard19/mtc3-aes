#ifndef _MTC3_PLATFORM_H
#define _MTC3_PLATFORM_H

#ifdef WIN32
#include<Windows.h>

// Macros for compiling with Visual Studio
#define _align(x) __declspec(align(x))
#define _rotl32(x,n) _rotl((x),(n))
#define _rotr32(x,n) _rotr((x),(n))
#define _bswap32(x) _byteswap_ulong((x))
#define _bswap64(x) _byteswap_uint64((x))
#else

// Macros for compiling with GCC
#define _align(x) __attribute__((aligned((x))))
#define _rotl32(x,n) ( ((x)<<(n)) | ((x)>>(32-(n))) )
#define _rotr(x,n) ( ((x)>>(n)) | ((x)<<(32-(n))) )
#define _bswap32(x) __bswap_32((x))
#define _bswap64(x) __bswap_64((x))

#endif

unsigned int get_tick_count();
bool aesni_supported();
void cpuid(int code, int codeinfo[4]);

#endif
