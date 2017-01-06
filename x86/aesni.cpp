
#ifdef _NO_AES_NI
#include<stdio.h>
#include<stdlib.h>

bool aesni_supported()
{
    return false;
}

void aes_ni_aes128_encrypt(unsigned int *plaintext, unsigned int *ciphertext, const unsigned int *key)
{
    (void)plaintext;
    (void)ciphertext;
    (void)key;

    printf("Not compiled with AES-NI\n");
    exit(1);
}

void aes_ni_aes128_decrypt(unsigned int *plaintext, unsigned int *ciphertext, const unsigned int *key)
{
    (void)plaintext;
    (void)ciphertext;
    (void)key;
    printf("Not compiled with AES-NI\n");
    exit(1);
}

#else


#include<stdio.h>
#include<wmmintrin.h>
#include"aes.h"
#include"mtc3_platform.h"

#define AES_KEY_EXPAND(key, rcon) aes_key_expand((key), _mm_aeskeygenassist_si128((key), (rcon)))


static inline __m128i sse_byte_swap(__m128i x)
{
    __m128i tmp;

    x = _mm_shufflelo_epi16((x), (0x02 << 6) | (0x03 << 4) | (0x00 << 2) | 0x01 );
    x = _mm_shufflehi_epi16((x), (0x02 << 6) | (0x03 << 4) | (0x00 << 2) | 0x01 );
    tmp = _mm_slli_epi16(x, 8);
    x = _mm_srli_epi16(x,8);

    return _mm_or_si128( x, tmp);
}

static inline __m128i aes_key_expand(__m128i key, __m128i keygened)
{
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
    return _mm_xor_si128(key, keygened);
}


/**
 Checks if AES-NI is supported on the CPU
 */
bool aesni_supported()
{
    int cpuinfo[4];

    cpuid(1, cpuinfo);

    return (cpuinfo[2] & (1 << 25)) != 0;
}

/**
 Encrypts 128 bits of data with a 128-bit key
 plaintext: Array of 16 bytes to encrypt
 ciphertext: Array of 16 bytes where the ciphertext is written
 key: Array of 16 bytes containing the key
 */
void aes_ni_aes128_encrypt(unsigned int *plaintext, unsigned int *ciphertext, const unsigned int *key)
{
    __m128i rKey[11];
    __m128i state;

    // Load the initial state   
    state = _mm_load_si128((__m128i *)plaintext);
    state = sse_byte_swap(state);

    // Load key
    rKey[0] = _mm_load_si128((__m128i *)key);
    rKey[0] = sse_byte_swap(rKey[0]);

    // Perform key expansion
    rKey[1] = AES_KEY_EXPAND(rKey[0],0x01);
    rKey[2] = AES_KEY_EXPAND(rKey[1],0x02);
    rKey[3] = AES_KEY_EXPAND(rKey[2],0x04);
    rKey[4] = AES_KEY_EXPAND(rKey[3],0x08);
    rKey[5] = AES_KEY_EXPAND(rKey[4],0x10);
    rKey[6] = AES_KEY_EXPAND(rKey[5],0x20);
    rKey[7] = AES_KEY_EXPAND(rKey[6],0x40);
    rKey[8] = AES_KEY_EXPAND(rKey[7],0x80);
    rKey[9] = AES_KEY_EXPAND(rKey[8],0x1b);
    rKey[10] = AES_KEY_EXPAND(rKey[9],0x36);

    // XOR the key with the state
    state = _mm_xor_si128(state, rKey[0]);

    // Rounds 1 - 9
    state = _mm_aesenc_si128(state, rKey[1]);
    state = _mm_aesenc_si128(state, rKey[2]);
    state = _mm_aesenc_si128(state, rKey[3]);
    state = _mm_aesenc_si128(state, rKey[4]);
    state = _mm_aesenc_si128(state, rKey[5]);
    state = _mm_aesenc_si128(state, rKey[6]);
    state = _mm_aesenc_si128(state, rKey[7]);
    state = _mm_aesenc_si128(state, rKey[8]);
    state = _mm_aesenc_si128(state, rKey[9]);

    // Round 10
    state = _mm_aesenclast_si128(state, rKey[10]);
    state = sse_byte_swap(state);

    // Store the ciphertext
    _mm_store_si128((__m128i *)ciphertext, state);
}

/**
 Decrypts 128 bits of data with a 128-bit key
 ciphertext: Array of 16 bytes to decrypt
 plaintext: The decrypted data is written here
 key: Array of 16 bytes containing the key
 */
 void aes_ni_aes128_decrypt(unsigned int *ciphertext, unsigned int *plaintext, const unsigned int *key)
{
    __m128i rKey[11];
    __m128i state;

    // Load key, swap endian
    rKey[0] = _mm_load_si128((__m128i *)key);
    rKey[0] = sse_byte_swap(rKey[0]);

    // Perform key expansion
    rKey[1] = AES_KEY_EXPAND(rKey[0],0x01);
    rKey[2] = AES_KEY_EXPAND(rKey[1],0x02);
    rKey[3] = AES_KEY_EXPAND(rKey[2],0x04);
    rKey[4] = AES_KEY_EXPAND(rKey[3],0x08);
    rKey[5] = AES_KEY_EXPAND(rKey[4],0x10);
    rKey[6] = AES_KEY_EXPAND(rKey[5],0x20);
    rKey[7] = AES_KEY_EXPAND(rKey[6],0x40);
    rKey[8] = AES_KEY_EXPAND(rKey[7],0x80);
    rKey[9] = AES_KEY_EXPAND(rKey[8],0x1b);
    rKey[10] = AES_KEY_EXPAND(rKey[9],0x36);

    // Load ciphertxt
    state = _mm_load_si128((__m128i *)ciphertext);
    state = sse_byte_swap(state);

    // Add key
    state = _mm_xor_si128(state, rKey[10]);

    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[9]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[8]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[7]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[6]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[5]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[4]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[3]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[2]));
    state = _mm_aesdec_si128(state, _mm_aesimc_si128(rKey[1]));
    state = _mm_aesdeclast_si128(state, rKey[0]);

    // Store plaintext
    state = sse_byte_swap(state);
    _mm_store_si128((__m128i *)plaintext, state);
}


#endif
