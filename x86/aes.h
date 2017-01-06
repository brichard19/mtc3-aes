#ifndef _AES_H
#define _AES_H

void aes_ni_aes128_decrypt(unsigned int *ciphertext, unsigned int *plaintext, const unsigned int *key);
void aes_cpu_aes128_decrypt(unsigned int *ciphertext, unsigned int *plaintext, const unsigned int *key);

#endif
