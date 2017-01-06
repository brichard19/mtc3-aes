#include<stdio.h>
#include<string.h>
#include"aes.h"

const unsigned int plaintext[1][4] = {
{0x41414141, 0x41414141, 0x41414141, 0x41414141}
};

const unsigned int ciphertext[1][4] = {
{0x14209CBF, 0xAA5579DE, 0x45E7384E, 0x53478CF8}
};

const unsigned int keys[1][4] = {
{0xFDE8F7A9, 0xB86C3BFF, 0x07C0D39D, 0x04605EDD}
};

int main(int argc, char **argv)
{
    for(int i = 0; i < 1; i++) {
        unsigned int pt[4];
        unsigned int ct[4];
        unsigned int k[4];
        
        memcpy((void *)ct, (void *)ciphertext[i], 16);
        memcpy((void *)k, (void *)keys[i], 16);

        printf("aes_cpu_aes128_decrypt()\n");
        aes_cpu_aes128_decrypt(ct, pt, k);

        printf("Ciphertext: %.8x %.8x %.8x %.8x\n", ct[0], ct[1], ct[2], ct[3]);
        printf("Expected: %.8x %.8x %.8x %.8x\n", plaintext[0][0], plaintext[0][1], plaintext[0][2], plaintext[0][3]);
        printf("Actual:   %.8x %.8x %.8x %.8x\n", pt[0], pt[1], pt[2], pt[3]);
        if(memcmp(pt, plaintext, 16)) {
            printf("Error\n"); 
        }
    }
}
