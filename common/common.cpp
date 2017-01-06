#include<stdio.h>
#include<string.h>
#include<ctype.h>
#include"common.h"
#include"mtc3_common.h"
#include"mtc3_platform.h"

#define STATUS "status:"
#define START "start:"
#define CURRENT "current:"
#define RESULT "result:"
#define KEY "key:"
#define CIPHERTEXT "ciphertext:"
#define PLAINTEXT "plaintext:"

static const char *STATUS_STRING[] = {"In progress", "Complete"};

/**
 *Removes newline and carriage return values from a string
 */
static void remove_newline(char *buffer)
{
    char *ptr = strchr(buffer, '\r');
    
    if(ptr != NULL) {
        *ptr = '\0';    
    }

    ptr = strchr(buffer,'\n');
    if(ptr != NULL) {
        *ptr = '\0';
    }
}

static void printBytes(FILE *fp, unsigned long long x)
{
    for(int i = 7; i >= 0; i--) {
        fprintf(fp, "%02x", (unsigned char)(x>>8*i));
        if(i > 0) {
            fprintf(fp, " ");
        }
    }
}

static void printBytes(FILE *fp, const unsigned char *ara, size_t count)
{
    for(size_t i = 0; i < count; i++) {
        fprintf(fp, "%02X", ara[i]);
        if(i < count - 1) {
            fprintf(fp, " ");
        }
    }
}


static void int64ToBytes(unsigned long long word, unsigned char *bytes)
{
    for(int i = 0; i < 8; i++) {
        bytes[i] = (unsigned char)(word>>(56-(i*8)));
    }
}

/**
 * Reads a 16-byte array
 */
static bool readByteArray(char *s, unsigned char *bytes)
{
    int tmp[16];
    int count = sscanf(s,
                    "%*[^:]: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
                    &tmp[0], &tmp[1], &tmp[2], &tmp[3],
                    &tmp[4], &tmp[5], &tmp[6], &tmp[7],
                    &tmp[8], &tmp[9], &tmp[10], &tmp[11],
                    &tmp[12], &tmp[13], &tmp[14], &tmp[15] );

    if(count != 16) {
        return false; 
    }

    for(int i = 0; i < 16; i++) {
        bytes[i] = (unsigned char)tmp[i];
    }

    return true;
}

/**
 * Case-insensitive string compare
 */
static int strnicmp(const char *s1, const char *s2, int len)
{
    for(int i = 0; i < len; i++) {
        if(s1[i] == '\0' && s2[i] == '\0') {
            break; 
        }

        if(s1[i] == '\0' || s2[i] == '\0') {
            return 1;
        }

        if(tolower(s1[i]) != tolower(s2[i])) {
            return 1;
        } 
    } 

    return 0;
}


/**
 *Converts the start and current values into a full 128-bit key.
 */
void get_key(unsigned long long start, unsigned long long current, unsigned long long *high, unsigned long long *low)
{
    *low = 0x7fffffffffffffff;
    *low |= current << 63;

    *high = (current & 0xffffffffff) >> 1;
    *high |= start << 43;
}

/**
 Saves current state of a work unit to a file
 */
int save_WorkUnit(struct WorkUnit *s)
{
    FILE *fp = NULL;

    fp = fopen(s->filename, "wt");

    if(fp == NULL) {
        fprintf(stderr, "Error opening %s for writing\n", s->filename);
        return -1;
    }

 
    fprintf(fp, "%s %d\n", STATUS, s->status);
    fprintf(fp, "%s %d\n", RESULT, s->result);
    fprintf(fp, "%s %llx\n", START, s->start);
    fprintf(fp, "%s %llx\n", CURRENT, s->current);

    // print ciphertext
    fprintf(fp, "%s ", CIPHERTEXT);
    printBytes(fp, s->ciphertext, 16);
    fprintf(fp, "\n");

    // If the result is found, add the key too
    if( s->result == 1 ) {
        fprintf(fp, "%s ", KEY);
        printBytes(fp, s->key, 16);
        fprintf(fp, "\n"); 

        fprintf(fp, "%s ", PLAINTEXT);
        printBytes(fp, s->plaintext, 16);
        fprintf(fp, "\n");     
    }
    

    fclose(fp);

    return 0;
}


/**
 Loads current state of a work unit from a file
 */
int load_WorkUnit(struct WorkUnit *s, const char *filename)
{
    char buffer[128] = "";
    FILE *fp = NULL;
    int success = 1;

    memset(s, 0, sizeof(struct WorkUnit));

    fp = fopen(filename, "rt");

    if(fp == NULL) {
        fprintf(stderr, "Error opening %s for reading\n", filename);
        return -1;  
    }

    // Copy file name into structure
    strcpy(s->filename, filename);

    while( fgets(buffer, 127, fp) != NULL) {
        remove_newline(buffer);

        // Skip empty lines
        if(buffer[0] == '\0') {
            continue;
        }

        if(strnicmp(buffer, STATUS, strlen(STATUS)) == 0) {
            if( sscanf(buffer, "%*[^:]: %d", &s->status) != 1) {
                fprintf(stderr, "Error reading status\n");
                success = 0;
                break;
            }
        } else if(strnicmp(buffer, START, strlen(START)) == 0) {
            if( sscanf(buffer, "%*[^:]: %llx", &s->start) != 1 ) {
                fprintf(stderr, "Error reading start\n");
                success = 0;
                break;
            }
        } else if(strnicmp(buffer, CURRENT, strlen(CURRENT)) == 0) {
            if( sscanf(buffer, "%*[^:]: %llx", &s->current) != 1 ) {
                fprintf(stderr, "Error reading current\n");
                success = 0;
                break;
            }
        } else if(strnicmp(buffer, RESULT, strlen(RESULT)) == 0) {
            if( sscanf(buffer, "%*[^:]: %d", &s->result) != 1) {
                fprintf(stderr, "Error reading result\n");
                success = 0;
                break;
            }
        } else if(strnicmp(buffer, KEY, strlen(KEY)) == 0) {
            if(!readByteArray(buffer, s->key)) {
                fprintf(stderr, "Error reading key\n");
                break;
            }
        } else if(strnicmp(buffer, CIPHERTEXT, strlen(CIPHERTEXT)) == 0) {
            if(!readByteArray(buffer, s->ciphertext)) {
                fprintf(stderr, "Error reading ciphertext\n");
                break;
            }
        } else if(strnicmp(buffer, PLAINTEXT, strlen(PLAINTEXT)) == 0) {
            if(!readByteArray(buffer, s->plaintext)) {
                fprintf(stderr, "Error reading plaintext\n");
                break;
            }
        } else {
            fprintf(stderr, "Unrecognized line in file: %s\n", buffer);
            success = 0;
            break;
        }
    }

    fclose(fp);


    if(success) {
        return 0;
    } else {
        return -1;
    }
}


void printWorkUnitStatus(struct WorkUnit *state, double speed, long timeRemaining)
{
    unsigned long long keyHigh = 0;
    unsigned long long keyLow = 0;

    // print status
    printf("Status:     %s\n", STATUS_STRING[state->status]);

    // print starting position
    get_key( state->start, 0, &keyHigh, &keyLow );
    printf("Start:      ");
    printBytes(stdout, keyHigh);
    printf(" ");
    printBytes(stdout, keyLow);
    printf("\n");

    // print ending position
    get_key( state->start + 1, 0, &keyHigh, &keyLow );
    printf("End:        ");
    printBytes(stdout, keyHigh);
    printf(" ");
    printBytes(stdout, keyLow);
    printf("\n");

    // Print current position
    get_key( state->start, state->current, &keyHigh, &keyLow );
    printf("Current:    ");
    printBytes(stdout, keyHigh);
    printf(" ");
    printBytes(stdout, keyLow);
    printf("\n");

    // Print the ciphertext
    printf("Ciphertext: ");
    printBytes(stdout, state->ciphertext, 16);
    printf("\n");
  
    if(state->result) {
        printf("Key:        ");
        printBytes(stdout, state->key, 16);
        printf("\n");
        printf("Plaintext:  ");
        printBytes(stdout, state->plaintext, 16);
        printf(" ");

        char buf[17];
        memcpy(buf, state->plaintext, 16);
        buf[16] = '\0';
        printf("(%s)\n", buf); 
    } 
  
    printf("Progress:   %4.4f%%", ((double)state->current / (double)((unsigned long long)0x10000000000))*100.0 );
   
    if(speed > 0) {
        printf("  %.2fM keys/sec",  speed); 
    }
    if(timeRemaining > 0) {
        printf("  Remaining: %02ld:%02ld:%02ld", timeRemaining/3600,(timeRemaining%3600)/60, timeRemaining%60);
    }
    printf("\n");
}


int mtc3_do_work(MTC3Processor *p)
{
    struct WorkUnit workUnit;
    p->getWorkUnit(&workUnit);
   
     
    printWorkUnitStatus(&workUnit, -1);
    while(1) {
        unsigned int t0 = get_tick_count();
        unsigned int k = workUnit.current;
        if(!p->run(&workUnit)) {
            printf("Error. Exiting..\n");
            break;
        }
        k = workUnit.current - k;

        unsigned int t1 = get_tick_count();
        double speed = ((double)k/(double)(t1 - t0))/1000.0;
        long timeRemaining = ((0x10000000000 - workUnit.current)/(long)speed)/1000000;

        printWorkUnitStatus(&workUnit, speed, timeRemaining);
        save_WorkUnit(&workUnit);

        if(workUnit.result) {
            break; 
        }
        if(workUnit.status == STATUS_COMPLETE) {
            break;
        }
    }

    return 0;
}
