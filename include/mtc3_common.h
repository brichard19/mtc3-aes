#ifndef _MTC3_COMMON_H
#define _MTC3_COMMON_H

#include<stdint.h>
#include<stdio.h>


#define STATUS_IN_PROGRESS 0
#define STATUS_COMPLETE 1

struct WorkUnit {

    char filename[256];

	// The current status of this work unit (new, in progress, or complete)
	int status;

	// Whether or not a result was found. 0 if not found, 1 if found
	int result;
	
	// Represents bits 0 to 35
	unsigned long long current;

	// Section of the keyspce to search
	unsigned long long start;

	// The decryption key if it was found.
	unsigned char key[16];

    // The value of the plaintext
    unsigned char plaintext[16];

	// Ciphertext to decrypt
	unsigned char ciphertext[16];
};


class MTC3Processor {

public:

/**
 * Processes keys and updates the work unit state. This method should be
 * called continually until the work unit is completed
 *
 * Returns true on success, false on failure
 */
virtual bool run(struct WorkUnit *state) = 0;
virtual void getWorkUnit(struct WorkUnit *state) = 0;
virtual bool setWorkUnit(struct WorkUnit *state) = 0;

virtual ~MTC3Processor(){}

};


int mtc3_do_work(MTC3Processor *p);
int load_WorkUnit(struct WorkUnit *s, const char *filename);
int save_WorkUnit(struct WorkUnit *s);
void printWorkUnitStatus(struct WorkUnit *state, double speed=-1, long timeRemaining=-1);

#endif
