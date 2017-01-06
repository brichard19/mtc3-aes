
#ifndef _WIN32

#include"threads.h"
#include<stdio.h>
#include<stdlib.h>

/*Function: thread_create
 *Description: Spawns and executes a new thread

 thread:	pointer to thread_t type
 routine:	pointer to a function that accepts a void pointer,
 		and returns a void pointer
arg:		pointer to void argument for the thread routine
*/
int thread_create(thread_t *thread, THREAD_FUNCTION (*routine)(VOID_PTR), void *arg)
{
	int r = 0;

	r = pthread_create(&thread->id, NULL, routine, arg);

	if( r != 0 ) {
		return -1;
	} else {
		return 0;
    }
}




/*Function:	thread_kill
 *Kills a thread
 */
int thread_kill(thread_t *thread )
{
	int r = 0;

	r = pthread_kill( thread->id, 0);

	if(r != 0)
		return -1;
	else
		return 0;
}



/*Function: thread_self
 *Returns a pointer to the thread_t structure of the calling
 *thread
 */
thread_t thread_self(void)
{
	thread_t self;

	self.id = pthread_self();

	return self;
}


void thread_wait( thread_t *threads, int n )
{
    int i = 0;

    for( i = 0; i < n; i++ ) {
        pthread_join( threads[ i ].id, NULL );
    }
}

#endif
