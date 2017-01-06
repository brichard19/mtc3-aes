/*File: thread.s
 *Header informatino for cross-platform threading
 *for windows and POSIX
 */
#ifndef _THREADS_H
#define _THREADS_H

#ifndef _WIN32

/* Use POSIX threading */
#include<pthread.h>
#include<signal.h>

typedef void* THREAD_FUNCTION;
typedef void* VOID_PTR;

typedef struct threadtype {
	pthread_t id;
}thread_t;

typedef struct mutextype {
    pthread_mutex_t id;
}mutex_t;

#else

/* Use windows threading */

#include<windows.h>

typedef DWORD THREAD_FUNCTION;
typedef LPVOID VOID_PTR;

typedef struct threadtype
{
	HANDLE handle;
	DWORD id;		
}thread_t;

typedef struct mutextype
{
	HANDLE handle;
}mutex_t;
#endif


// API Functions

int thread_create( thread_t *thread, THREAD_FUNCTION (*routine)(VOID_PTR), void *arg );
thread_t thread_self( void );
int thread_kill( thread_t *thread );
void thread_wait( thread_t *threads, int n );

/*
int mutex_create( mutex_t *mutex );
int mutex_grab( mutex_t *mutex );
int mutex_release( mutex_t *mutex );
int mutex_destroy( mutex_t *mutex );
*/

#endif
