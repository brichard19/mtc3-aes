#ifdef _WIN32

/*File: threads_win32.c
 *Description: Code for portable Win32 threads
 */
#include"threads.h"


int thread_create( thread_t *thread, DWORD (*routine)(LPVOID), void *arg)
{
	thread_t newThread;

	newThread.handle = CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)routine,arg,0,&newThread.id);

	if( thread != NULL ) {
		*thread = newThread;
	}

	return 0;
}



int thread_kill( thread_t *thread )
{
	CloseHandle( thread->handle );

	return 0;
}

/*Wait for an array of threads to finish
 */
void thread_wait( thread_t *threads, int n )
{
	int i = 0;
	HANDLE *threadHandles = (HANDLE *)malloc( sizeof( HANDLE ) * n );

	for( i = 0; i < n; i++ ) {
		threadHandles[ i ] = threads[ i ].handle;
	}

	WaitForMultipleObjects( n, threadHandles, TRUE, INFINITE );
}

/* Grab the mutex if it exists
 */
int mutex_grab( mutex_t *mutex )
{
	if( WaitForSingleObject( mutex->handle, INFINITE ) != 0 ) {
		return -1;
	}

	return 0;
}

/* Release theb mutex
 */
int mutex_release( mutex_t *mutex )
{
	if( ReleaseMutex( mutex->handle ) ) {
		return 0;
	}

	return -1;
}

/*Create a new mutex
 */
int mutex_create( mutex_t *mutex )
{
	mutex->handle = CreateMutex( NULL, FALSE, NULL );

	if( mutex->handle == NULL ) {
		return -1;
	}

	return 0;
}


int mutex_destroy( mutex_t *mutex )
{
	if( CloseHandle( mutex->handle ) ) {
		return 0;
	}

	return -1;
}

#endif
