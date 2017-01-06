SRC=$(wildcard *.cpp)

x86:
	${CXX} ${SRC} -o client-x86 -D_USE_X86 ${INCLUDE} ${LIBS} ${CXXFLAGS} -lmtc3_x86 -lmtc3_common -lmtc3_platform

generic:
	${CXX} ${SRC} -o client-cpu -D_USE_GENERIC ${INCLUDE} ${LIBS} ${CXXFLAGS} -lmtc3_generic -lmtc3_common -lmtc3_platform


cuda:
	${NVCC} ${SRC} -o client-cuda -D_USE_CUDA ${NVCCFLAGS} ${INCLUDE} ${CUDA_LIBS} ${LIBS} -lmtc3_cuda -lmtc3_common -lmtc3_platform -lcuda -lcudart

clean:
	rm -f *.o
	rm -f client-cpu
