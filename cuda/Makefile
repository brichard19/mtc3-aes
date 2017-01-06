SRC=$(wildcard *.cu)
SRC+=$(wildcard *.cpp)
OBJS=$(SRC:.cu=.o)

all:    ${SRC}
	${NVCC} ${SRC} -o client-cuda ${NVCCFLAGS} ${INCLUDE} ${LIBS} -lmtc3_common -lmtc3_platform

clean:
	rm -f *.o
	rm -f client-cuda
