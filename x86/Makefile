SRC=$(wildcard *.cpp)
OBJS=$(SRC:.cpp=.o)

all:    ${SRC}
	${CXX} ${CXXFLAGS} ${SRC} -o client-x86 ${INCLUDE} ${LIBS} -lmtc3_common -lmtc3_platform
	
clean:
	rm -f *.o
	rm -f client-x86
