SRC=$(wildcard *.cpp)
OBJS=$(SRC:.cpp=.o)

all:	${SRC}
	for file in ${SRC} ; do\
		${CXX} -c $$file -lpthread ${INCLUDE} ${CXXFLAGS};\
	done
	mkdir -p ${LIBDIR}
	ar rvs ${LIBDIR}/libmtc3_platform.a ${OBJS}

clean:
	rm -f *.o
