
all:    aes_test

aes_test:   aes_test.cpp
	${CXX} aes_test.cpp -o aes_test ${INCLUDE} -I../x86 ${CXXFLAGS} ${LIBS} -lmtc3_x86

clean:
	rm -f aes_test
