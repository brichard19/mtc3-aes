INCLUDE += -I$(shell pwd)/include
LIBDIR=$(shell pwd)/lib
LIBS +=-L$(LIBDIR)

CXX=g++
CXXFLAGS=-O3 -pthread -D_NO_AES_NI -Wall -Wextra -march=native
NVCC=nvcc
COMPUTE_CAP=30
CUDA_LIBS=-L/usr/local/cuda-5.0/lib64
NVCCFLAGS=-O2 -gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP}\" -Xcompiler "${CXXFLAGS}"

export CXX
export NVCC
export NVCCFLAGS
export CXXFLAGS
export INCLUDE
export LIBDIR
export LIBS
export COMPUTE_CAP
export CUDA_LIBS


all: mtc3_client_x86 mtc3_client_cuda

cuda: mtc3_client_cuda

x86: mtc3_client_x86

test: mtc3_common mtc3_platform mtc3_x86

clean:
	make --directory common clean
	make --directory x86 clean
	make --directory cuda clean
	make --directory platform clean
	make --directory test clean
	rm -rf ${LIBDIR}

mtc3_common:
	make --directory common

mtc3_platform:
	make --directory platform

mtc3_client_x86:    mtc3_common mtc3_platform
	make --directory x86

mtc3_client_cuda:	mtc3_common mtc3_platform
	make --directory cuda

test:
	make --directory test
