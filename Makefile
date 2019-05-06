.KEEP_STATE:

all: sc

OPTFLAGS = -ansi -pedantic -Wall -O2
INCFLAGS = -I.
CFLAGS = $(OPTFLAGS) $(INCFLAGS)
NVCCFLAGS = $(INCFLAGS) -O2 --ptxas-options=-v -arch=sm_61
LDFLAGS = -O2

CXX = g++
XCP = -ccbin $(CXX) -Xcompiler "-std=c++11"
NVCC = nvcc

lodepng.o : lodepng.h lodepng.cpp
	$(NVCC) $(XCP) $(NVCCFLAGS) -o $@ -c lodepng.cpp

main.o : main.cpp seq.cpp cuda.cu
	$(NVCC) $(XCP) $(NVCCFLAGS) -o $@ -c main.cpp

sc: main.o lodepng.o
	$(NVCC) $(XCP) $(LDFLAGS) main.o lodepng.o -o $@

clean:
	rm -rf *.o sc