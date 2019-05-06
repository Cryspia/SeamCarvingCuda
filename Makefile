.KEEP_STATE:

all: sc

OPTFLAGS = -ansi -pedantic -Wall -Wextra -O2
INCFLAGS = -I.
CFLAGS = $(OPTFLAGS) $(INCFLAGS)
NVCCFLAGS = $(CFLAGS) --ptxas-options=-v -arch=sm_61

LOADPNGH = lodepng.h
LOADPNGCPP = lodepng.cpp

CC = gcc
NVCC = nvcc

sc: main.cpp seq.cpp cuda.cu $(LOADPNGH) $(LOADPNGCPP)
	$(nvcc) $(NVCCFLAGS) -c main.cpp $(LOADPNGCPP) -o $@

clean:
	rm sc