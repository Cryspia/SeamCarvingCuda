.KEEP_STATE:

all: sc

OPTFLAGS = -O2 -std=c++11
INCFLAGS = -I.
CFLAGS = $(OPTFLAGS) $(INCFLAGS)
NVCCFLAGS = $(CFLAGS) --ptxas-options=-v -rdc=true -arch=sm_61

CXX = g++
XCP = -ccbin $(CXX) -Xcompiler "-std=c++11"
NVCC = nvcc

C_SRC = main.cpp sc_seq.cpp lodepng.cpp
CU_SRC = sc_cuda.cu

C_OBJ = $(C_SRC:%.cpp=%.o)
CU_OBJ = $(CU_SRC:%.cu=%.o)

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.o : %.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

sc: $(C_OBJ) $(CU_OBJ)
	$(NVCC) $(NVCCFLAGS) $(C_OBJ) $(CU_OBJ) -o $@

clean:
	rm -rf *.o sc