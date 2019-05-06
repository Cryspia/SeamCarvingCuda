/* 18645 Spring 2019 Mini project
 * Seam Carving with Cuda
 * Author: kaiyuan1
 */

#ifndef SEAM_CARVING_CUDA_SC_CUDA_CU
#define SEAM_CARVING_CUDA_SC_CUDA_CU

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>

#include "sc_cuda.h"

#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define THREAD 128

inline void checkCuda(cudaError_t e) {
  if (e != cudaSuccess) {
    err("CUDA Error: %s\n", cudaGetErrorString(e));
  }
}

inline void checkLastCudaError() {
  checkCuda(cudaGetLastError());
}

__device__ inline static
unsigned diffRGB(RGBA p1, RGBA p2) {
  return abs(int(p1.r) - int(p2.r)) +
         abs(int(p1.g) - int(p2.g)) +
         abs(int(p1.b) - int(p2.b));
}

__global__ static
void pathWidth(RGBA* img, unsigned h, unsigned w, unsigned ow, unsigned *trace, unsigned *diff) {
  unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= w) return;
  for (unsigned i = 1; i < h; ++i) {
    unsigned t = j;
    unsigned d = diffRGB(img[i*ow + j], img[(i-1)*ow + j]) + diff[(i-1)*ow + j];
    if (j != 0) {
      unsigned pd = diffRGB(img[i*ow + j], img[(i-1)*ow + j-1]) + diff[(i-1)*ow + j-1];
      if (pd < d) {
        d = pd;
        t = j - 1;
      }
    }
    if (j != w-1) {
      unsigned nd = diffRGB(img[i*ow + j], img[(i-1)*ow + j+1]) + diff[(i-1)*ow + j+1];
      if (nd < d) {
        d = nd;
        t = j + 1;
      }
    }
    diff[i*ow + j] = d;
    trace[i*ow + j] = t;
    __syncthreads();
  }
}

__global__ static
void removeWidth(RGBA* in, RGBA* out, unsigned* from, unsigned w, unsigned ow) {
  unsigned j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= w) return;
  unsigned i = blockIdx.x;
  out[i*ow + j] = (j < from[i])?in[i*ow + j]:in[i*ow + j + 1];
}

void cudaSC(RGBA **inImg, unsigned inW, unsigned inH,
            RGBA ** outImg, unsigned outW, unsigned outH) {
  // data alloc for cuda
  RGBA *deviceImg0, *deviceImg1, *deviceIn, *deviceOut;
  unsigned *deviceTrace, *deviceDiff, *deviceFrom, **hostTrace, **hostDiff, *hostFrom;
  checkCuda(cudaMalloc(&deviceImg0, inH * inW * sizeof(RGBA)));
  checkCuda(cudaMalloc(&deviceImg1, inH * inW * sizeof(RGBA)));
  checkCuda(cudaMalloc(&deviceTrace, inH * inW * sizeof(unsigned)));
  checkCuda(cudaMalloc(&deviceDiff, inH * inW * sizeof(unsigned)));
  checkCuda(cudaMalloc(&deviceFrom, std::max(inH, inW) * sizeof(unsigned)));
  new2D(hostTrace, inH, inW, unsigned);
  new2D(hostDiff, inH, inW, unsigned);
  hostFrom = new unsigned[std::max(inH, inW)];

  // data init for cuda
  checkCuda(cudaMemcpy(deviceImg0, inImg[0], inH * inW*sizeof(RGBA), cudaMemcpyHostToDevice));
  checkCuda(cudaMemset(deviceTrace, 0, inH * inW * sizeof(unsigned)));
  checkCuda(cudaMemset(deviceDiff, 0, inH * inW * sizeof(unsigned)));
  bool use0 = true;
  deviceOut = deviceImg0;

  // shrink width, use Cuda
  for (auto W = inW; W > outW; --W, use0 = !use0) {
    if (use0) {
      deviceIn = deviceImg0;
      deviceOut = deviceImg1;
    } else {
      deviceIn = deviceImg1;
      deviceOut = deviceImg0;
    }
    unsigned n = (W + THREAD - 1) / THREAD;
    // DP
    pathWidth <<< n, THREAD >>> (deviceIn, inH, W, inW, deviceTrace, deviceDiff);
    cudaDeviceSynchronize(); checkLastCudaError();

    // find minimum, use CPU
    checkCuda(cudaMemcpy(hostDiff[0], deviceDiff, inH * inW *sizeof(unsigned), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(hostTrace[0], deviceTrace, inH * inW *sizeof(unsigned), cudaMemcpyDeviceToHost));
    unsigned min = std::numeric_limits<unsigned>::max();
    unsigned idx = 0;
    for (unsigned j = 0; j < W; ++j) {
      if (hostDiff[inH-1][j] <= min) {
        min = hostDiff[inH-1][j];
        idx = j;
      }
    }

    // retrace, use CPU
    hostFrom[inH-1] = idx;
    for (auto i = inH-1; i >= 1; --i) {
      hostFrom[i-1] = hostTrace[i][hostFrom[i]];
    }
    checkCuda(cudaMemcpy(deviceFrom, hostFrom, inH *sizeof(unsigned), cudaMemcpyHostToDevice));

    // remove deleted, use Cuda
    n = (W + THREAD - 2) / THREAD;
    dim3 gridSize (inH, n);
    removeWidth <<< gridSize, THREAD >>> (deviceIn, deviceOut, deviceFrom, W-1, inW);
    cudaDeviceSynchronize(); checkLastCudaError();
  }

  // reset arrays
  checkCuda(cudaMemset(deviceTrace, 0, inH * inW * sizeof(unsigned)));
  checkCuda(cudaMemset(deviceDiff, 0, inH * inW * sizeof(unsigned)));

  // copy to outImg
  checkCuda(cudaMemcpy(outImg[0], deviceOut, inH * inW*sizeof(RGBA), cudaMemcpyDeviceToHost));

  // free
  checkCuda(cudaFree(deviceImg0));
  checkCuda(cudaFree(deviceImg1));
  checkCuda(cudaFree(deviceTrace));
  checkCuda(cudaFree(deviceDiff));
  checkCuda(cudaFree(deviceFrom));
  delete[] hostTrace[0];
  delete[] hostDiff[0];
  delete[] hostTrace;
  delete[] hostDiff;
  delete[] hostFrom;
}


#endif //SEAM_CARVING_CUDA_SC_CUDA_CU