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
#include <cooperative_groups.h>

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
void searchPath(RGBA* img, unsigned h, unsigned w, unsigned dim, unsigned *trace, unsigned *diff) {
  unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= w) return;
  cooperative_groups::grid_group g = cooperative_groups::this_grid();
  for (unsigned i = 1; i < h; ++i) {
    unsigned t = j;
    unsigned d = diffRGB(img[i*dim + j], img[(i-1)*dim + j]) + diff[(i-1)*dim + j];
    if (j != 0) {
      unsigned pd = diffRGB(img[i*dim + j], img[(i-1)*dim + j-1]) + diff[(i-1)*dim + j-1];
      if (pd < d) {
        d = pd;
        t = j - 1;
      }
    }
    if (j != w-1) {
      unsigned nd = diffRGB(img[i*dim + j], img[(i-1)*dim + j+1]) + diff[(i-1)*dim + j+1];
      if (nd < d) {
        d = nd;
        t = j + 1;
      }
    }
    diff[i*dim + j] = d;
    trace[i*dim + j] = t;
    // sync among blocks
    g.sync();
  }
}

__global__ static
void removeMin(RGBA* in, RGBA* out, unsigned* from, unsigned w, unsigned dim) {
  unsigned j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= w) return;
  unsigned i = blockIdx.x;
  out[i*dim + j] = (j < from[i])?in[i*dim + j]:in[i*dim + j + 1];
}

__global__ static
void flipImg(RGBA* in, RGBA* out, unsigned w, unsigned dim) {
  unsigned j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= w) return;
  unsigned i = blockIdx.x;
  out[j*dim + i] = in[i*dim + j];
}

inline static void shrink(RGBA *img0, RGBA *img1,
                          RGBA *&inImg, RGBA *&outImg,
                          bool &use0,
                          unsigned iW,
                          unsigned tW,
                          unsigned H,
                          unsigned dim,
                          unsigned *deviceTrace, unsigned *deviceDiff, unsigned *deviceFrom,
                          unsigned **hostTrace, unsigned *hostDiff, unsigned *hostFrom) {
  for (auto W = iW; W > tW; --W, use0 = !use0) {
    if (use0) {
      inImg = img0;
      outImg = img1;
    } else {
      inImg = img1;
      outImg = img0;
    }
    unsigned n = (W + THREAD - 1) / THREAD;
    // DP
    void *args[] = {
      (void*) &inImg, (void*) &H, (void*) &W, (void*) &dim,
      (void*) &deviceTrace, (void*) &deviceDiff
    };
    // Use cudaLaunchCooperativeKernel instead of <<< ... >>> for cooperative_groups APIs
    checkCuda(cudaLaunchCooperativeKernel((void*)searchPath, n, THREAD, args));
    cudaDeviceSynchronize(); checkLastCudaError();

    // find minimum, use CPU
    checkCuda(cudaMemcpy(hostDiff, &(deviceDiff[(H-1) * dim]), W *sizeof(unsigned), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(hostTrace[0], deviceTrace, H * dim *sizeof(unsigned), cudaMemcpyDeviceToHost));
    unsigned min = std::numeric_limits<unsigned>::max();
    unsigned idx = 0;
    for (unsigned j = 0; j < W; ++j) {
      if (hostDiff[j] <= min) {
        min = hostDiff[j];
        idx = j;
      }
    }

    // retrace, use CPU
    hostFrom[H-1] = idx;
    for (auto i = H-1; i >= 1; --i) {
      hostFrom[i-1] = hostTrace[i][hostFrom[i]];
    }
    checkCuda(cudaMemcpy(deviceFrom, hostFrom, H * sizeof(unsigned), cudaMemcpyHostToDevice));

    // remove deleted, use Cuda
    n = (W + THREAD - 2) / THREAD;
    dim3 gridSize (H, n);
    removeMin <<< gridSize, THREAD >>> (inImg, outImg, deviceFrom, W-1, dim);
    cudaDeviceSynchronize(); checkLastCudaError();
  }
}

void cudaSC(RGBA **inImg, unsigned inW, unsigned inH,
            RGBA ** outImg, unsigned outW, unsigned outH) {
  // data alloc for cuda
  unsigned dim = std::max(inW, inH);
  RGBA *deviceImg0, *deviceImg1, *deviceIn, *deviceOut;
  unsigned *deviceTrace, *deviceDiff, *deviceFrom, **hostTrace, *hostDiff, *hostFrom;
  checkCuda(cudaMalloc(&deviceImg0, dim * dim * sizeof(RGBA)));
  checkCuda(cudaMalloc(&deviceImg1, dim * dim * sizeof(RGBA)));
  checkCuda(cudaMalloc(&deviceTrace, dim * dim * sizeof(unsigned)));
  checkCuda(cudaMalloc(&deviceDiff, dim * dim * sizeof(unsigned)));
  checkCuda(cudaMalloc(&deviceFrom, dim * sizeof(unsigned)));
  new2D(hostTrace, dim, dim, unsigned);
  hostDiff = new unsigned[dim];
  hostFrom = new unsigned[dim];

  // data init for cuda
  for (unsigned i = 0; i < inH; ++i)
    checkCuda(cudaMemcpy(&(deviceImg0[i*dim]), inImg[i], inW*sizeof(RGBA), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(deviceImg1, deviceImg0, dim * dim * sizeof(RGBA), cudaMemcpyDeviceToDevice));
  checkCuda(cudaMemset(deviceTrace, 0, dim * dim * sizeof(unsigned)));
  checkCuda(cudaMemset(deviceDiff, 0, dim * dim * sizeof(unsigned)));
  bool use0 = true;
  deviceIn = deviceImg0;
  deviceOut = deviceImg1;

  // shrink width on Cuda
  shrink(deviceImg0, deviceImg1, deviceIn, deviceOut,
         use0, inW, outW, inH, dim,
         deviceTrace, deviceDiff, deviceFrom, hostTrace, hostDiff, hostFrom);

  if (inH > outH) {
    // reset arrays
    checkCuda(cudaMemset(deviceTrace, 0, dim * dim * sizeof(unsigned)));
    checkCuda(cudaMemset(deviceDiff, 0, dim * dim * sizeof(unsigned)));

    // flip x and y axis
    unsigned n = (outW + THREAD - 1)/THREAD;
    flipImg <<< dim3(inH, n), THREAD >>> (deviceOut, deviceIn, outW, dim);
    use0 = !use0;

    // shrink height on Cuda
    shrink(deviceImg0, deviceImg1, deviceIn, deviceOut,
           use0, inH, outH, outW, dim,
           deviceTrace, deviceDiff, deviceFrom, hostTrace, hostDiff, hostFrom);

    // flip back
    n = (outH + THREAD - 1)/THREAD;
    flipImg <<< dim3(outW, n), THREAD >>> (deviceOut, deviceIn, outH, dim);
    deviceOut = deviceIn;
  }

  // copy to outImg
  for (unsigned i = 0; i < outH; ++i)
    checkCuda(cudaMemcpy(outImg[i], &(deviceOut[i*dim]), outW*sizeof(RGBA), cudaMemcpyDeviceToHost));

  // free
  checkCuda(cudaFree(deviceImg0));
  checkCuda(cudaFree(deviceImg1));
  checkCuda(cudaFree(deviceTrace));
  checkCuda(cudaFree(deviceDiff));
  checkCuda(cudaFree(deviceFrom));
  delete[] hostTrace[0];
  delete[] hostTrace;
  delete[] hostDiff;
  delete[] hostFrom;
}


#endif //SEAM_CARVING_CUDA_SC_CUDA_CU