/* 18645 Spring 2019 Mini project
 * Seam Carving with Cuda
 * Author: kaiyuan1
 */

#ifndef SEAM_CARVING_CUDA_UTIL_H
#define SEAM_CARVING_CUDA_UTIL_H

#include <cassert>

#define new2D(name, xDim, yDim, type) do { \
    name = new type *[xDim];               \
    assert(name != nullptr);               \
    name[0] = new type [xDim * yDim];      \
    assert(name[0] != nullptr);            \
    for (size_t i = 1; i < xDim; i++)      \
      name[i] = name[i-1] + yDim;          \
} while (0)

struct RGBA {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;
};

#endif //SEAM_CARVING_CUDA_UTIL_H
