/* 18645 Spring 2019 Mini project
 * Seam Carving with Cuda
 * Author: kaiyuan1
 */

#ifndef SEAM_CARVING_CUDA_SEQ_CPP
#define SEAM_CARVING_CUDA_SEQ_CPP

#include <cstdlib>
#include <cstring>
#include <deque>

#include "util.h"

inline unsigned diffRGB(RGBA p1, RGBA p2) {
  return abs(int(p1.r) - int(p2.r)) +
    abs(int(p1.g) - int(p2.g)) +
    abs(int(p1.b) - int(p2.b));
}

void seqSC(RGBA **inImg, unsigned inW, unsigned inH,
      RGBA ** outImg, unsigned outW, unsigned outH) {
  // array for recording
  auto ** trace = new unsigned* [inH];
  auto ** diff = new unsigned* [inH];

  // copy input to output
  for (unsigned i = 0; i < inH; ++i) {
    trace[i] = new unsigned [inW];
    diff[i] = new unsigned [inW];
    for (unsigned j = 0; j < inW; ++j)
      outImg[i][j] = inImg[i][j];
  }

  // shrink width
  for (auto W = inW; W > outW; --W) {
    // DP
    for (unsigned i = 1; i < inH; ++i) {
      for (unsigned j = 0; j < W; ++j) {
        unsigned t = j;
        unsigned d = diffRGB(outImg[i][j], outImg[i-1][j]) + diff[i-1][j];
        if (j != 0) {
          unsigned pd = diffRGB(outImg[i][j], outImg[i-1][j-1]) + diff[i-1][j-1];
          if (pd < d) {
            d = pd;
            t = j - 1;
          }
        }
        if (j != W-1) {
          unsigned nd = diffRGB(outImg[i][j], outImg[i-1][j+1]) + diff[i-1][j+1];
          if (nd < d) {
            d = nd;
            t = j + 1;
          }
        }
        diff[i][j] = d;
        trace[i][j] = t;
      }
    }

    // find minimum
    unsigned min = UINT32_MAX;
    unsigned idx = 0;
    for (unsigned j = 0; j < W; ++j) {
      if (diff[inH-1][j] < min) {
        min = diff[inH-1][j];
        idx = j;
      }
    }

    // retrace
    std::deque<unsigned> from;
    from.push_front(idx);
    for (auto i = inH-1; i >= 1; --i) {
      from.push_front(trace[i][from[0]]);
    }

    // remove deleted
    for (unsigned i = 0; i < inH; ++i)
      for (unsigned j = from[i]; j < W-1; ++j)
        outImg[i][j] = outImg[i][j+1];
  }

  // reset arrays
  for (unsigned i = 0; i < inH; ++i) {
    memset(trace[i], 0, sizeof(unsigned) * inW);
    memset(diff[i], 0, sizeof(unsigned) * inW);
  }

  // shrink height
  for (auto H = inH; H > outH; --H) {
    // DP
    for (unsigned j = 1; j < outW; ++j) {
      for (unsigned i = 0; i < H; ++i) {
        unsigned t = i;
        unsigned d = diffRGB(outImg[i][j], outImg[i][j-1]) + diff[i][j-1];
        if (i != 0) {
          unsigned pd = diffRGB(outImg[i][j], outImg[i-1][j-1]) + diff[i-1][j-1];
          if (pd < d) {
            d = pd;
            t = i - 1;
          }
        }
        if (i != H-1) {
          unsigned nd = diffRGB(outImg[i][j], outImg[i+1][j-1]) + diff[i+1][j-1];
          if (nd < d) {
            d = nd;
            t = i + 1;
          }
        }
        diff[i][j] = d;
        trace[i][j] = t;
      }
    }

    // find minimum
    unsigned min = UINT32_MAX;
    unsigned idx = 0;
    for (unsigned i = 0; i < H; ++i) {
      if (diff[i][outW-1] < min) {
        min = diff[i][outW-1];
        idx = i;
      }
    }

    // retrace
    std::deque<unsigned> from;
    from.push_front(idx);
    for (auto j = outW-1; j >= 1; --j) {
      from.push_front(trace[from[0]][j]);
    }

    // remove deleted
    for (unsigned j = 0; j < outW; ++j)
      for (unsigned i = from[j]; i < H-1; ++i)
        outImg[i][j] = outImg[i+1][j];
  }

  // free
  for (unsigned i = 0; i < inH; ++i) {
    delete[] trace[i];
    delete[] diff[i];
  }
  delete[] trace;
  delete[] diff;
}

#endif //SEAM_CARVING_CUDA_SEQ_CPP