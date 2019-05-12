/* 18645 Spring 2019 Mini project
 * Seam Carving with Cuda
 * Author: kaiyuan1
 */

#ifndef SEAM_CARVING_CUDA_SC_SEQ_CPP
#define SEAM_CARVING_CUDA_SC_SEQ_CPP

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <algorithm>

#include "sc_seq.h"

inline static unsigned diffRGB(RGBA p1, RGBA p2) {
  return abs(int(p1.r) - int(p2.r)) +
         abs(int(p1.g) - int(p2.g)) +
         abs(int(p1.b) - int(p2.b));
}

inline static void shrink(RGBA **img,
                          unsigned iW,
                          unsigned tW,
                          unsigned H,
                          unsigned **trace,
                          unsigned **diff,
                          unsigned *from) {
  for (auto W = iW; W > tW; --W) {
    // DP
    for (unsigned i = 1; i < H; ++i) {
      for (unsigned j = 0; j < W; ++j) {
        unsigned t = j;
        unsigned e = diffRGB(img[i][j], img[i-1][j]);
        unsigned en = 1;
        unsigned d = diff[i-1][j];
        if (j != 0) {
          e += diffRGB(img[i][j], img[i-1][j-1]);
          en ++;
          unsigned pd = diff[i-1][j-1];
          if (pd < d) {
            d = pd;
            t = j - 1;
          }
        }
        if (j != W-1) {
          e += diffRGB(img[i][j], img[i-1][j+1]);
          en ++;
          unsigned nd = diff[i-1][j+1];
          if (nd < d) {
            d = nd;
            t = j + 1;
          }
        }
        if (i != H-1) {
          e += diffRGB(img[i][j], img[i+1][j]);
          en ++;
        }
        diff[i][j] = d + e/en;
        trace[i][j] = t;
      }
    }

    // find minimum
    unsigned min = std::numeric_limits<unsigned>::max();
    unsigned idx = 0;
    for (unsigned j = 0; j < W; ++j) {
      // use "<=" to reduce time for adjusting array
      if (diff[H-1][j] <= min) {
        min = diff[H-1][j];
        idx = j;
      }
    }

    // retrace
    from[H-1] = idx;
    for (auto i = H-1; i >= 1; --i) {
      from[i-1] = trace[i][from[i]];
    }

    // remove deleted
    for (unsigned i = 0; i < H; ++i)
      for (unsigned j = from[i]; j < W-1; ++j)
        img[i][j] = img[i][j+1];
  }
}

void seqSC(RGBA **inImg, unsigned inW, unsigned inH,
           RGBA ** outImg, unsigned outW, unsigned outH) {
  // array for recording
  unsigned dim = std::max(inH, inW);
  RGBA **fImg;
  new2D(fImg, dim, dim, RGBA);
  unsigned **trace, **diff, *from;
  new2D(trace, dim, dim, unsigned);
  new2D(diff, dim, dim, unsigned);
  from = new unsigned[dim];

  // copy input to output
  memcpy(outImg[0], inImg[0], inH * inW * sizeof(RGBA));

  // shrink width
  shrink(outImg, inW, outW, inH, trace, diff, from);

  if (inH > outH) {
    // need height shrink
    // reset arrays
    memset(trace[0], 0, dim * dim * sizeof(unsigned));
    memset(diff[0], 0, dim * dim * sizeof(unsigned));

    // flip x and y axis and store into fImg
    for (unsigned i = 0; i < inH; ++i) {
      for (unsigned j = 0; j < outW; ++j) {
        fImg[j][i] = outImg[i][j];
      }
    }

    // shrink height
    shrink(fImg, inH, outH, outW, trace, diff, from);

    // restore image back into outImg
    for (unsigned i = 0; i < outH; ++i) {
      for (unsigned j = 0; j < outW; ++j) {
        outImg[i][j] = fImg[j][i];
      }
    }
  }

  // free
  delete[] trace[0];
  delete[] diff[0];
  delete[] trace;
  delete[] diff;
  delete[] from;
}

#endif //SEAM_CARVING_CUDA_SC_SEQ_CPP