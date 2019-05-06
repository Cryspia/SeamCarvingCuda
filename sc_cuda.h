//
// Created by Azure on 2019-05-06.
//

#ifndef SEAM_CARVING_CUDA_SC_CUDA_H
#define SEAM_CARVING_CUDA_SC_CUDA_H

#include "util.h"

void cudaSC(RGBA **inImg, unsigned inW, unsigned inH,
            RGBA ** outImg, unsigned outW, unsigned outH);

#endif //SEAM_CARVING_CUDA_SC_CUDA_H
