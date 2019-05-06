/* 18645 Spring 2019 Mini project
 * Seam Carving with Cuda
 * Author: kaiyuan1
 */

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "lodepng.h"
#include "util.h"
#include "sc_seq.h"
#include "sc_cuda.h"

int main(int argc, char** argv) {
  // parse input arguments
  unsigned targetW = 0;
  unsigned targetH = 0;
  bool usingCuda = false;
  std::string inputF;
  std::string outputF;
  for (int i = 1; i < argc; ++i) {
    std::string arg = std::string(argv[i]);
    if (arg == "-w") {
      i++;
      if (i == argc) {
        std::cout<<"arguments broken: please input target width"<<std::endl;
        return 1;
      }
      targetW = std::strtol(argv[i], nullptr, 10);
    }
    else if (arg == "-h") {
      i++;
      if (i == argc) {
        std::cout<<"arguments broken: please input target height"<<std::endl;
        return 1;
      }
      targetH = std::strtol(argv[i], nullptr, 10);
    }
    else if (arg == "-cuda") {
      usingCuda = true;
    }
    else if (arg == "-o") {
      i++;
      if (i == argc) {
        std::cout<<"<arguments broken: please input target output file"<<std::endl;
        return 1;
      }
      outputF = std::string(argv[i]);
    } else {
      inputF = std::string(argv[i]);
    }
  }
  if (inputF.empty()) {
    std::cout<<"please input a file for operation"<<std::endl;
  }
  if (outputF.empty()) {
    outputF = inputF + ".a";
  }

  // read input
  std::vector<unsigned char> imageVec;
  unsigned width, height;
  unsigned err = lodepng::decode(imageVec, width, height, inputF);
  if (err) {
    std::cout<<"decode image fail: "<<err<<"("<<lodepng_error_text(err)<<")"<<std::endl;
    return 1;
  }

  // prepare for seam carving
  if (targetH == 0 || targetH > height) targetH = height;
  if (targetW == 0 || targetW > width) targetW = width;
  RGBA **inputImg, **outputImg;
  new2D(inputImg, height, width, RGBA);
  new2D(outputImg, height, width, RGBA);
  unsigned offset = 0;
  for (unsigned i = 0; i < height; ++i) {
    for (unsigned j = 0; j < width; ++j) {
      inputImg[i][j].r = imageVec[offset++];
      inputImg[i][j].g = imageVec[offset++];
      inputImg[i][j].b = imageVec[offset++];
      inputImg[i][j].a = imageVec[offset++];
    }
  }
  imageVec.clear();

  // seam carving
  if (usingCuda) {
    cudaSC(inputImg, width, height, outputImg, targetW, targetH);
  } else {
    seqSC(inputImg, width, height, outputImg, targetW, targetH);
  }

  // output image
  for (unsigned i = 0; i < targetH; ++i) {
    for (unsigned j = 0; j < targetW; ++j) {
      imageVec.push_back(outputImg[i][j].r);
      imageVec.push_back(outputImg[i][j].g);
      imageVec.push_back(outputImg[i][j].b);
      imageVec.push_back(outputImg[i][j].a);
    }
  }
  delete[] inputImg[0];
  delete[] outputImg[0];
  delete[] inputImg;
  delete[] outputImg;
  err = lodepng::encode(outputF, imageVec, targetW, targetH);
  if (err) {
    std::cout << "encode image fail: " << err << "(" << lodepng_error_text(err) << ")" << std::endl;
  }
  return 0;
}