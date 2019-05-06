18645 mini project: Seam Curving using CUDA

loadpng.cpp & loadpng.h are copied from a light weighted PNG load/save library: https://github.com/lvandeve/lodepng

Require CUDA verion >= 9 and Nvidia Pascal or newer GPU structures (tested on GTX1080 with CUDA 10.0)

Simply type "make" to compile and generate an executabe "sc"

Usage:
  sc
    file                      : specify input png file, mandatory
    -w int                    : specify target output width, optional
    -h int                    : specify target output height, optional
    -o file                   : specify target output file, optional (default to be input_file_name.out)
    -cuda                     : whether to use cuda, optional (without this flag, will use sequential version on CPU)

Example:
  sc sample.png -w 100 -h 100 -o sample.out.png -cuda

Note:
  sequential version and cuda version generates exactly the same output (tested via shasum). The only difference is the speed.