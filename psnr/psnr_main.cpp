/*
 * psnr_main.cpp
 *
 *  Created on: Oct 25, 2013
 *      Author: pujadesr
 */

#include <common/gsl_image.h>

using namespace coco;

int main( int argc, char **argv )
{
  gsl_image *ref = gsl_image_load(argv[1]);
  gsl_image *candidate = gsl_image_load(argv[2]);

  double psnr = gsl_image_psnr(ref, candidate);
  double ssim = gsl_image_ssim(ref, candidate);

  printf("& %.4g & %d\n", psnr, (int)(10000*(1-ssim)/2));
}
