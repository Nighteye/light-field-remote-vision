/** \file cuda_coco_algorithms.h

    Easy to use entry functions for COCOLIB with some commonly used algorithms

    Copyright (C) 2013 Bastian Goldluecke,
    <first name>AT<last name>.net

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
   
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CUDA_COCO_ALGORITHMS_H
#define __CUDA_COCO_ALGORITHMS_H

#include <assert.h>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

#include "cuda_interface.h"
#include "../common/debug.h"
#include "../common/gsl_image.h"


/********************************************************
  Useful high level entry functions for COCOLIB
*********************************************************/
namespace coco {
  
  // TV-L^p denoising of a raw float array
  // supports p=1 or p=2, global and local regularizer weights
  bool coco_tv_Lp_denoising( float *data, size_t W, size_t H, float lambda, int p, size_t iter, float *weight=NULL );

  // TV-L^p denoising of a matrix
  // supports p=1 or p=2 and local regularizer weight
  bool coco_tv_Lp_denoising( gsl_matrix *M, float lambda, int p, size_t iter, float *weight=NULL );

  // TV-L^p denoising of an image
  // supports p=1 or p=2 and local regularizer weight
  bool coco_vtv_Lp_denoising( gsl_image *M, float lambda, int p, size_t iter, float *weight=NULL );


  // TV-segmentation using input matrix as data term
  // result not thresholded yet
  bool coco_tv_segmentation( gsl_matrix *M, float lambda, size_t iter, float *weight=NULL );

  // TV-segmentation using input matrix as data term
  // result not thresholded yet
  bool coco_tv_segmentation( size_t W, size_t H, float *a, float lambda, size_t iter, float *weight=NULL );


  // Multilabel algorithm (TV regularity - lifting method)
  //
  // L         : number of labels
  // lmin, lmax: label range
  // lambda    : global regularizer weight (larger = more smoothing)
  // rho       : data term
  //             layout: offset for rho( x,y,l ) is l * W*H + y*W + x
  // niter     : number of iterations (depends on problem size and data term precision)
  // result    : pointer to W*H float array, layout y*W + x
  // reg_weight: point-wise weight for regularizer (overrides lambda)
  //
  bool coco_tv_multilabel( size_t W, size_t H, size_t L,
			   float lmin, float lmax,
			   float lamba, float *rho,
			   size_t niter,
			   float *result,
			   float *reg_weight=NULL );

};

#endif
