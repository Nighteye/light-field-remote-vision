/* -*-c++-*- */
/** \file tv_deblurring.cuh
   Algorithms to solve the TV model with deblurring data term.

   Local CUDA workspace structure definition.

   Copyright (C) 2010 Bastian Goldluecke,
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

#include "../cuda/cuda_convolutions.h"

namespace coco {

  // Workspace structure, stores allocated GPU data
  struct tv_deblurring_workspace
  {
    // Deblurring variables
    // Kernels
    cuda_kernel *_k;
    cuda_kernel *_k2;
    // Lambda
    float _lambda;
    // Blurred image and convolved blurred image
    float *_f;
    float *_kf;
    // Convolution of current solution
    float *_ku;

    // CUDA block dimensions
    dim3 _dimBlock;
    dim3 _dimGrid;
  };


}
