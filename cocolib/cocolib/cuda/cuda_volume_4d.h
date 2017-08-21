/** \file volume_4d.h
   Data structure for 4D-volume

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

#ifndef __CUDA_VOLUME_4D_H
#define __CUDA_VOLUME_4D_H

#include <string.h>
#include <stdio.h>
#include <vector>

#include "cuda_convolutions.h"

namespace coco {

  // 4D volume data
  struct cuda_volume_4d
  {
    // Size of the grid
    size_t _X, _Y, _S, _T;
    // 4D volume internal data
    struct cuda_volume_4d_workspace *_w;
  };



  /*****************************************************************************
       4D volume creation/access
  *****************************************************************************/

  // Create empty 4D volume
  cuda_volume_4d *cuda_volume_4d_alloc( size_t X, size_t Y, size_t S, size_t T );
  // Destroy 4D volume
  bool cuda_volume_4d_free( cuda_volume_4d *V );

  // Return offset of element at specific position
  size_t cuda_volume_4d_offset( cuda_volume_4d *V, size_t x, size_t y, size_t s, size_t t );

  // Set one layer of the volume
  bool cuda_volume_4d_set_layer( cuda_volume_4d *V, size_t s, size_t t, float *data );
  // Get one layer of the volume
  bool cuda_volume_4d_get_layer( cuda_volume_4d *V, size_t s, size_t t, float *data );



  /*****************************************************************************
       Layer extraction / write back
       Used in light field analysis (yields epipolar plane images)
  *****************************************************************************/

  // Extraction for constant y/t (horizontal slice)
  bool cuda_volume_4d_extract_yt_slice( cuda_volume_4d *V, size_t y, size_t t, float *slice );

  // Write back for constant y/t (horizontal slice)
  bool cuda_volume_4d_write_back_yt_slice( cuda_volume_4d *V, size_t y, size_t t, float *slice );

  // Extraction for constant x/s (vertical slice)
  bool cuda_volume_4d_extract_xs_slice( cuda_volume_4d *V, size_t x, size_t s, float *slice );

  // Write back for constant x/s (vertical slice)
  bool cuda_volume_4d_write_back_xs_slice( cuda_volume_4d *V, size_t x, size_t s, float *slice );

  // Return image buffer at (s,t) location
  float* cuda_volume_4d_image_buffer( cuda_volume_4d *V, size_t s, size_t t );





  /*****************************************************************************
       4D volume derivative filters, computed on a single (S,T) layer.

       CURRENTLY NOT IMPLEMENTED (LFA IS USING EXTRACTION FUNCTIONS)
  *****************************************************************************/

  // central derivative X direction
  bool cuda_volume_4d_derivative_central_x( cuda_volume_4d *V, size_t s, size_t t, float *target_layer );
  // central derivative Y direction
  bool cuda_volume_4d_derivative_central_y( cuda_volume_4d *V, size_t s, size_t t, float *target_layer );
  // central derivative S direction
  bool cuda_volume_4d_derivative_central_s( cuda_volume_4d *V, size_t s, size_t t, float *target_layer );
  // central derivative T direction
  bool cuda_volume_4d_derivative_central_t( cuda_volume_4d *V, size_t s, size_t t, float *target_layer );



  /*****************************************************************************
       4D convolutions with 1D kernels, computed on a single (S,T) layer.

       CURRENTLY NOT IMPLEMENTED (LFA IS USING EXTRACTION FUNCTIONS)
  *****************************************************************************/

  // convolution X direction
  bool cuda_volume_4d_convolution_x( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer );
  // convolution Y direction
  bool cuda_volume_4d_convolution_y( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer );
  // convolution S direction
  bool cuda_volume_4d_convolution_s( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer );
  // convolution T direction
  bool cuda_volume_4d_convolution_t( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer );



  /*****************************************************************************
       Full 4D convolution with separable kernel,
       computed on a vector of (S,T) layers
       Number of target layers must be equal to (smax-smin+1) * (tmax-tmin+1)
       Kernel components must be 1D (full 4d kernel is built by convolution)

       CURRENTLY NOT IMPLEMENTED (LFA IS USING EXTRACTION FUNCTIONS)
  *****************************************************************************/

  // Gaussian convolution
  bool cuda_volume_4d_separable_convolution( cuda_volume_4d *V,
					     const std::vector<cuda_kernel*> &kernels,
					     size_t smin, size_t smax,
					     size_t tmin, size_t tmax,
					     std::vector<float*> target_layers );

}


#endif
