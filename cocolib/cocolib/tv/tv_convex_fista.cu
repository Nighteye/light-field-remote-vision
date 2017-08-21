/* -*-c++-*- */
/** \file tv_l2_fgp.cu

   3. Fast gradient projection
   Implements FGP from Beck/Teboulle 2009,
   "Fast Gradient-based algorithms for constrained total variation
   image denoising and deblurring problems."

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

#include <iostream>
#include "tv_convex.h"
#include "tv_convex.cuh"
#include "../tv/tv_l2.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"


__global__ void compute_fista_rof_function( int W, int H,
					    float L,
					    float *y, float *df )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.x * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;
  df[o] = y[o] - df[o] / L;
}

__global__ void update_fista_relaxation_device( int W, int H,
						float alpha,
						float *y, float *p0, float *p1 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  float p = p0[o];
  y[o] = p + alpha * (p - p1[o]);
}


// Perform one iteration
bool coco::tv_convex_iteration_fista( tv_convex_data *data )
{
  // Dual and Primal steps
  bool ok = true;

  // Compute gradient of data term given current relaxation
  size_t W = data->_W;
  size_t H = data->_H;
  tv_convex_workspace *w = data->_workspace;  
  tv_l2_workspace *w_rof = data->_rof->_workspace;
  data->_fn_grad_data_term( data->_callback_context, W,H, w->_y, w_rof->_f );

  // Compute approximated function for ROF iterations
  compute_fista_rof_function<<< w_rof->_dimGrid, w_rof->_dimBlock >>>
    ( data->_W, data->_H,
      data->_L,
      w->_y, w_rof->_f );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Copy to initial value of ROF minimization
  CUDA_SAFE_CALL( cudaMemcpy( w_rof->_u, w_rof->_f, data->_nfbytes, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Test
  /*
  CUDA_SAFE_CALL( cudaMemcpy( w->_u, w_rof->_u, data->_nfbytes, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
  */

  // Perform inner ROF iterations
  data->_rof->_lambda = 1.0f / data->_L;
  tv_l2_initialize( data->_rof, NULL );
  for ( size_t i=0; i<data->_gp_iter; i++ ) {
    tv_l2_iteration_fgp( data->_rof );
  }

  // Update relaxation
  float alpha_new = 0.5f + 0.5f * hypotf( 1.0f, 2.0f*(data->_alpha));
  float r = (data->_alpha - 1.0f) / alpha_new;
  update_fista_relaxation_device<<< w_rof->_dimGrid, w_rof->_dimBlock >>>
    ( data->_W, data->_H,
      r,
      w->_y, w_rof->_u, w->_u );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Update fields
  std::swap( w_rof->_u, w->_u );
  data->_alpha = alpha_new;
  return ok;
}
