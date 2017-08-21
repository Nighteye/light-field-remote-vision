/* -*-c++-*- */
/** \file tv_l2_pd_semi_implicit.cu

   TV-L2 solver, primal-dual semi-implicit descent.
   Implements Chambolle 2004,
   "An algorithm for total variation minimization and applications".

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
#include "tv_l2.h"
#include "tv_l2.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"

// Perform one iteration
bool coco::tv_l2_iteration_pd_semi_implicit( tv_l2_data *data )
{
  bool ok = true;
  if ( !tv_l2_dual_step_pd_semi_implicit( data )) {
    ok = false;
  }
  if ( !tv_l2_primal_step_pd_semi_implicit( data )) {
    ok = false;
  }
  return ok;
}




__global__ void tv_l2_primal_step_device_semi_implicit( int W, int H,
							float lambda,
							float *u,
							float *f,
							float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Copy to shared memory.
  extern __shared__ float bd[];
  const int SW = blockDim.x + 1;
  float *bdx = bd;
  float *bdy = bd + SW*(blockDim.y+1);
  int ol = threadIdx.x + threadIdx.y*SW + SW + 1;
  bdx[ol] = px[o];
  bdy[ol] = py[o];
  // Last row & col copies overlap
  if ( threadIdx.x==0 ) {
    bdx[threadIdx.y*SW + SW] = px[o-1];
    bdy[threadIdx.y*SW + SW] = py[o-1];
  }
  if ( threadIdx.y==0 ) {
    bdx[threadIdx.x+1] = px[o-W];
    bdy[threadIdx.x+1] = py[o-W];
  }
  __syncthreads();

  // Step equals divergence of p, backward differences, dirichlet
  // Then projection onto allowed range
  float unew = f[o] - lambda * (bdx[ol] + bdy[ol] - (ox>0)*bdx[ol-1] - (oy>0)*bdy[ol-SW]);
  u[o] = max( 0.0, min( 1.0,unew ));
}

// Perform one primal step
bool coco::tv_l2_primal_step_pd_semi_implicit( tv_l2_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  tv_l2_workspace *w = data->_workspace;

  // Kernel call for each channel
  size_t dimShared = 2 * (w->_dimBlock.x+1) * (w->_dimBlock.y+1) * sizeof(float);
  tv_l2_primal_step_device_semi_implicit<<< w->_dimGrid, w->_dimBlock, dimShared >>>
    ( W,H, 
      data->_lambda,
      w->_u, w->_f, w->_x1, w->_x2 );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


__global__ void tv_l2_dual_step_device_semi_implicit( int W, int H, float tstep,
						      float *u,
						      float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Copy to shared memory.
  extern __shared__ float bd[];
  int SW = blockDim.x + 1;
  int ol = threadIdx.x + threadIdx.y*SW;
  bd[ol] = u[o];
  // Last row & col copies overlap
  if ( threadIdx.x==blockDim.x-1 ) {
    bd[blockDim.x + threadIdx.y*SW] = u[o+1];
  }
  if ( threadIdx.y==blockDim.y-1 ) {
    bd[threadIdx.x + blockDim.y*SW] = u[o+W];
  }
  __syncthreads();

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float px_inc = tstep*(ox<W-1)*(bd[ol+1] - bd[ol]);
  // Y
  float py_inc = tstep*(oy<H-1)*(bd[ol+SW] - bd[ol]);
  // Reproject X,Y
  float n = 1.0f + hypotf( px_inc, py_inc );
  px[o] = (px[o] - px_inc) / n;
  py[o] = (py[o] - py_inc) / n;
}



// Perform one dual step
bool coco::tv_l2_dual_step_pd_semi_implicit( tv_l2_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  tv_l2_workspace *w = data->_workspace;

  // Kernel call for each channel
  if ( w->_g != NULL ) {
    // Weighted TV not supported here
    ERROR( "Weighted TV is not supported for semi-implicit scheme." << std::endl );
    ERROR( "Please Use GP or FPG (both are better anyway)." << std::endl );
    return false;
  }
  else {
    // Regular TV
    size_t dimShared = (w->_dimBlock.x+1) * (w->_dimBlock.y+1) * sizeof(float);
    tv_l2_dual_step_device_semi_implicit<<< w->_dimGrid, w->_dimBlock, dimShared >>>
      ( W,H, data->_tau / data->_lambda,
	w->_u, w->_x1, w->_x2 );
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}
