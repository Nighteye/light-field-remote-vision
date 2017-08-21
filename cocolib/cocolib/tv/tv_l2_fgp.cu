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
#include "tv_l2.h"
#include "tv_l2.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"



// Perform one iteration
bool coco::tv_l2_iteration_fgp( tv_l2_data *data )
{
  // Dual and Primal steps
  bool ok = true;
  if ( !tv_l2_primal_step_fgp( data )) {
    ok = false;
  }
  if ( !tv_l2_dual_step_fgp( data )) {
    ok = false;
  }

  // Update relaxation
  float alpha_new = 0.5f + 0.5f * hypotf( 1.0f, 2.0f*(data->_alpha));
  update_fgp_relaxation( data, (data->_alpha - 1.0f) / alpha_new );
  data->_alpha = alpha_new;
  return ok;
}





__global__ void update_fgp_relaxation_device( int W, int H, 
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


bool coco::update_fgp_relaxation( tv_l2_data *data, float r )
{
  size_t W = data->_W;
  size_t H = data->_H;
  tv_l2_workspace *w = data->_workspace;

  // Kernel call for each channel
  update_fgp_relaxation_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H, r,
      w->_y1, w->_x1, w->_x1e );
  update_fgp_relaxation_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H, r,
      w->_y2, w->_x2, w->_x2e );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Update fields
  std::swap( w->_x1, w->_x1e );
  std::swap( w->_x2, w->_x2e );
  return true;
}



__global__ void tv_l2_primal_step_fgp_device( int W, int H,
					      float lambda,
					      float *u,
					      float *f,
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
  const int SW = blockDim.x+1;
  const int SH = blockDim.y+1;
  float *bdx = bd;
  float *bdy = bd + SW*SH;
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
  float unew = f[o] + lambda * (bdx[ol] + bdy[ol] - (ox>0)*bdx[ol-1] - (oy>0)*bdy[ol-SW]);
  u[o] = max( 0.0f, min( 1.0f,unew ));
}

// Perform one primal step
bool coco::tv_l2_primal_step_fgp( tv_l2_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  tv_l2_workspace *w = data->_workspace;

  // Kernel call for each channel
  size_t dimShared = sizeof(float) * (w->_dimBlock.x+1) * (w->_dimBlock.y+1) * 2;
  tv_l2_primal_step_fgp_device<<< w->_dimGrid, w->_dimBlock, dimShared >>>
    ( W,H,
      data->_lambda,
      w->_u, w->_f, w->_y1, w->_y2 );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


__global__ void tv_l2_dual_step_fgp_device( int W, int H, float tstep,
					    float *u,
					    float *px, float *py,
					    float *rx, float *ry )
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
  int SW = blockDim.x+1;
  int ol = threadIdx.x + threadIdx.y*SW;
  bd[ol] = u[o];
  // Last row & col copies overlap
  if ( threadIdx.x==blockDim.x-1 ) {
    bd[blockDim.x + threadIdx.y * SW] = u[o+1];
  }
  if ( threadIdx.y==blockDim.y-1 ) {
    bd[threadIdx.x + blockDim.y * SW] = u[o+W];
  }
  __syncthreads();

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float px_new = px[o] + tstep*(ox<W-1)*(bd[ol+1] - bd[ol]);
  // Y
  float py_new = py[o] + tstep*(oy<H-1)*(bd[ol+SW] - bd[ol]);
  // Reproject X,Y
  float n = max( 1.0f, hypotf( px_new, py_new ));
  rx[o] = px_new / n;
  ry[o] = py_new / n;
}


__global__ void tv_l2_dual_step_fgp_weighted_tv_device( int W, int H, float tstep,
							float *g,
							float *u,
							float *px, float *py,
							float *rx, float *ry )
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
  float px_new = px[o] + tstep*(ox<W-1)*(bd[ol+1] - bd[ol]);
  // Y
  float py_new = py[o] + tstep*(oy<H-1)*(bd[ol+SW] - bd[ol]);
  // Reproject X,Y
  float gv = g[o];
  if ( gv != 0.0f ) {
    float n = max( 1.0f, hypotf( px_new, py_new ) / gv );
    rx[o] = px_new / n;
    ry[o] = py_new / n;
  }
  else {
    rx[o] = 0.0f;
    ry[o] = 0.0f;
  }
}



// Perform one dual step
bool coco::tv_l2_dual_step_fgp( tv_l2_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  tv_l2_workspace *w = data->_workspace;

  // Kernel call
  int dimShared = (w->_dimBlock.x + 1) * (w->_dimBlock.y+1) * sizeof(float) * 2;
  if ( w->_g != NULL ) {
    // Weighted TV
    tv_l2_dual_step_fgp_weighted_tv_device<<< w->_dimGrid, w->_dimBlock, dimShared >>>
      ( W,H, data->_tau / data->_lambda,
	w->_g, w->_u, w->_y1, w->_y2, w->_x1, w->_x2 );
  }
  else {
    // Regular TV
    tv_l2_dual_step_fgp_device<<< w->_dimGrid, w->_dimBlock, dimShared >>>
      ( W,H, data->_tau / data->_lambda,
	w->_u, w->_y1, w->_y2, w->_x1, w->_x2 );
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}
