/* -*-c++-*- */
/** \file tv_inpainting.cu
   Algorithms to solve the TV model with inpainting data term.

   Workspace handling and access code.

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
#include "tv_inpainting.h"
#include "tv_inpainting.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"


// Compute gradient of inpainting data term (device)
__global__ void compute_inpainting_gradient_device( int W, int H,
						    float *dE, double lambda,
						    float *mask, float *u, float *f )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  dE[o] = mask[o] * ( u[o] - f[o] ) / lambda;
}

// Compute gradient of inpainting data term (device code wrapper)
static bool compute_inpainting_gradient( void* context,
					 size_t W, size_t H,
					 float *u, float *dE )
{
  // Recover workspace
  coco::tv_inpainting_workspace *w = (coco::tv_inpainting_workspace*)context;

  // Kernel call for each channel
  compute_inpainting_gradient_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W, H,
      dE, w->_lambda, w->_mask, u, w->_f );

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Alloc PDE data with sensible defaults
coco::tv_inpainting_data* coco::tv_inpainting_data_alloc( gsl_matrix *mask, gsl_matrix *f )
{
  tv_inpainting_data *data = new tv_inpainting_data;
  size_t W = f->size2;
  size_t H = f->size1;

  // Workspace
  data->_workspace = new tv_inpainting_workspace;
  memset( data->_workspace, 0, sizeof( tv_inpainting_workspace ));
  tv_inpainting_workspace *w = data->_workspace;

  // CUDA Block dimensions
  w->_dimBlock = dim3( cuda_default_block_size_x(),
		       cuda_default_block_size_y() );
  size_t blocks_w = W / w->_dimBlock.x;
  if ( W % w->_dimBlock.x != 0 ) {
    blocks_w += 1;
  }
  size_t blocks_h = H / w->_dimBlock.y;
  if ( H % w->_dimBlock.y != 0 ) {
    blocks_h += 1;
  }
  w->_dimGrid = dim3(blocks_w, blocks_h);

  // Arrays for f and k*f
  size_t Nf = W*H*sizeof(float);
  CUDA_SAFE_CALL( cudaMalloc( &w->_f, Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_mask, Nf ));

  // Copy f
  cuda_memcpy( w->_f, f );
  // Copy mask
  cuda_memcpy( w->_mask, mask );

  // TV-Convex optimizer
  data->_lambda = 0.01;
  data->_tv_convex = tv_convex_data_alloc( f->size2, f->size1,
					   &compute_inpainting_gradient,
					   w );
  return data;
}



// Free up PDE data
bool coco::tv_inpainting_data_free( tv_inpainting_data *data )
{
  // Free GPU fields
  tv_inpainting_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaFree( w->_f ));
  CUDA_SAFE_CALL( cudaFree( w->_mask ));
  tv_convex_data_free( data->_tv_convex );
  delete data->_workspace;
  delete data;
  return true;
}



// Initialize workspace with current solution
bool coco::tv_inpainting_initialize( tv_inpainting_data *data,
				     gsl_matrix* u )
{
  data->_workspace->_lambda = data->_lambda;
  data->_tv_convex->_L = 1.0 / data->_lambda;
  return tv_convex_initialize( data->_tv_convex, u );
}

// Get current solution
bool coco::tv_inpainting_get_solution( tv_inpainting_data *data,
				       gsl_matrix* u )
{
  return tv_convex_get_solution( data->_tv_convex, u );
}


double coco::tv_inpainting_energy( tv_inpainting_data *data )
{
  return tv_convex_energy( data->_tv_convex );
}


/*****************************************************************************
       TV-Inpainting algorithm I: Specialized FISTA (Beck/Teboulle 2008)
*****************************************************************************/

// Perform one full iteration
bool coco::tv_inpainting_iteration_fista( tv_inpainting_data *data )
{
  return tv_convex_iteration_fista( data->_tv_convex );
}

