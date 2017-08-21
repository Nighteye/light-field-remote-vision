/* -*-c++-*- */
/** \file tv_linear.cu
   Algorithms to solve the TV model with linear data term.

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
#include "tv_linear.h"
#include "tv_linear.cuh"
#include "tv_convex.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../common/gsl_matrix_helper.h"


// Compute gradient of deblurring data term (device)
__global__ void compute_linear_gradient_device( int W, int H,
						float *dE,
						float *a, float *u )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  dE[o] = a[o];
}

// Compute gradient of deblurring data term (device code wrapper)
static bool compute_linear_gradient( void* context,
				     size_t W, size_t H,
				     float *u, float *dE )
{
  // Recover workspace
  coco::tv_linear_workspace *w = (coco::tv_linear_workspace*)context;

  // Kernel call
  compute_linear_gradient_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W, H, dE, w->_a, u );

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Alloc PDE data with sensible defaults
coco::tv_linear_data* coco::tv_linear_data_alloc( gsl_matrix *a, gsl_matrix *g )
{
  tv_linear_data *data = new tv_linear_data;
  size_t W = a->size2;
  size_t H = a->size1;

  // Workspace
  data->_workspace = new tv_linear_workspace;
  memset( data->_workspace, 0, sizeof( tv_linear_workspace ));
  tv_linear_workspace *w = data->_workspace;

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

  // Arrays for linear term
  size_t Nf = W*H*sizeof(float);
  CUDA_SAFE_CALL( cudaMalloc( &w->_a, Nf ));

  // Copy a
  cuda_memcpy( w->_a, a );
  data->_a = gsl_matrix_alloc( H,W );
  gsl_matrix_copy_to( a, data->_a );

  // TV-Convex optimizer
  data->_tv_convex = tv_convex_data_alloc( a->size2, a->size1,
					   &compute_linear_gradient,
					   w, g );
  return data;
}



// Free up PDE data
bool coco::tv_linear_data_free( tv_linear_data *data )
{
  // Free GPU fields
  tv_linear_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaFree( w->_a ));
  tv_convex_data_free( data->_tv_convex );
  gsl_matrix_free( data->_a );
  delete data->_workspace;
  delete data;
  return true;
}



// Initialize workspace with current solution
bool coco::tv_linear_initialize( tv_linear_data *data,
				 gsl_matrix* u )
{
  return tv_convex_initialize( data->_tv_convex, u );
}

// Get current solution
bool coco::tv_linear_get_solution( tv_linear_data *data,
				   gsl_matrix* u )
{
  return tv_convex_get_solution( data->_tv_convex, u );
}


double coco::tv_linear_energy( tv_linear_data *data )
{
  // Slow: works on CPU
  // Copy back fields
  tv_convex_workspace *wc = data->_tv_convex->_workspace;
  size_t W = data->_tv_convex->_W;
  size_t H = data->_tv_convex->_H;
  float *u = new float[W*H];
  CUDA_SAFE_CALL( cudaMemcpy( u, wc->_u, W*H*sizeof(float), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  // Compute energy
  double energy = 0.0;
  double ef = 1.0;
  size_t n = 0;
  gsl_matrix *g = data->_tv_convex->_rof->_g;
  for ( size_t y=0; y<H-1; y++ ) {
    for ( size_t x=0; x<W-1; x++ ) {
      double tv_weight = (g == NULL) ? 1.0 : g->data[n];
      energy += tv_weight * hypotf( u[n+1]-u[n], u[n+W]-u[n] );
      energy += ef * data->_a->data[n] * u[n];
      n++;
    }
    // Skip one.
    n++;
  }
  // Cleanup and return
  delete[] u;
  return energy / double(W*H);
}


/*****************************************************************************
       TV-Deblurring algorithm I: Specialized FISTA (Beck/Teboulle 2008)
*****************************************************************************/

// Perform one full iteration
bool coco::tv_linear_iteration_fista( tv_linear_data *data )
{
  return tv_convex_iteration_fista( data->_tv_convex );
}

