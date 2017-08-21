/* -*-c++-*- */
/** \file tv_l2.cu
   Algorithms to solve the TV-L2 model:

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
#include "tv_l2.h"
#include "tv_l2.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"

// Alloc PDE data with sensible defaults
coco::tv_l2_data* coco::tv_l2_data_alloc( gsl_matrix* f, gsl_matrix *g )
{
  tv_l2_data *data = new tv_l2_data;

  // Texture sizes
  data->_W = f->size2;
  data->_H = f->size1;
  data->_N = data->_W * data->_H;
  // Smoothness parameter
  data->_lambda = 1.0f;
  // Maximum step size with proven convergence
  data->_tau = 0.125;
  // Relaxation variable
  data->_alpha = 1.0;
  // Workspace
  data->_workspace = new tv_l2_workspace;
  memset( data->_workspace, 0, sizeof( tv_l2_workspace ));

  // Size of image matrices in bytes
  data->_nfbytes = data->_N * sizeof(float);

  // Alloc fields
  tv_l2_workspace *w = data->_workspace;

  // Primal variable components
  CUDA_SAFE_CALL( cudaMalloc( &w->_u, data->_nfbytes ));
  // Dual variable XI and relaxation states
  CUDA_SAFE_CALL( cudaMalloc( &w->_x1, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_x2, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_x1e, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_x2e, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_y1, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_y2, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_f, data->_nfbytes ));
  w->_g = NULL;

  // CUDA Block dimensions
  size_t W = data->_W;
  size_t H = data->_H;
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

  // Copy f to GPU
  data->_f = gsl_matrix_alloc( data->_H, data->_W );
  memcpy( data->_f->data, f->data, data->_nfbytes );
  assert( f->size2 == data->_W );
  assert( f->size1 == data->_H );
  cuda_memcpy( w->_f, f );
  // Copy TV weight to GPU
  data->_g = NULL;
  if ( g != NULL ) {
    data->_g = gsl_matrix_alloc( data->_H, data->_W );
    memcpy( data->_g->data, g->data, data->_nfbytes );
    CUDA_SAFE_CALL( cudaMalloc( &w->_g, data->_nfbytes ));
    assert( g->size2 == data->_W );
    assert( g->size1 == data->_H );
    cuda_memcpy( w->_g, g );
  }

  return data;
}



// Free up PDE data
bool coco::tv_l2_data_free( tv_l2_data *data )
{
  // Free GPU fields
  tv_l2_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaFree( w->_u ));
  CUDA_SAFE_CALL( cudaFree( w->_f ));
  CUDA_SAFE_CALL( cudaFree( w->_x1 ));
  CUDA_SAFE_CALL( cudaFree( w->_x2 ));
  CUDA_SAFE_CALL( cudaFree( w->_x1e ));
  CUDA_SAFE_CALL( cudaFree( w->_x2e ));
  CUDA_SAFE_CALL( cudaFree( w->_y1 ));
  CUDA_SAFE_CALL( cudaFree( w->_y2 ));
  if ( w->_g != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_g ));
  }

  gsl_matrix_free( data->_f );
  if ( data->_g != NULL ) {
    gsl_matrix_free( data->_g );
  }
  delete data->_workspace;
  delete data;
  return true;
}



// Initialize workspace with current solution
bool coco::tv_l2_initialize( tv_l2_data *data,
			     gsl_matrix* u )
{
  data->_alpha = 1.0;
  tv_l2_workspace *w = data->_workspace;
  if ( u != NULL ) {
    cuda_memcpy( w->_u, u );
  }
  CUDA_SAFE_CALL( cudaMemset( w->_x1, 0, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_x2, 0, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_x1e, 0, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_x2e, 0, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_y1, 0, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_y2, 0, data->_nfbytes ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}

// Get current solution
bool coco::tv_l2_get_solution( tv_l2_data *data,
				  gsl_matrix* u )
{
  tv_l2_workspace *w = data->_workspace;
  assert( u->size2 == data->_W );
  assert( u->size1 == data->_H );
  cuda_memcpy( u, w->_u );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Get dual variable XI (vector of dimension 2)
bool coco::tv_l2_get_dual_xi( tv_l2_data *data,
			      std::vector<gsl_matrix*> &XI )
{
  tv_l2_workspace *w = data->_workspace;
  assert( XI.size() == 2 );
  for ( size_t i=0; i<2; i++ ) {
    gsl_matrix *xi = XI[i];
    assert( xi->size2 == data->_W );
    assert( xi->size1 == data->_H );
  }
  cuda_memcpy( XI[0], w->_x1 );
  cuda_memcpy( XI[1], w->_x2 );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


double coco::tv_l2_energy( tv_l2_data *data )
{
  // Slow: works on CPU
  // Copy back fields
  tv_l2_workspace *w = data->_workspace;
  size_t W = data->_W;
  size_t H = data->_H;
  float *f = new float[W*H];
  float *u = new float[W*H];
  CUDA_SAFE_CALL( cudaMemcpy( f, w->_f, W*H*sizeof(float), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaMemcpy( u, w->_u, W*H*sizeof(float), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  // Compute energy
  double energy = 0.0;
  double ef = 1.0 / (2.0 * data->_lambda);
  size_t n = 0;
  for ( size_t y=0; y<H-1; y++ ) {
    for ( size_t x=0; x<W-1; x++ ) {
      energy += hypotf( u[n+1]-u[n], u[n+W]-u[n] );
      energy += ef * powf( f[n] - u[n], 2.0f );
      n++;
    }
    // Skip one.
    n++;
  }
  // Cleanup and return
  delete[] u;
  delete[] f;
  return energy / double(W*H);
}
