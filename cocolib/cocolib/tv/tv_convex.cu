/* -*-c++-*- */
/** \file tv_convex.cu
   Algorithms to solve the TV model with convex data term.

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
#include "tv_convex.h"
#include "tv_convex.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"

// Alloc PDE data with sensible defaults
coco::tv_convex_data* coco::tv_convex_data_alloc( size_t W, size_t H,
						  fn_compute_matrix_callback fn_grad_f,
						  void *callback_context,
						  gsl_matrix *g )
{
  tv_convex_data *data = new tv_convex_data;

  // Texture sizes
  data->_W = W;
  data->_H = H;
  data->_N = data->_W * data->_H;
  // Smoothness parameter
  data->_L = 1.0f;
  // Maximum step size with proven convergence
  data->_tau = 0.125;
  // Inner iterations
  data->_gp_iter = 15;
  // Relaxation variable
  data->_alpha = 1.0;
  // Workspace
  data->_workspace = new tv_convex_workspace;
  memset( data->_workspace, 0, sizeof( tv_convex_workspace ));

  // Callback Functions
  data->_fn_grad_data_term = fn_grad_f;
  data->_fn_data_term = NULL;
  data->_callback_context = callback_context;

  // Size of image matrices in bytes
  data->_nfbytes = data->_N * sizeof(float);

  // Alloc fields
  tv_convex_workspace *w = data->_workspace;

  // Primal variable components
  CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_u), data->_nfbytes ));
  // Relaxation states
  CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_y), data->_nfbytes ));

  // Alloc ROF workspace
  gsl_matrix *tmp = gsl_matrix_alloc( H,W );
  data->_rof = tv_l2_data_alloc( tmp, g );
  gsl_matrix_free( tmp );

  return data;
}



// Free up PDE data
bool coco::tv_convex_data_free( tv_convex_data *data )
{
  // Free GPU fields
  tv_convex_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaFree( w->_u ));
  CUDA_SAFE_CALL( cudaFree( w->_y ));
  tv_l2_data_free( data->_rof );
  delete data->_workspace;
  delete data;
  return true;
}



// Initialize workspace with current solution
bool coco::tv_convex_initialize( tv_convex_data *data,
				 gsl_matrix* u )
{
  data->_alpha = 1.0;
  tv_convex_workspace *w = data->_workspace;
  cuda_memcpy( w->_u, u );
  cuda_memcpy( w->_y, u );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}

// Get current solution
bool coco::tv_convex_get_solution( tv_convex_data *data,
				   gsl_matrix* u )
{
  tv_convex_workspace *w = data->_workspace;
  assert( u->size2 == data->_W );
  assert( u->size1 == data->_H );
  cuda_memcpy( u, w->_u );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


double coco::tv_convex_energy( tv_convex_data *data )
{
  // TODO
  return 0.0;

  /*
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
  */
}
