/* -*-c++-*- */
/** \file tc_linear.cu
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
#include "tc_linear.h"
#include "tc.cuh"
#include "tc_arrays.cuh"
#include "tc_energies.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../common/gsl_image.h"
#include "../common/gsl_matrix_derivatives.h"
#include "../common/menger_curvature.h"
#include "../common/profiler.h"

using namespace std;

// Alloc PDE data with sensible defaults
coco::tc_data* coco::tc_data_alloc( size_t N, gsl_matrix *a )
{
  tc_data *data = new tc_data;
  size_t W = a->size2;
  size_t H = a->size1;

  data->_W = W;
  data->_H = H;
  data->_N = N;
  data->_iteration = 0;

  // Params
  data->_alpha = 0.1;
  data->_p = 2.0;
  data->_lambda = 0.01;
  data->_tv_lambda = 0.0;
  data->_alpha = 1.0;
  data->_inner_iterations = 15;

  // Workspace
  data->_workspace = new tc_workspace;
  memset( data->_workspace, 0, sizeof( tc_workspace ));
  tc_workspace *w = data->_workspace;
  w->_N2 = (N-1)/2;
  assert( w->_N2 * 2 + 1 == (int)data->_N );
  w->_N = data->_N;
  w->_H = data->_H;
  w->_W = data->_W;

  // CUDA arrays
  w->_Nf = W*H*sizeof(stcflt);
  w->_Nv = N*N*N*N*w->_Nf;
  w->_num_dual = NUM_DUAL;

  // Compute total required memory (before allocating)
  TRACE( "Allocating tc data " << W << " x " << H << " x " << N << " ..." << endl );
  size_t bpM = 1048576;
  // u,uq, a,f
  size_t nprimal = ( w->_Nf * 4 ) / bpM;
  TRACE( "  " << nprimal << " Mb for primal and auxiliary variables." << endl );
  // px, py
  size_t ndual = ( 0 * w->_Nv + 2 * w->_Nf ) / bpM;
  ndual += ( w->_num_dual * w->_Nv + w->_Nf ) / bpM;
  TRACE( "  " << ndual << " Mb for dual variables." << endl );
  size_t ntotal_gpu = nprimal + ndual;
  TRACE( "Total memory required (gpu): " << ntotal_gpu << " Mb." << endl );

  // Primal variable
  CUDA_SAFE_CALL( cudaMalloc( &w->_u, w->_Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_uq, w->_Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_u_star, w->_Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_D, w->_Nf ));
  // Data term
  CUDA_SAFE_CALL( cudaMalloc( &w->_a, w->_Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_f, w->_Nf ));
  cuda_memcpy( w->_a, a );
  cuda_memcpy( w->_f, a );

  // Epigraph variable
  //  cvl_alloc_array( w, w->_v );
  //cvl_alloc_array( w, w->_vq );
  //cvl_alloc_array( w, w->_vf );

  // Dual variables
  TRACE( "Alloc dual variables." << endl );
  CUDA_SAFE_CALL( cudaMalloc( &w->_px, w->_Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_py, w->_Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_pxq, w->_Nf ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_pyq, w->_Nf ));
  for ( size_t i=0; i<w->_num_dual; i++ ) {
    //    for ( size_t j=0; j<2; j++ ) {
    //  for ( size_t k=0; k<2; k++ ) {

    vector<stcflt*> &xiv = w->_p[i]; //cvl_get_dual_variable( w, i,j,k );
    cvl_alloc_array( w, xiv );

  }
  TRACE( "Alloc complete." << endl );

  // Precomputed curvature weights
  int N2 = w->_N2;
  int N4 = N*N*N*N;
  w->_cp_cpu = new stcflt[ N4 ];
  for ( int y0o=-N2; y0o<=N2; y0o++ ) {
    for ( int y1o=-N2; y1o<=N2; y1o++ ) {
      for ( int z0o=-N2; z0o<=N2; z0o++ ) {
	for ( int z1o=-N2; z1o<=N2; z1o++ ) {
	  int idx = AIND( y0o+N2, y1o+N2, z0o+N2, z1o+N2 );
	  assert( idx >= 0 && idx < N4 );
	  w->_cp_cpu[ idx ] = menger_curvature_weight( y0o, y1o, z0o, z1o );
	}
      }
    }
  }

  CUDA_SAFE_CALL( cudaMalloc( &w->_cp, N4*sizeof( stcflt ) ));
  CUDA_SAFE_CALL( cudaMemcpy( w->_cp, w->_cp_cpu, N4*sizeof( stcflt ), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Finalize
  TRACE( "Finalizing init." << endl );
  w->_sigma = 1.0;
  w->_lambda = data->_lambda;
  w->_tv_lambda = data->_tv_lambda;
  w->_rof_lambda = data->_lambda;
  w->_L = 1.0;
  w->_t = 1.0;
  //
  w->_curvature = true;
  w->_mask = NULL;
  w->_b = NULL;
  w->_bq = NULL;
  //
  w->_energy_scale_u = 1.0 / double(W*H);
  w->_energy_scale_v = 1.0 / double(W*H*N*N*N*N);

  // Block sizes
  w->_dimBlock = dim3( cuda_default_block_size_x(),
		       cuda_default_block_size_y() );
  w->_dimGrid = dim3( W / w->_dimBlock.x + (W%w->_dimBlock.x) == 0 ? 0 : 1,
		      H / w->_dimBlock.y + (H%w->_dimBlock.y) == 0 ? 0 : 1 );

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return data;
}


// Set mask for inpainting model
bool coco::tc_set_inpainting_mask( tc_data *data, gsl_matrix *mask )
{
  tc_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaMalloc( &w->_mask, w->_Nf ));
  cuda_memcpy( w->_mask, mask );
  return true;
}



// Free up PDE data
bool coco::tc_data_free( tc_data *data )
{
  // Free CUDA arrays
  tc_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaFree( w->_u ));
  CUDA_SAFE_CALL( cudaFree( w->_uq ));
  CUDA_SAFE_CALL( cudaFree( w->_u_star ));
  CUDA_SAFE_CALL( cudaFree( w->_D ));
  CUDA_SAFE_CALL( cudaFree( w->_a ));
  CUDA_SAFE_CALL( cudaFree( w->_f ));
  CUDA_SAFE_CALL( cudaFree( w->_cp ));

  // Dual vars
  CUDA_SAFE_CALL( cudaFree( w->_px ));
  CUDA_SAFE_CALL( cudaFree( w->_py ));
  CUDA_SAFE_CALL( cudaFree( w->_pxq ));
  CUDA_SAFE_CALL( cudaFree( w->_pyq ));

  for ( size_t i=0; i<w->_num_dual; i++ ) {
    vector<stcflt*> &xi = w->_p[i];
    cvl_free_array( w, xi );
  }

  // Kernels
#ifdef CUDA_DOUBLE
  if ( w->_b != NULL ) {
    cuda_kernel_dbl_free( w->_b );
    //Copy    cuda_kernel_dbl_free( w->_bq );
  }
#else
  if ( w->_b != NULL ) {
    cuda_kernel_free( w->_b );
    //Copy    cuda_kernel_free( w->_bq );
  }
#endif

  // Free up workspace
  delete[] w->_cp_cpu;
  delete data->_workspace;
  delete data;
  return true;
}


// Get current solution
bool coco::tc_get_solution( tc_data *data,
					  gsl_matrix* u )
{
  tc_workspace *w = data->_workspace;
  cuda_memcpy( u, w->_u );
  // Wait for GPU and return
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Initialize workspace with current solution
bool coco::tc_initialize( tc_data *data,
					gsl_matrix* u )
{
  // Copy u
  tc_workspace *w = data->_workspace;

  // TEST: Neutral init
  //gsl_matrix_set_all( u, 0.0 );
  // Copy current solution to GPU
  data->_iteration = 0;
  cuda_memcpy( w->_u, u );
  cuda_memcpy( w->_uq, u );

  // Init v
  /*
    tc_init_product_field( data, w->_vq, w->_u );
  */

  // Clear dual variables
  CUDA_SAFE_CALL( cudaMemset( w->_px, 0, w->_Nf ));
  CUDA_SAFE_CALL( cudaMemset( w->_py, 0, w->_Nf ));
  CUDA_SAFE_CALL( cudaMemset( w->_pxq, 0, w->_Nf ));
  CUDA_SAFE_CALL( cudaMemset( w->_pyq, 0, w->_Nf ));

  // Init dual for v
  //reset_dual_variables( data, w->_v );
  for ( size_t i=0; i<w->_num_dual; i++ ) {
    cvl_clear_array( w, w->_p[i] );
  }

  // Finalize
  if ( data->_lambda != 0.0 ) {
    w->_curvature = true;
  }
  else {
    w->_curvature = false;
  }
  w->_tv_lambda = data->_tv_lambda;

  // Wait for GPU and return
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return false;
}


// Init kernel
bool coco::tc_set_separable_kernel( tc_data *data, gsl_vector *kernel )
{
  tc_workspace *w = data->_workspace;
  assert( w->_b == NULL );
  assert( w->_bq == NULL );
#ifdef CUDA_DOUBLE
  w->_b = coco::cuda_kernel_dbl_alloc_separable( kernel, kernel );
  w->_bq = w->_b;
#else
  w->_b = coco::cuda_kernel_alloc_separable( kernel, kernel );
  w->_bq = w->_b;
#endif

  // TODO: Assuming that kernel is symmetric
  cout << "WARNING: ASSUMING SYMMETRIC KERNEL IN DEBLURRING." << endl;
  return true;
}



// Init kernel
bool coco::tc_set_kernel( tc_data *data, gsl_matrix *kernel )
{
  tc_workspace *w = data->_workspace;
  assert( w->_b == NULL );
  assert( w->_bq == NULL );

  gsl_matrix *kq = gsl_matrix_alloc( kernel->size2, kernel->size1 );
  gsl_matrix_transpose_memcpy( kq, kernel );
#ifdef CUDA_DOUBLE
  w->_b = coco::cuda_kernel_dbl_alloc( kernel );
  w->_bq = coco::cuda_kernel_dbl_alloc( kq );
#else
  w->_b = coco::cuda_kernel_alloc( kernel );
  w->_bq = coco::cuda_kernel_alloc( kq );
#endif

  return true;
}





