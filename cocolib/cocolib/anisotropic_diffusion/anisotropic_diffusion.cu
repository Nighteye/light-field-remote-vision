/* -*-c++-*- */
/** \file anisotropic_diffusion.h
    Perona-Malik isotropic and Weickert's coherence-enhancing diffusion,
    different discretizations,
    inpainting models.

    Copyright (C) 2012 Bastian Goldluecke,
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


#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"
#include "../cuda/cuda_image_processing.h"

#include "anisotropic_diffusion.h"
#include "anisotropic_diffusion.cuh"
#include "anisotropic_diffusion_kernels.cu"

using namespace std;

// Alloc PDE data with sensible defaults
coco::coco_diffusion_data* coco::coco_diffusion_data_alloc( vector<gsl_matrix*> F )
{
  size_t nchannels = F.size();
  assert( nchannels > 0 );
  coco_diffusion_data *data = new coco_diffusion_data;

  // Texture sizes
  data->_nchannels = nchannels;
  data->_W = F[0]->size2;
  data->_H = F[0]->size1;
  data->_N = data->_W * data->_H;
  // Smoothness parameter
  data->_lambda = 1.0f;
  // Data term model
  data->_type = coco_diffusion_data::DIFFUSION_PLAIN;
  
  // Chosen according to heuristics in paper
  // Step size
  data->_tau = 0.1f;

  // Workspace
  data->_workspace = new coco_diffusion_workspace;
  memset( data->_workspace, 0, sizeof( coco_diffusion_workspace ));

  // Alloc fields
  coco_diffusion_workspace *w = data->_workspace;
  w->_U.resize( nchannels );
  w->_F.resize( nchannels );

  // Size of fields over Image space
  size_t W = data->_W;
  size_t H = data->_H;
  size_t N = data->_N;
  w->_nfbytes = N * sizeof(float);

  for ( size_t i=0; i<nchannels; i++ ) {
    // Primal variable components
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_U[i]), w->_nfbytes ));

    // Copy RHS to GPU
    gsl_matrix *f = F[i];
    assert( f != NULL );
    assert( f->size2 == W );
    assert( f->size1 == H );
    float *tmp = cuda_alloc_floats( W*H );
    cuda_memcpy( tmp, f );
    w->_F[i] = tmp;
    CUDA_SAFE_CALL( cudaMemcpy( w->_U[i], w->_F[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
  }

  // Auxiliary fields
  // Number of auxiliary fields during diffusion process
  size_t aux_fields = 6;
  CUDA_SAFE_CALL( cudaMalloc( &w->_aux, aux_fields * w->_nfbytes ));

  // Stencil, if defined
  w->_stencil = NULL;
  // Diffusion tensor
  w->_a = cuda_alloc_floats( N );
  w->_b = cuda_alloc_floats( N );
  w->_c = cuda_alloc_floats( N );

  // Init rest of data
  w->_iteration = 0;
  // CUDA Block dimensions
  cuda_default_grid( data->_W, data->_H, w->_dimGrid, w->_dimBlock );
  // Identity Tensor
  cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, w->_a, 1.0f );
  cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, w->_b, 0.0f );
  cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, w->_c, 1.0f );

  // Default diffusion parameters
  // Step size
  data->_tau = 0.1f;
  // Discretization
  data->_discretization = coco_diffusion_data::DISCRETIZATION_ROTATION_INVARIANT;
  data->_roi_filter_weight = 0.02;
  // Data term model
  data->_type = coco_diffusion_data::DIFFUSION_PLAIN;

  // Diffusion tensor model
  data->_tensor = coco_diffusion_data::TENSOR_COHERENCE_ENHANCING;
  // Diffusion tensor parameters
  // Global weight
  data->_lambda = 1.0;
  // For Perona-Malik and coherence-enhancing: edge strength constant
  data->_K = 0.01;
  // For coherence-enhancing: strength of across-edge diffusion
  data->_c1 = 0.1;
  // Outer scale for structure tensor
  data->_outer_scale = 0.7;
  // Inner scale for structure tensor
  data->_inner_scale = 0.7;

  // Value range
  data->_rmin = 0.0;
  data->_rmax = 1.0;

  // Done
  return data;
}


// Free up PDE data
bool coco::coco_diffusion_data_free( coco_diffusion_data *data )
{
  // Free GPU fields
  coco_diffusion_workspace *w = data->_workspace;
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    CUDA_SAFE_CALL( cudaFree( w->_U[i] ));
    CUDA_SAFE_CALL( cudaFree( w->_F[i] ));
  }

  cuda_free( w->_aux );
  cuda_free( w->_stencil );
  cuda_free( w->_a );
  cuda_free( w->_b );
  cuda_free( w->_c );

  delete data->_workspace;
  delete data;
  return true;
}



// Initialize workspace with current solution
bool coco::coco_diffusion_set_solution( coco_diffusion_data *data,
					vector<gsl_matrix*> &U )
{
  coco_diffusion_workspace *w = data->_workspace;
  assert( U.size() == data->_nchannels );

  for ( size_t i=0; i<data->_nchannels; i++ ) {
    gsl_matrix *u = U[i];
    assert( u->size2 == data->_W );
    assert( u->size1 == data->_H );
    cuda_memcpy( w->_U[i], u );
  }

  return true;
}

// Get current solution
bool coco::coco_diffusion_get_solution( coco_diffusion_data *data,
					vector<gsl_matrix*> &U )
{
  coco_diffusion_workspace *w = data->_workspace;
  assert( U.size() >= data->_nchannels );
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    gsl_matrix *u = U[i];
    assert( u->size2 == data->_W );
    assert( u->size1 == data->_H );
    cuda_memcpy( u, w->_U[i] );
  }
  for ( size_t i=data->_nchannels; i<U.size(); i++ ) {
    gsl_matrix *u = U[i];
    assert( u->size2 == data->_W );
    assert( u->size1 == data->_H );
    cuda_memcpy( u, w->_U[0] );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Init stencil (e.g. for inpainting)
bool coco::coco_diffusion_set_stencil( coco_diffusion_data *data, gsl_matrix *stencil )
{
  coco_diffusion_workspace *w = data->_workspace;
  assert( w->_stencil == NULL );
  assert( stencil != NULL );
  CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_stencil), w->_nfbytes ));
  cuda_memcpy( w->_stencil, stencil );
  return true;
}


// Init diffusion tensor
bool coco::coco_diffusion_set_diffusion_tensor( coco_diffusion_data *data,
						gsl_matrix *a, gsl_matrix *b, gsl_matrix *c )
{
  coco_diffusion_workspace *w = data->_workspace;
  cuda_memcpy( w->_a, a );
  cuda_memcpy( w->_b, b );
  cuda_memcpy( w->_c, c );
  return true;
}



// Compute structure tensor for all channels
bool coco::coco_diffusion_compute_structure_tensor( coco_diffusion_data *data )
{
  coco_diffusion_workspace *w = data->_workspace;
  size_t W = data->_W;
  size_t H = data->_H;

  for ( size_t i=0; i<data->_nchannels; i++ ) {

    // compute structure tensor (TODO: multichannel)
    cuda_structure_tensor( W,H,
			   data->_outer_scale, data->_inner_scale,
			   w->_U[i],
			   w->_a, w->_b, w->_c,
			   w->_aux );

  }

  return true;
}



// Compute diffusion tensor according to pre-defined parameters
bool coco::coco_diffusion_compute_diffusion_tensor( coco_diffusion_data *data )
{
  coco_diffusion_workspace *w = data->_workspace;
  size_t W = data->_W;
  size_t H = data->_H;

  for ( size_t i=0; i<data->_nchannels; i++ ) {

    // compute diffusion tensor from structure tensor
    switch ( data->_tensor ) {
    case coco_diffusion_data::TENSOR_IDENTITY:
      {
	cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, w->_a, 1.0f );
	cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, w->_b, 0.0f );
	cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, w->_c, 1.0f );
      }
      break;

    case coco_diffusion_data::TENSOR_TV:
      {
      };
      break;

    case coco_diffusion_data::TENSOR_PERONA_MALIK:
      {
	// compute diffusion tensor (Perona-Malik)
	float K_sq = data->_K * data->_K;
	cuda_perona_malik_diffusion_tensor( W,H, K_sq, w->_a, w->_b, w->_c );
      }
      break;

    case coco_diffusion_data::TENSOR_COHERENCE_ENHANCING:
      {
	// compute diffusion tensor (Weickert, coherence-enhancing)
	float c1 = data->_c1;
	float c2 = data->_K * data->_K;
	cuda_coherence_enhancing_diffusion_tensor( W,H, 
						   c1,c2,
						   w->_a, w->_b, w->_c );
      }
      break;

    }
  }

  return true;
}


// Perform one diffusion iteration for currently activated scheme
bool coco::coco_diffusion_iteration( coco_diffusion_data *data )
{
  coco_diffusion_workspace *w = data->_workspace;
  size_t W = data->_W;
  size_t H = data->_H;

  for ( size_t i=0; i<data->_nchannels; i++ ) {

    // perform diffusion depending on chosen scheme
    switch ( data->_discretization ) {
    case coco_diffusion_data::DISCRETIZATION_STANDARD:
      {
	cuda_anisotropic_diffusion_simple( W,H,
					   data->_tau,
					   w->_a, w->_b, w->_c,
					   w->_U[i], w->_aux );
      }
      break;
    
    case coco_diffusion_data::DISCRETIZATION_NON_NEGATIVITY:
      {
	/* SEEMS TO NOT WORK IF u<0 ?
	static bool nonneg_warn = true;
	if ( nonneg_warn ) {
	  nonneg_warn = false;
	  ERROR( "non-negativity discretization is currently bugged." << endl );
	}
	*/
	cuda_anisotropic_diffusion_nonneg( W,H,
					   data->_tau,
					   w->_a, w->_b, w->_c,
					   w->_U[i], w->_aux );
      }
      break;

    case coco_diffusion_data::DISCRETIZATION_ROTATION_INVARIANT:
      {
	cuda_anisotropic_diffusion_roi( W,H,
					data->_tau,
					w->_a, w->_b, w->_c,
					w->_U[i],
					w->_aux );
      }
      break;
    }

    
    // Handle different data terms
    switch ( data->_type ) {
    case coco_diffusion_data::DIFFUSION_PLAIN:
      {
	// nothing specific to do
      }
      break;
      
    case coco_diffusion_data::DIFFUSION_L2:
      {
	// slow implementation
	CUDA_SAFE_CALL( cudaMemcpy( w->_aux, w->_F[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
	cuda_subtract_from_device<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, w->_U[i], w->_aux );
	cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, w->_aux, data->_lambda * data->_tau );
	cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, w->_aux, w->_U[i] );
      }
      break;

    case coco_diffusion_data::DIFFUSION_INPAINTING:
      {
	// restore original image in the valid region (outside inpainting mask)
	assert( w->_stencil != NULL );
	cuda_masked_copy_to_device<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, w->_F[i], w->_stencil, w->_U[i] );
      }
      break;
    }

    cuda_clamp_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_U[i], data->_rmin, data->_rmax );
  }

  return true;
}





/*********************************************************************
 ** DIFFUSION TENSORS
 *********************************************************************/

// Compute diffusion tensor from structure tensor (inplace)
// Variant 1: Weickert coherence-enhancing anisotropic diffusion
bool coco::cuda_coherence_enhancing_diffusion_tensor( size_t W, size_t H,
						      float c1, float c2,
						      gpu_float_array a,
						      gpu_float_array b,
						      gpu_float_array c )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  diffusion_tensor_coherence_enhancing<<< dimGrid, dimBlock >>> ( W,H,
								  c1,c2,
								  a,b,c );

  return true;
}


// Compute diffusion tensor from structure tensor (inplace)
// Variant 2: Perona-Malik edge-enhancing isotropic diffusion
bool coco::cuda_perona_malik_diffusion_tensor( size_t W, size_t H,
					       float K_sq,
					       gpu_float_array a,
					       gpu_float_array b,
					       gpu_float_array c )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  diffusion_tensor_perona_malik<<< dimGrid, dimBlock >>> ( W,H,
							   K_sq,
							   a,b,c );
  
  return true;
}




/*********************************************************************
 ** ANISOTROPIC DIFFUSION
 *********************************************************************/

// Compute single diffusion step for non-negativity scheme using a diffusion tensor
bool coco::cuda_anisotropic_diffusion_roi( size_t W, size_t H,
					   float tau,
					   gpu_float_array a, gpu_float_array b, gpu_float_array c,
					   gpu_float_array u,
					   gpu_float_array workspace )
{
  size_t N = W*H;
  float *ux = workspace + N;
  float *uy = workspace + 2*N;
  float *jx = workspace + 3*N;
  float *jy = workspace + 4*N;
  float *filter_i5x = workspace + 3*N;
  float *filter_i5y = workspace + 4*N;
  
  cuda_dx_roi_neumann( W,H, u, ux, workspace );
  cuda_dy_roi_neumann( W,H, u, uy, workspace );

  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  compute_anisotropic_diffusion_flux_field<<< dimGrid, dimBlock >>> ( W,H, a,b,c, ux, uy, jx, jy );

  cuda_dx_roi_dirichlet( W,H, jx, ux, workspace );
  cuda_dy_roi_dirichlet( W,H, jy, uy, workspace );

  identity_filter_x_5<<< dimGrid, dimBlock >>> ( W,H, u, filter_i5x );
  identity_filter_y_5<<< dimGrid, dimBlock >>> ( W,H, u, filter_i5y );

  float xi = 0.05f;
  compute_anisotropic_diffusion_update<<< dimGrid, dimBlock >>> ( W,H,
								  tau, ux, uy,
								  xi, filter_i5x, filter_i5y,
								  u );
  
  return true;
}


// Compute single diffusion step for non-negativity scheme using a diffusion tensor
bool coco::cuda_anisotropic_diffusion_nonneg( size_t W, size_t H,
					      float tau,
					      gpu_float_array a, gpu_float_array b, gpu_float_array c,
					      gpu_float_array u,
					      gpu_float_array workspace )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );
  compute_anisotropic_diffusion_nonneg_update<<< dimGrid, dimBlock >>> ( W,H, tau, a,b,c, u );
  return true;
}


// Compute single diffusion step for non-negativity scheme using a diffusion tensor
bool coco::cuda_anisotropic_diffusion_simple( size_t W, size_t H,
					      float tau,
					      gpu_float_array a, gpu_float_array b, gpu_float_array c,
					      gpu_float_array u,
					      gpu_float_array workspace )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );
  compute_anisotropic_diffusion_simple_update<<< dimGrid, dimBlock >>> ( W,H, tau, a,b,c, u );
  return true;
}

