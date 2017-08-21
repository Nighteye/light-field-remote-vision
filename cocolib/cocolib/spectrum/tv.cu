/* -*-c++-*- */

#include <algorithm> 
#include "../defs.h"
#include "../cuda/cuda_arrays.h"
#include "../cuda/cuda_kernels.cuh"
#include "../vtv/vtv.h"
#include "../vtv/vtv.cuh"

#include "tv.h"
#include "tv.cuh"


// Alloc VTV spectral decomposition for an image
coco::coco_vtv_spectrum_data* coco::coco_vtv_spectrum_alloc( vector<gsl_matrix*> F )
{
  coco_vtv_spectrum_data *data = new coco_vtv_spectrum_data;
  data->_N = F.size();
  assert( data->_N > 0 );
  data->_W = F[0]->size2;
  data->_H = F[0]->size1;
  int W = data->_W;
  int H = data->_H;
  int N = data->_N;

  // Workspace
  data->_workspace = new coco_vtv_spectrum_workspace;
  coco_vtv_spectrum_workspace *w = data->_workspace;

  // Buffers for source and iterates
  w->_F.alloc( N, W,H );
  w->_U[0].alloc( N, W,H );
  w->_U[1].alloc( N, W,H );
  w->_U[2].alloc( N, W,H );
  w->_residual.alloc( N, W,H );

  // Ring buffer
  w->_current = 0;
  w->_next = 1;
  w->_previous = 2;
  w->_iteration = 0;

  // Load input image
  w->_F.copy_from_cpu( F );
  w->_U[w->_current].copy_from_gpu( w->_F );

  // Block layout
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

  // VTV submodel
  w->_vtv = coco_vtv_alloc( F );

  // Algorithm data
  // Time step
  data->_dt = 1.0;
  // Number of iterations (time steps)
  // If zero, proceeds until residual is zero
  data->_iterations = 500;

  // Number of inner iterations (fista/fgp)
  data->_inner_iterations = 100;

  // Regularizer
  data->_regularizer = 1;

  // Done
  return data;
}


// Free up VTV spectral data
bool coco::coco_vtv_spectrum_free( coco_vtv_spectrum_data *data )
{
  coco_vtv_free( data->_workspace->_vtv );
  for ( size_t i=0; i<data->_workspace->_phi.size(); i++ ) {
    delete data->_workspace->_phi[i];
  }
  delete data->_workspace;
  data->_workspace = NULL;
  delete data;
  return true;
}




/*****************************************************************************
       Spectral decomposition
*****************************************************************************/

// Perform full decomposition within given range
bool coco::coco_vtv_spectrum_decomposition( coco_vtv_spectrum_data *data )
{
  TRACE( "Spectrum decomposition " << data->_W << " x " << data->_H
	 << ", " << data->_N << " channels." << endl );
  TRACE( data->_iterations << " iterations total." << endl );
  
  TRACE( "  [" );

  clock_t t1 = clock();
  data->_workspace->_iteration = 0;
  for ( size_t i=0; i<data->_iterations; i++ ) {
    if ( (i%(data->_iterations/20+1))==0 ) {
      TRACE( "." );
    }

    coco_vtv_spectrum_decomposition_iteration( data );
  }

  clock_t t2 = clock();


  TRACE( "] done." << endl );

  double secs = double(t2-t1) / double(CLOCKS_PER_SEC);

  TRACE( "total runtime : " << secs << "s." << endl );
  TRACE( "per iteration : " << secs / double(data->_iterations)  << "s." << endl );
  TRACE( "iter / s      : " << double(data->_iterations) / secs  << endl );
  
  return true;
}


// Reprojection for RGB, TV_S
__global__ void spectrum_compute_phi_and_residual( int W, int H,
						   float iteration,
						   const float *Up, const float *Uc, const float *Un,
						   float *phi,
						   float *residual )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // get values: previous (n-1), current (n), next (n+1)
  float Upv = Up[o];
  float Ucv = Uc[o];
  float Unv = Un[o];

  // phi: formula (15) in techreport
  phi[o] = iteration * ( Upv + Unv - 2.0f * Ucv );
  // residual: formula (17) in techreport
  residual[o] = (iteration+1.0f) * Ucv - iteration * Unv;
}


// Perform full decomposition within given range
bool coco::coco_vtv_spectrum_decomposition_iteration( coco_vtv_spectrum_data *data )
{
  // Compute next iterate
  coco_vtv_spectrum_workspace *w = data->_workspace;
  w->_vtv->_lambda = data->_dt;
  coco_vtv_rof_gpu_reinit( w->_vtv, w->_U[ w->_current ].mem() );
  for ( size_t i=0; i<data->_inner_iterations; i++ ) {
    coco_vtv_rof_iteration_fista( w->_vtv );
  }
  w->_U[ w->_next ].copy_from_gpu( w->_vtv->_workspace->_U );

  /*
  write_test_image_rgb( data->_W, data->_H,
			   w->_U[w->_current].channel(0),
			   w->_U[w->_current].channel(1),
			   w->_U[w->_current].channel(2),
			   "./out/Uc_iteration_%04i.png", w->_iteration,
			   false );
  write_test_image_rgb( data->_W, data->_H,
			   w->_U[w->_next].channel(0),
			   w->_U[w->_next].channel(1),
			   w->_U[w->_next].channel(2),
			   "./out/Un_iteration_%04i.png", w->_iteration,
			   false );
  */

  // Compute next phi and residual
  if ( w->_iteration == 0 ) {
    // First iteration, no phi and residual equals input
    w->_residual.copy_from_gpu( w->_F );
  }
  else {
    gpu_2D_float_array_vector* phi = new gpu_2D_float_array_vector;
    phi->alloc( data->_N, data->_W, data->_H );
    for ( int n=0; n<data->_N; n++ ) {
      spectrum_compute_phi_and_residual<<< w->_dimGrid, w->_dimBlock >>>
	( data->_W, data->_H,
	  w->_iteration,
	  w->_U[ w->_previous ].channel( n ),
	  w->_U[ w->_current ].channel( n ),
	  w->_U[ w->_next ].channel( n ),
	  phi->channel( n ),
	  w->_residual.channel( n ) );
    }
    w->_phi.push_back( phi );

    // TODO: compute norm
    w->_spectrum.push_back( l1_norm( *phi ) );
  }

  // Update ring buffer
  int tmp = w->_previous;
  w->_previous = w->_current;
  w->_current = w->_next;
  w->_next = tmp;

  w->_iteration ++;
  return true;
}



// Return reconstruction with given coefficients
bool coco::coco_vtv_spectrum_reconstruction( coco_vtv_spectrum_data *data,
					     vector<float> &coefficients,
					     bool add_residual,
					     vector<gsl_matrix*> &result )
{
  // Alloc reconstruction buffer
  int N = data->_N;
  int W = data->_W;
  int H = data->_H;
  gpu_2D_float_array_vector reconstruction;
  reconstruction.alloc( N, W,H );

  // Init buffer with zero or residual
  coco_vtv_spectrum_workspace *w = data->_workspace;
  if ( add_residual ) {
    reconstruction.copy_from_gpu( w->_residual );
  }
  else {
    reconstruction.set_zero();
  }

  // Add all phis multiplied with spectrum
  int R = (int)min( coefficients.size(), w->_phi.size() );
  for ( int i=0; i<R; i++ ) {
    for ( int n=0; n<N; n++ ) {
      cuda_add_scaled_to_device<<< w->_dimGrid, w->_dimBlock >>>
	( W,H, w->_phi[i]->channel(n), coefficients[i], reconstruction.channel(n) );
    }
  }

  // Copy to CPU
  reconstruction.copy_to_cpu( result );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Return current spectrum
bool coco::coco_vtv_spectrum_get_spectrum( coco_vtv_spectrum_data *data, vector<float> &spectrum )
{
  spectrum.clear();
  coco_vtv_spectrum_workspace *w = data->_workspace;
  spectrum.insert( spectrum.end(), w->_spectrum.begin(), w->_spectrum.end() );
  return true;
}


// Return single mode
bool coco::coco_vtv_spectrum_get_mode( coco_vtv_spectrum_data *data, int index, vector<gsl_matrix*> &mode )
{
  coco_vtv_spectrum_workspace *w = data->_workspace;
  assert( index < (int)w->_phi.size() );
  w->_phi[index]->copy_to_cpu( mode );
  return true;
}

