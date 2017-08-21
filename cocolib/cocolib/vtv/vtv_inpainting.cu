/* -*-c++-*- */
#include <iostream>

#include "vtv.h"
#include "vtv.cuh"

#include "../common/gsl_matrix_helper.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"
#include "../defs.h"


// Perform one single shrinkage step (ISTA)
bool coco::coco_vtv_inpainting_ista_step( coco_vtv_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  coco_vtv_workspace *w = data->_workspace;

  // Start descent from current solution
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    CUDA_SAFE_CALL( cudaMemcpy( w->_Uq[i], w->_U[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Compute gradient of data term
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    // copy u
    CUDA_SAFE_CALL( cudaMemcpy( w->_G[i], w->_Uq[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
    // u-f
    cuda_subtract_from_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_F[i], w->_G[i] );
    // mask * (u-f)
    cuda_multiply_with_device<<<w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_G[i], w->_stencil );
    // 1/L * ...
    cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_G[i], -1.0 / ( data->_lambda * data->_L ));

    // Add current solution
    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_Uq[i], w->_G[i] );
  }

  // Perform a number of primal/dual ROF iterations
  data->_tau = 0.3 / sqrt( 8.0 );
  data->_sigma = 0.3 / sqrt( 8.0 );
  const size_t iter_rof = 10;
  for ( size_t k=0; k<iter_rof; k++ ) {
    coco_vtv_rof_dual_step( data );

    // Primal step kernel call for each channel
    for ( size_t i=0; i<data->_nchannels; i++ ) {
      cuda_rof_primal_prox_step_device<<< w->_dimGrid, w->_dimBlock >>>
	( data->_W, data->_H, data->_tau, 1.0 / data->_L,
	  w->_Uq[i], w->_Uq[i], w->_G[i], w->_X1[i], w->_X2[i] );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }
  }

  return true;
}



// Compute primal energy
double coco::coco_vtv_inpainting_primal_energy( coco_vtv_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  coco_vtv_workspace *w = data->_workspace;

  // Compute gradient of current solution
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    cuda_compute_gradient_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_U[i], w->_X1q[i], w->_X2q[i] );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Compute largest singular value of gradient matrix
  cuda_compute_largest_singular_value_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H,
      w->_X1q[0], w->_X2q[0], w->_X1q[1], w->_X2q[1], w->_X1q[2], w->_X2q[2],
      w->_G[0] );

  // Compute gradient of data term
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    // b*u
    cuda_convolution( w->_b, data->_W, data->_H,
		      w->_Uq[i], w->_temp[i] );
    // b*u-f
    cuda_subtract_from_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_F[i], w->_temp[i] );
    // square
    cuda_square_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_temp[i] );
    // 1/(2 lambda) * ...
    cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_temp[i], 1.0 / ( 2.0 * data->_lambda ));

    // Add to smoothness term
    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_temp[i], w->_G[0] );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Copy temp array to CPU
  double *E = new double[ W*H ];
  CUDA_SAFE_CALL( cudaMemcpy( E, w->_G[0], w->_nfbytes, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  // Compute total energy
  double e = 0.0;
  for ( size_t i=0; i<W*H; i++ ) {
    e += E[i];
  }
  delete[] E;
  return e / double(W*H);
}


// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_inpainting_iteration_fista( coco_vtv_data *data )
{
  // Todo: verify correct maximum step sizes.
  data->_tau = 0.3 / sqrt( 8.0 );
  data->_sigma = 0.3 / sqrt( 8.0 );
  data->_L = 1.0 / data->_lambda;
  bool ok2 = coco_vtv_inpainting_ista_step( data );
  double alpha_new = 0.5 * ( 1.0 + sqrt( 1.0 + 4.0 * pow( data->_alpha, 2.0 ) ));
  bool ok3 = coco_vtv_rof_overrelaxation( data, ( data->_alpha - 1.0 ) / alpha_new );
  data->_alpha = alpha_new;
  return ok2 && ok3;
}


