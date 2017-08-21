/* -*-c++-*- */
#include <iostream>

#include "vtv.h"
#include "vtv.cuh"

#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"
#include "../defs.h"


// Init functional
bool coco::coco_vtv_rof_sum_init( coco_vtv_data *data,
				  std::vector<float*> Fs,
				  std::vector<float*> weights )
{
  coco_vtv_rof_sum_data_free( data );
  coco_vtv_workspace *w = data->_workspace;
  assert( Fs.size() != 0 );
  assert( weights.size() == Fs.size() );
  for ( size_t i=0; i<Fs.size(); i++ ) {
    w->_sum_Fs.push_back( Fs[i] );
    w->_sum_weights.push_back( weights[i] );
  }
  return true;
}
 
// Cleanup for ROF-sum
bool coco::coco_vtv_rof_sum_data_free( coco_vtv_data *data )
{
  coco_vtv_workspace *w = data->_workspace;
  if ( w->_delete_rof_sum_data ) {
    for ( size_t i=0; i<w->_sum_Fs.size(); i++ ) {
      CUDA_SAFE_CALL( cudaFree( w->_sum_Fs[i] ));
      CUDA_SAFE_CALL( cudaFree( w->_sum_weights[i] ));
    }
    w->_sum_Fs.clear();
    w->_sum_weights.clear();
    w->_delete_rof_sum_data = false;
  }
  return true;
}



bool coco::coco_vtv_rof_sum_ista_step( coco_vtv_data *data )
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
  // Currently only supports single channel

  assert( data->_nchannels == 1 );
  CUDA_SAFE_CALL( cudaMemset( w->_G[0], 0, w->_nfbytes ));
  for ( size_t i=0; i<w->_sum_Fs.size(); i++ ) {
    // u
    CUDA_SAFE_CALL( cudaMemcpy( w->_temp[0], w->_Uq[0], w->_nfbytes, cudaMemcpyDeviceToDevice ));
    // u-f_i
    cuda_subtract_from_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_sum_Fs[i], w->_temp[0] );
    // c_i * (u-f_i)
    cuda_add_scaled_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_temp[0], w->_sum_weights[i], w->_G[0] );
  }

  // 1/L * ...
  cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H, w->_G[0], -1.0 / ( data->_lambda * data->_L ));
  
  // Add current solution
  cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H, w->_Uq[0], w->_G[0] );

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




// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_rof_sum_iteration_fista( coco_vtv_data *data )
{
  // Todo: verify correct maximum step sizes.
  data->_tau = 0.3 / sqrt( 8.0 );
  data->_sigma = 0.3 / sqrt( 8.0 );
  data->_L = data->_workspace->_sum_Fs.size() / data->_lambda;
  bool ok2 = coco_vtv_rof_sum_ista_step( data );
  cuflt alpha_new = 0.5 * ( 1.0 + sqrt( 1.0 + 4.0 * pow( data->_alpha, 2.0 ) ));
  bool ok3 = coco_vtv_rof_overrelaxation( data, ( data->_alpha - 1.0 ) / alpha_new );
  data->_alpha = alpha_new;
  return ok2 && ok3;
}



