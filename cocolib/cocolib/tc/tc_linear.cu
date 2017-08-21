/* -*-c++-*- */
/** \file tc_linear.cu
   Algorithms to solve the TC model with linear data term.

   Copyright (C) 2011 Bastian Goldluecke,
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
#include "tc_l2.h"
#include "tc_linear.h"
#include "tc.cuh"
#include "tc_arrays.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"
#include "../common/gsl_image.h"
#include "../common/gsl_matrix_derivatives.h"
#include "../common/menger_curvature.h"
#include "../common/profiler.h"


/******************************************
    EXPERIMENTAL MINIMIZATION ALGORITHM
    see initial version paper ICCV 2011
*******************************************/


/******************************************
    MAIN ITERATION: Fista
******************************************/

__global__ void cvl_fista_init_derivative_device( int W, int H,
						  stcflt step,
						  stcflt *uq,
						  stcflt *mask, stcflt *f,
						  stcflt *D )
{
  // Global thread index
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int x1 = blockDim.y * blockIdx.y + threadIdx.y;
  if ( x0>=W || x1>=H ) {
    return;
  }
  int o = x1*W + x0;
  stcflt uqv = uq[o];
  D[o] = uqv - step * f[o];
}


__global__ void cvl_fista_overrelaxation_device( int W, int H,
						 stcflt alpha,
						 stcflt *u,
						 stcflt *uq )

{
  // Global thread index
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int x1 = blockDim.y * blockIdx.y + threadIdx.y;
  if ( x0>=W || x1>= H ) {
    return;
  }
  int o = x1*W + x0;
  stcflt uo = u[o];
  stcflt un = uq[o];
  u[o] = un;
  uq[o] = uo + alpha * (un - uo);
}



// Perform one iteration (outer loop)
bool coco::tc_linear_fista_init( tc_data *data )
{
  tc_workspace *w = data->_workspace;
  w->_t = 1.0;
  w->_lambda = data->_lambda;
  w->_L = (w->_lambda != 0.0) ? 1.0 / w->_lambda : 10.0;
  cv_init_rof_stepsize( data, 1.0 / w->_L );

  // Start iteration with input f
  CUDA_SAFE_CALL( cudaMemcpy( w->_u, w->_a, w->_Nf, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaMemcpy( w->_uq, w->_a, w->_Nf, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  return true;
}


bool coco::tc_linear_fista_iteration( tc_data *data )
{
  tc_workspace *w = data->_workspace;
  data->_iteration ++;

  TRACE5( "****** FISTA ITERATION " << data->_iteration << " ******" << std::endl );

  // Step 1: Compute derivative of uq using inpainting mask and Lipschitz
  // constant
  float factor = (w->_lambda != 0.0) ? 1.0 / (w->_lambda * w->_L) : 10.0;
  cvl_fista_init_derivative_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H,
      factor,
      w->_uq, w->_mask, w->_a,
      w->_f );

  // Step 1a: Backup old u for overrelaxation
  CUDA_SAFE_CALL( cudaMemcpy( w->_u_star, w->_u, w->_Nf, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Step 2: Perform inner ROF iterations
  TRACE6( "  ROF iterations [" );
  for ( size_t i=0; i<20; i++ ) {
    TRACE6( "." );

    // Primal step: Compute K^* xi - 1/lambda f
    if (!tc_l2_iteration( data )) {
      ERROR( "ROF iteration failure." << std::endl );
      return false;
    }
  }
  TRACE6( "]" << std::endl );
    
  // Step 3: Recover u and compute overrelaxation in uq
  double new_t = 0.5 * ( 1.0 + sqrt( 1.0 + 4.0 * pow( (double)w->_t, 2.0 )) );
  double alpha = (w->_t - 1.0) / new_t;
  cvl_fista_overrelaxation_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H,
      alpha,
      w->_u_star,
      w->_uq );
  w->_t = new_t;
  return true;
}






