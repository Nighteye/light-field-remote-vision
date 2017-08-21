/* -*-c++-*- */
/** \file data_term_denoising.cu

    Data term of the L^p denoising model.

    Copyright (C) 2014 Bastian Goldluecke.

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

#include "denoising.h"

#include "../compute_api/kernels_vtv.h"
#include "../compute_api/kernels_algebra.h"
#include "../compute_api/kernels_reprojections.h"

//#include "../../cuda/cuda_helper.h"
//#include "../../cuda/cuda_kernels.cuh"

using namespace coco;


// Construction and destruction
data_term_denoising::data_term_denoising()
{
  _params["p"] = 1.0f;
}

data_term_denoising::~data_term_denoising()
{
}

// Resize problem
bool data_term_denoising::resize( compute_grid *G, int N )
{
  data_term_rof::resize( G,N );

  // One dual variable required per solution dimension
  _K = N;
  return true;
}


// Algorithm implementation

// Operator row sum norm of B for primal variables
bool data_term_denoising::accumulate_operator_norm_U( float *norm_U ) const
{
  // scalar product <u-f,q> adds operator norm 1
  for ( int i=0; i<_N; i++ ) {
    norm_U[i] += 1.0f;
  }
  return true;
}

// Operator column sum norm of B for dual variables
bool data_term_denoising::accumulate_operator_norm_Q( float *norm_Q ) const
{
  // scalar product <u-f,q> adds operator norm 1
  for ( int i=0; i<_N; i++ ) {
    norm_Q[i] += 1.0f;
  }
  return true;
}


// Perform primal update, i.e. gradient step + reprojection
bool data_term_denoising::primal_update( vector_valued_function_2D *U,
				   float *step_U,
				   vector_valued_function_2D *Q )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert( Q != NULL );
  assert_valid_dims( U,Q );

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0

  // Kernel call for each channel
  for ( int i=0; i<U->N(); i++ ) {
    // Descent step is equal to -q times step size
    kernel_linear_combination
      ( U->grid(), -step_U[i], Q->channel(i), 1.0f, U->channel(i), U->channel(i) );
    kernel_clamp
      ( U->grid(), U->channel(i), 0.0f, 1.0f );
  }

  return true;
}



// Perform dual update, i.e. gradient step + reprojection
bool data_term_denoising::dual_update( const vector_valued_function_2D *U,
				       vector_valued_function_2D *Q,
				       float *step_Q )
{
  assert( U != NULL );
  assert( Q != NULL );
  assert_valid_dims( U,Q );
  assert( U->equal_dim( _F ));

  // Kernel call for each channel
  for ( int i=0; i<U->N(); i++ ) {
    // Ascend step is equal to u-f times step size
    kernel_linear_combination
      ( U->grid(), step_Q[i], U->channel(i), 1.0f, Q->channel(i), Q->channel(i) );
    kernel_linear_combination
      ( U->grid(), -step_Q[i], _F->channel(i), 1.0f, Q->channel(i), Q->channel(i) );
  }

  dual_reprojection( Q, step_Q );
  return true;
}


// Perform dual reprojection
bool data_term_denoising::dual_reprojection( vector_valued_function_2D *Q,
					     float *step_Q )
{
  assert( Q != NULL );
  float lambda = get_parameter( "lambda" );

  // Kernel call for each channel
  for ( int i=0; i<Q->N(); i++ ) {
    // Reprojection for Q depends on value of p
    int p = (int)get_parameter( "p" );
    switch (p) {
    case 1:
      {
	kernel_reproject_euclidean_1D
	  ( Q->grid(),
	    1.0f / (2.0f * lambda),
	    Q->channel(i) );
      }
      break;

    case 2:
      {
	kernel_scale
	  ( Q->grid(), Q->channel(i), 1.0f / ( 1.0f + 0.5f * step_Q[i] * lambda ) );
      }
      break;

    default:
      ERROR( "L^p denoising implemented only for p=1,2." << std::endl );
      assert( false );
    }
  }

  return true;
}
