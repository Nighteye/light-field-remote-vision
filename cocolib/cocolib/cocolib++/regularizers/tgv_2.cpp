/* -*-c++-*- */
/** \file regularizer_tgv_2.cu

    Total generalized variation regularizer, order 2
    after Bredies et al.

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

#include "tgv_2.h"
#include <math.h>

#include "../compute_api/kernels_vtv.h"
#include "../compute_api/kernels_algebra.h"
#include "../compute_api/reprojections.h"


using namespace coco;
using namespace std;


// Construction and destruction
regularizer_tgv_2::regularizer_tgv_2()
{
  _params["alpha"] = 1.0f;
  _params["beta"] = -1.0f;
}

regularizer_tgv_2::~regularizer_tgv_2()
{
}


// Resize problem (default dual variable layout)
bool regularizer_tgv_2::resize( compute_grid *G, int N )
{
  regularizer::resize( G,N );

  // Extra primal variables W, requires twice as many dual variables
  _K = 6*N;
  _M = 2*N;
  return true;
}

// Perform primal step
bool regularizer_tgv_2::primal_step( vector_valued_function_2D *U,
				     float *step_U,
				     const vector_valued_function_2D *P,
				     vector_valued_function_2D *V,
				     float *step_V )
{
  assert( U != NULL );
  assert( P != NULL );
  assert( V != NULL );
  assert_valid_dims( U,P, V );
  float beta = get_parameter( "beta" );

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0

  // Kernel call for each channel, primal variable U
  for ( int i=0; i<U->N(); i++ ) {
    kernel_gradient_operator_primal_step
      ( _G,
	step_U[i],
	U->channel(i),
	P->channel(2*i + 0), P->channel(2*i + 1) );
  }

  // Kernel call for each channel, primal variable V
  for ( int j=0; j<2*_N; j++ ) {
    kernel_gradient_operator_primal_step
      ( _G,
	step_V[j],
	V->channel(j),
	P->channel(2*_N + 2*j + 0), P->channel(2*_N + 2*j + 1) );

    // Add -beta*P[j] to channel j (descent in (V,P))
    kernel_linear_combination
      ( _G,
	-beta*step_V[j],
	P->channel(j),
	1.0f,
	V->channel(j),
	V->channel(j) );
  }

  return true;
}

			  
// Operator row sum norm of K for primal variables
bool regularizer_tgv_2::accumulate_operator_norm_U( float *norm_U ) const
{
  // Default for gradient operator
  for ( int i=0; i<_N; i++ ) {
    norm_U[i] += 4.0f;
  }
  return true;
}

// Operator column sum norms of K and L for dual variables
bool regularizer_tgv_2::accumulate_operator_norm_P( float *norm_P ) const
{
  float beta = get_parameter( "beta" );
  for ( int i=0; i<2*_N; i++ ) {
    norm_P[i] += 2.0f + fabs(beta);
  }
  for ( int j=0; j<4*_N; j++ ) {
    norm_P[j+2*_N] += 2.0f;
  }
  return true;
}


// Operator column sum norms of K and L for dual variables
bool regularizer_tgv_2::accumulate_operator_norm_V( float *norm_V ) const
{
  float beta = get_parameter( "beta" );
  for ( int i=0; i<2*_N; i++ ) {
    norm_V[i] += 4.0f + fabs(beta);
  }
  return true;
}


// Perform primal reprojection
bool regularizer_tgv_2::primal_prox( vector_valued_function_2D *V  )
{
  // VTV_S reprojection for V
  assert( V != NULL );
  assert_valid_dims( NULL,NULL,V );
  reprojection_max_norm( V, 1.0f, 0, _N );
  return true;
}

// Perform dual step
bool regularizer_tgv_2::dual_step( const vector_valued_function_2D *U,
				   vector_valued_function_2D* P,
				   float *step_P,
				   const vector_valued_function_2D *V )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert( P != NULL );
  assert( V != NULL );
  assert_valid_dims( U,P, V );
  float beta = get_parameter( "beta" );

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0
  for ( int i=0; i<U->N(); i++ ) {
    kernel_gradient_operator_dual_step
      ( _G,
	step_P[i],
	U->channel(i),
	P->channel(2*i + 0), P->channel(2*i + 1) );
  }

  // Regularizer dual components for V are stored
  // in the same order just after the ones for U
  for ( int j=0; j<2*_N; j++ ) {
    kernel_gradient_operator_dual_step
      ( _G,
	step_P[2*_N + j],
	V->channel(j),
	P->channel(2*_N + 2*j + 0), P->channel(2*_N + 2*j + 1) );

    // Add beta*V[j] to channel j (ascent in beta(V,P))
    kernel_linear_combination
      ( _G,
	beta*step_P[j],
	V->channel(j),
	1.0f,
	P->channel(j),
	P->channel(j) );
  }

  return true;
}


// Perform dual reprojection
bool regularizer_tgv_2::dual_prox( vector_valued_function_2D* P )
{
  // VTV_F reprojections for U
  assert( P != NULL );
  assert_valid_dims( NULL,P, NULL );
  float alpha = get_parameter( "alpha" );
  reprojection_frobenius_norm( P, alpha, 0, _N );

  // Then perform VTV_F reprojections for V
  // Note: Two consecutive Vs correspond to the gradient of U,
  // i.e. a 2D vectorial function.
  // Thus, groups of four dual variables for V have to be reprojected together
  // TODO: Think about a VTV_J equivalent for this.
  //
  // Note: this corresponds to the primary regularizer weight
  float lambda = get_parameter( "lambda" );
  for ( int j=0; j<_N; j++ ) {
    reprojection_frobenius_norm( P, lambda, 2*_N + 4*j, 2 );
  }

  return true;
}
