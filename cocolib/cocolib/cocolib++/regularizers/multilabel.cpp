/* -*-c++-*- */
/** \file regularizer_multilabel.cu

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

#include "multilabel.h"
#include <math.h>

#include "../compute_api/kernels_vtv.h"
#include "../compute_api/kernels_algebra.h"
#include "../compute_api/reprojections.h"


using namespace coco;
using namespace std;


// Construction and destruction
regularizer_multilabel::regularizer_multilabel()
{
  _SIGMA = 0;
}

regularizer_multilabel::~regularizer_multilabel()
{
}


// Resize problem (default dual variable layout)
bool regularizer_multilabel::resize( compute_grid *G, int N )
{
  regularizer::resize( G,N );

  // One extra dual variable for simplex constraint
  // One extra buffer variable (TODO: use grid framework)
  _K = 2*N + 2;
  _SIGMA = 2*N;
  return true;
}

// Perform primal step
bool regularizer_multilabel::primal_step( vector_valued_function_2D *U,
					  float *step_U,
					  const vector_valued_function_2D *P,
					  vector_valued_function_2D *V,
					  float * )
{
  assert( U != NULL );
  assert( P != NULL );
  assert( V == NULL );
  assert_valid_dims( U,P, V );

  //TRACE( "simplex primal step dual variable " << _SIGMA << endl );
  //U->trace_pixel( 100,100 );
  //P->trace_pixel( 100,100 );

  // Kernel call for simplex constraint, update for
  // regularizer components of P must be overridden
  for ( int i=0; i<U->N(); i++ ) {
    kernel_linear_combination
      ( _G,
	step_U[i],
	P->channel( _SIGMA ),
	1.0f,
	U->channel(i),
	U->channel(i) );

    kernel_clamp( _G, U->channel(i), 0.0f, 1.0f );
  }

  //U->trace_pixel( 100,100 );
  return true;
}

			  
// Operator row sum norm of K for primal variables
bool regularizer_multilabel::accumulate_operator_norm_U( float *norm_U ) const
{
  // plus 1 for simplex constraint
  for ( int i=0; i<_N; i++ ) {
    norm_U[i] += 1.0f;
  }
  return true;
}

// Operator column sum norms of K and L for dual variables
bool regularizer_multilabel::accumulate_operator_norm_P( float *norm_P ) const
{
  // Simplex constraint dual depends on every variable u.
  norm_P[_SIGMA] += _N;
  // Buffer variable, dummy init
  norm_P[_SIGMA+1] += 1.0f;
  return true;
}


// Perform dual step
bool regularizer_multilabel::dual_step( const vector_valued_function_2D *U,
					vector_valued_function_2D* P,
					float *step_P,
					const vector_valued_function_2D *V )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert( P != NULL );
  assert( V == NULL );
  assert_valid_dims( U,P, NULL );

  // Simplex constraint dual variable is increased by 1 - \sum_i u_i
  // Requires temp. storage, for which we use a mock dual variable
  compute_buffer &sigma = P->channel( _SIGMA );
  compute_buffer &step = P->channel( _SIGMA+1 );
  kernel_set_all( _G, step, 1.0f );
  for ( int i=0; i<U->N(); i++ ) {
    kernel_subtract_from( _G, U->channel( i ), step );
  }
  kernel_linear_combination( _G, step_P[_SIGMA], step, 1.0f, sigma, sigma );

  return true;
}


// Perform dual reprojection
bool regularizer_multilabel::dual_prox( vector_valued_function_2D* )
{
  // Sigma is unconstrained.
  return true;
}
