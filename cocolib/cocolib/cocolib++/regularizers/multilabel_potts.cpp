/* -*-c++-*- */
/** \file regularizer_multilabel_potts.cu

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

#include "multilabel_potts.h"
#include <math.h>

#include "../compute_api/kernels_vtv.h"
#include "../compute_api/kernels_algebra.h"
#include "../compute_api/reprojections.h"


using namespace coco;
using namespace std;


// Construction and destruction
regularizer_multilabel_potts::regularizer_multilabel_potts()
{
}

regularizer_multilabel_potts::~regularizer_multilabel_potts()
{
}


// Perform primal step
bool regularizer_multilabel_potts::primal_step( vector_valued_function_2D *U,
						float *step_U,
						const vector_valued_function_2D *P,
						vector_valued_function_2D *V,
						float * )
{
  assert( U != NULL );
  assert( P != NULL );
  assert( V == NULL );
  assert_valid_dims( U,P, V );

  // Update for simplex constaint
  regularizer_multilabel::primal_step( U, step_U, P, NULL, NULL );

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

  return true;
}

			  
// Operator row sum norm of K for primal variables
bool regularizer_multilabel_potts::accumulate_operator_norm_U( float *norm_U ) const
{
  // Simplex constraint
  regularizer_multilabel::accumulate_operator_norm_U( norm_U );

  // Default for gradient operator
  for ( int i=0; i<_N; i++ ) {
    norm_U[i] += 4.0f;
  }

  return true;
}

// Operator column sum norms of K and L for dual variables
bool regularizer_multilabel_potts::accumulate_operator_norm_P( float *norm_P ) const
{
  // Simplex constraint
  regularizer_multilabel::accumulate_operator_norm_P( norm_P );

  // Default for gradient operator
  for ( int i=0; i<2*_N; i++ ) {
    norm_P[i] += 2.0f;
  }
  return true;
}


// Perform dual step
bool regularizer_multilabel_potts::dual_step( const vector_valued_function_2D *U,
				   vector_valued_function_2D* P,
				   float *step_P,
				   const vector_valued_function_2D *V )
{
  // First perform update for simplex constraint
  regularizer_multilabel::dual_step( U, P, step_P, NULL );

  // Default is one gradient operator step
  assert( U != NULL );
  assert( P != NULL );
  assert( V == NULL );
  assert_valid_dims( U,P, V );

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0
  for ( int i=0; i<U->N(); i++ ) {
    kernel_gradient_operator_dual_step
      ( _G,
	step_P[i],
	U->channel(i),
	P->channel(2*i + 0), P->channel(2*i + 1) );
  }

  return true;
}


// Perform dual reprojection
bool regularizer_multilabel_potts::dual_prox( vector_valued_function_2D* P )
{
  // VTV_S reprojections for P (all channels independent for Potts)
  // Weight 1/2 for Potts
  assert( P != NULL );
  assert_valid_dims( NULL,P, NULL );
  reprojection_max_norm( P, 0.5f, 0, _N );
  return true;
}
