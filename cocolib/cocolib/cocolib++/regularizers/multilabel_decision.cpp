/* -*-c++-*- */
/** \file regularizer_multilabel_decision.cu

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

#include "multilabel_decision.h"
#include <math.h>

#include "../compute_api/kernels_vtv.h"
#include "../compute_api/kernels_algebra.h"
#include "../compute_api/reprojections.h"


using namespace coco;
using namespace std;


// Construction and destruction
regularizer_multilabel_decision::regularizer_multilabel_decision( regularizer *embedded )
{
  assert( embedded != NULL );
  _R = embedded;
  _B = new vector_valued_function_2D();
  _Ub = new vector_valued_function_2D();
}

regularizer_multilabel_decision::~regularizer_multilabel_decision()
{
  delete _Ub;
  delete _B;
}


// Access basis
vector_valued_function_2D *regularizer_multilabel_decision::basis()
{
  return _B;
}

const vector_valued_function_2D *regularizer_multilabel_decision::basis() const
{
  return _B;
}


// Resize problem (default dual variable layout)
bool regularizer_multilabel_decision::resize( compute_grid *G, int N )
{
  // For constraint update
  regularizer_multilabel::resize( G,N );

  // Embedded regularizer is 1D on the same grid
  _R->resize( G, 1 );

  // Number of dual variables is equal to embedded regularizer
  // plus one for constraint plus one for update buffer
  _K = _R->dual_dimension() + 2;
  _SIGMA = _R->dual_dimension();
  TRACE( "resizing decision, #dual=" << _K << "  sigma offset=" << _SIGMA << endl );

  // Extra primal same as internal regularizer
  _M = _R->extra_primal_dimension();
  TRACE( "                  extra primal = " << _M << endl );

  // Basis has one dimension per primal variable
  _B->alloc( G, _N );
  _Ub->alloc( G, 1 );
  return true;
}


// Perform primal step
bool regularizer_multilabel_decision::primal_step( vector_valued_function_2D *U,
						   float *step_U,
						   const vector_valued_function_2D *P,
						   vector_valued_function_2D *V,
						   float *step_V )
{
  assert( U != NULL );
  assert( P != NULL );
  assert_valid_dims( U,P, V );

  // Perform primal step on each outer variable, adjust by internal weight,
  // in temporary buffer
  for ( int i=0; i<U->N(); i++ ) {
    _Ub->channel(0).memcpy_from_engine( &U->channel( i ) );
    _R->primal_step( _Ub, step_U + i, P, V, step_V );

    // Difference is multiplied by alpha and added to original.
    kernel_subtract_from( U->grid(), U->channel(i), _Ub->channel(0) );
    kernel_multiply_and_add_to( U->grid(), _Ub->channel(0), _B->channel(i), U->channel(i) );
  }

  // Finally, primal step with regards to SIGMA
  //TRACE( "reg. primal step before/after" << endl );
  //U->trace_pixel( 100,100 );
  regularizer_multilabel::primal_step( U, step_U, P, NULL, NULL );
  //U->trace_pixel( 100,100 );
  return true;
}

			  
// Operator row sum norm of K for primal variables
bool regularizer_multilabel_decision::accumulate_operator_norm_U( float *norm_U ) const
{
  // Each variable U influenced by 1D regularizer
  for ( int i=0; i<_N; i++ ) {
    _R->accumulate_operator_norm_U( norm_U+i );
  }
//  for ( int i=0; i<_N; i++ ) {
//    norm_U[i] *= 5.0f;
//  }
  regularizer_multilabel::accumulate_operator_norm_U( norm_U );
  return true;
}

// Operator column sum norms of K and L for dual variables
bool regularizer_multilabel_decision::accumulate_operator_norm_P( float *norm_P ) const
{
  // Influenced by u of each step
  for ( int i=0; i<_N; i++ ) {
    _R->accumulate_operator_norm_P( norm_P );
  }
//  for ( int i=0; i<_R->dual_dimension(); i++ ) {
//    norm_P[i] *= 5.0f;
//  }
  regularizer_multilabel::accumulate_operator_norm_P( norm_P );
  return true;
}


// Operator column sum norms of K and L for dual variables
bool regularizer_multilabel_decision::accumulate_operator_norm_V( float *norm_V ) const
{
  _R->accumulate_operator_norm_V( norm_V );
  // since update for V is called N times, divide steps.
  // not efficient, TODO: separate updates for U,V
  for ( int i=0; i<_M; i++ ) {
    norm_V[i] /= float(_N);
  }
  return true;
}


// Perform primal reprojection
bool regularizer_multilabel_decision::primal_prox( vector_valued_function_2D *V  )
{
  return _R->primal_prox( V );
}

// Perform dual step
bool regularizer_multilabel_decision::dual_step( const vector_valued_function_2D *U,
						 vector_valued_function_2D* P,
						 float *step_P,
						 const vector_valued_function_2D *V )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert( P != NULL );
  assert_valid_dims( U,P, V );

  // Update for simplex constraint
  regularizer_multilabel::dual_step( U,P, step_P, NULL );
  //TRACE( "dual var before reg. dual step" << endl );
  //P->trace_pixel( 100,100 );

  // Dual step is performed for sum function
  // Force solution computation
  solution( U );
  return _R->dual_step( _Ub, P, step_P, V );
  //return true;
}




// Perform dual step
const vector_valued_function_2D *regularizer_multilabel_decision::solution( const vector_valued_function_2D *U )
{
  assert( U->equal_dim( _B ));
  assert( U->grid()->is_compatible( _Ub->grid() ) );

  // For now, slow implementation for quick testing
  _Ub->set_zero();
  for ( int i=0; i<U->N(); i++ ) {
    kernel_multiply_and_add_to( U->grid(), U->channel(i), _B->channel(i), _Ub->channel(0) );
  }

  return _Ub;
}




// Perform dual reprojection
bool regularizer_multilabel_decision::dual_prox( vector_valued_function_2D* P )
{
  //TRACE( "dual prox before/after" << endl );
  //P->trace_pixel( 100,100 );
  _R->dual_prox( P );
  //P->trace_pixel( 100,100 );
  return true;
}
