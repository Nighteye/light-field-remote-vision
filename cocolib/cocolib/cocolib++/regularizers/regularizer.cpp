/* -*-c++-*- */
/** \file regularizer.cu

    Base data structure for the regularizer of an inverse problem.

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

#include "regularizer.h"
#include "../compute_api/kernels_vtv.h"

using namespace coco;
using namespace std;

bool regularizer::assert_valid_dims( const vector_valued_function_2D *U,
				     const vector_valued_function_2D *P,
				     const vector_valued_function_2D *W ) const
{
  // Regularizer might be used in embedded mode,
  // so just make sure dimensions are at least sufficient
  if ( U != NULL ) {
    assert( U->N() >= _N );
    assert( U->grid() == grid() );
  }
  if ( P != NULL ) {
    assert( P->N() >= _K );
    assert( P->grid() == grid() );
  }
  if ( W != NULL ) {
    assert( W->N() >= _M );
    assert( W->grid() == grid() );
  }
  return true;
}


// Construction and destruction
regularizer::regularizer()
{
  _N = 0;
  _K = 0;
  _M = 0;
  _G = NULL;
  _params["lambda"] = 1.0f;
}

regularizer::~regularizer()
{
}


// Resize problem (default dual variable layout)
bool regularizer::resize( compute_grid *G, int N )
{
  assert( N>0 );
  assert( G != NULL );
  _N = N;
  _G = G;
  _K = 2*N;
  _M =0;
  return true;
}

// Regularizer parameter set
bool regularizer::set_parameter( const string &param, float value )
{
  map<string,float>::iterator it = _params.find( param );
  if ( it == _params.end() ) {
    ERROR( "Parameter '" << param << "' not supported by the regularizer." << endl );
    return false;
  }
  (*it).second = value;
  return true;
}


// Regularizer parameter get
float regularizer::get_parameter( const string &param ) const
{
  map<string,float>::const_iterator it = _params.find( param );
  if ( it == _params.end() ) {
    ERROR( "Parameter '" << param << "' not supported by the regularizer." << endl );
    return 0.0f;
  }
  return (*it).second;
}


// Get problem size
int regularizer::N() const
{return _N;}

compute_grid *regularizer::grid() const
{return _G;}

// Total dimension of dual variables
int regularizer::dual_dimension() const
{return _K;}

// Total dimension of additional primal variables w (excludes solution)
int regularizer::extra_primal_dimension() const
{return _M;}


// Perform primal step
bool regularizer::primal_step( vector_valued_function_2D *U,
			       float *step_U,
			       const vector_valued_function_2D *P,
			       vector_valued_function_2D *V,
			       float *step_V )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert( P != NULL );
  assert_valid_dims( U,P, V );

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0

  // Kernel call for each channel
  for ( int i=0; i<U->N(); i++ ) {
    kernel_gradient_operator_primal_step
      ( _G,
	step_U[i],
	U->channel(i),
	P->channel(2*i + 0), P->channel(2*i + 1) );
  }

  // no sensible default implementation for V
  // derive if primal variables W are required
  assert( V==NULL );
  return true;
}

			  
// Operator row sum norm of K for primal variables
bool regularizer::accumulate_operator_norm_U( float *norm_U ) const
{
  // Default for gradient operator
  for ( int i=0; i<_N; i++ ) {
    norm_U[i] += 4.0f;
  }
  return true;
}

// Operator column sum norms of K and L for dual variables
bool regularizer::accumulate_operator_norm_P( float *norm_P ) const
{
  for ( int i=0; i<2*_N; i++ ) {
    norm_P[i] += 2.0f;
  }
  return true;
}


// Operator row sum norm of L for extra primal variables
bool regularizer::accumulate_operator_norm_V( float *norm_V ) const
{
  // no sensible default implementation
  // derive if extra primal variables are used
  assert( false );
  return false;
}


// Perform primal reprojection
bool regularizer::primal_prox( vector_valued_function_2D *V  )
{
  // no sensible default implementation
  // derive if primal variables V are required
  assert( V==NULL );
  return true;
}

// Perform dual step
bool regularizer::dual_step( const vector_valued_function_2D *U,
			     vector_valued_function_2D* P,
			     float *step_P,
			     const vector_valued_function_2D *V )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert( P != NULL );
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

  // this is just a default gradient dual step which does not support updates in V
  assert( V == NULL );
  return true;
}


// Perform dual reprojection
bool regularizer::dual_prox( vector_valued_function_2D* P )
{
  // no sensible default implementation
  return false;
}

