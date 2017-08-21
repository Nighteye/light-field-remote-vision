/* -*-c++-*- */
/** \file data_term.cu

    Base data structure for the data term of an inverse problem.

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

#include "data_term.h"
#include "../compute_api/kernels_vtv.h"


using namespace coco;
using namespace std;

bool data_term::assert_valid_dims( const vector_valued_function_2D *U,
				   const vector_valued_function_2D *Q ) const
{
  assert( U != NULL );
  assert( U->N() == _N );
  assert( U->grid() == grid() );
  if ( Q != NULL ) {
    assert( Q->N() == _K );
    assert( Q->grid() == grid() );
  }
  return true;
}


// Construction and destruction
data_term::data_term()
{
  _N = 0;
  _K = 0;
  _G = NULL;
}

// Resize problem
bool data_term::resize( compute_grid *G, int N )
{
  assert( N>0 );
  assert( G != NULL );
  _N = N;
  _G = G;
  _K = 0;
  return true;
}


data_term::~data_term()
{
}



// Dataterm parameter set
bool data_term::set_parameter( const string &param, float value )
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
float data_term::get_parameter( const string &param ) const
{
  map<string,float>::const_iterator it = _params.find( param );
  if ( it == _params.end() ) {
    ERROR( "Parameter '" << param << "' not supported by the regularizer." << endl );
    return 0.0f;
  }
  return (*it).second;
}


// Total dimension of additional dual variables
int data_term::dual_dimension() const
{
  return _K;
}

// Get problem size
int data_term::N() const
{return _N;}

compute_grid* data_term::grid() const
{return _G;}


// Operator row sum norm of B for primal variables
bool data_term::accumulate_operator_norm_U( float *norm_U ) const
{
  // default: no operator in data term
  return true;
}

// Operator column sum norm of B for dual variables
bool data_term::accumulate_operator_norm_Q( float *norm_Q ) const
{
  // default: no operator in data term
  return true;
}


// Algorithm implementation

// Perform primal update, i.e. gradient step + reprojection
bool data_term::primal_update( vector_valued_function_2D *U,
			       float *stepU,
			       vector_valued_function_2D *Q )
{
  // no valid default implementation
  assert( false );
  return false;
}

// Perform dual update, i.e. gradient step + reprojection
bool data_term::dual_update( const vector_valued_function_2D *U,
			     vector_valued_function_2D *Q,
			     float *step_Q )
{
  // no valid default implementation
  assert( Q == NULL );
  return false;
}
