/* -*-c++-*- */
/** \file data_term_rof.cu

    Data term of the ROF model.

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

#include "rof.h"
#include "../compute_api/kernels_vtv.h"


using namespace coco;


// Construction and destruction
data_term_rof::data_term_rof()
{
  _params["lambda"] = 1.0f;
  _F = new vector_valued_function_2D;
}

data_term_rof::~data_term_rof()
{
  delete _F;
}

// Resize problem
bool data_term_rof::resize( compute_grid *G, int N )
{
  data_term::resize( G,N );

  // Shape function to be denoised
  _F->alloc( G,N );
  return true;
}

// Return prox function F
const vector_valued_function_2D *data_term_rof::F() const
{
  // must be allocated before retrieved, make sure to call resize once
  assert( _F );
  return _F;
}
vector_valued_function_2D *data_term_rof::F()
{
  // must be allocated before retrieved, make sure to call resize once
  assert( _F );
  return _F;
}


// Algorithm implementation

// Perform primal update, i.e. gradient step + reprojection
bool data_term_rof::primal_update( vector_valued_function_2D *U,
				   float *step_U,
				   vector_valued_function_2D *Q )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert_valid_dims( U,Q );
  assert( U->equal_dim( _F ));
  float lambda = get_parameter( "lambda" );

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0

  // Kernel call for each channel
  for ( int i=0; i<U->N(); i++ ) {
    kernel_rof_primal_prox
      ( U->grid(),
	step_U[i],
	U->channel(i),
	lambda,
	_F->channel(i) );
    TRACE9( "  primal update U(" << i << ") step size " << step_U[i] << "  lambda " << lambda << std::endl );
  }

  // Q should not be in use
  assert( Q==NULL );
  return true;
}
