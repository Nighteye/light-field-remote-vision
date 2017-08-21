/* -*-c++-*- */
/** \file multilabel_linear_cost.cpp

    Data term of the multilabel model, linear assignment cost

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

#include "multilabel_linear_cost.h"
#include "../compute_api/kernels_algebra.h"


// temp, needs to be placed in cc++
#include "../../cuda/simplex_reprojection.h"


using namespace coco;


// Construction and destruction
multilabel_linear_cost::multilabel_linear_cost()
{
  _params["lambda"] = 1.0f;
  _a = new vector_valued_function_2D;
}

multilabel_linear_cost::~multilabel_linear_cost()
{
  delete _a;
}

// Resize problem
bool multilabel_linear_cost::resize( compute_grid *G, int N )
{
  data_term::resize( G,N );
  _a->alloc( G,N );
  return true;
}

// Return prox function F
const vector_valued_function_2D *multilabel_linear_cost::a() const
{
  // must be allocated before retrieved, make sure to call resize once
  assert( _a );
  return _a;
}
vector_valued_function_2D *multilabel_linear_cost::a()
{
  // must be allocated before retrieved, make sure to call resize once
  assert( _a );
  return _a;
}


// Algorithm implementation

// Perform primal update, i.e. gradient step + reprojection
bool multilabel_linear_cost::primal_update( vector_valued_function_2D *U,
				   float *step_U,
				   vector_valued_function_2D *Q )
{
  // Default is one gradient operator step
  assert( U != NULL );
  assert_valid_dims( U,Q );
  assert( U->equal_dim( _a ));
  float lambda = get_parameter( "lambda" );

  //TRACE( "testing mlc update step lambda=" << lambda << endl );
  //TRACE( "  step sizes " );
  //for ( int i=0; i<U->N(); i++ ) {
  //  TRACE( step_U[i] << " " );
  //}
  //TRACE( endl );
  //U->trace_pixel( 100,100 );
  //_a->trace_pixel( 100,100 );


  // Kernel call for each channel
  for ( int i=0; i<U->N(); i++ ) {
    kernel_linear_combination
      ( U->grid(),
	-lambda * step_U[i],
	_a->channel(i),
	1.0f,
	U->channel(i),
	U->channel(i) );

/*
    kernel_clamp
      ( U->grid(),
	U->channel(i),
	0.0f, 1.0f );
    U->trace_pixel( 100,100 );
*/
    TRACE9( "  primal update U(" << i << ") step size " << step_U[i] << "  lambda " << lambda << std::endl );
  }

  // projection
  //simplex_reprojection( U->grid()->W(), U->grid()->H(), U->N(), U->buffer() );

  //U->trace_pixel( 100,100 );

  // Q should not be in use
  assert( Q==NULL );
  return true;
}
