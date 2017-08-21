/* -*-c++-*- */
/** \file regularizer_vtv_f.cu

    Base data structure for the total variation regularizer.
    Reprojection for Frobenius norm.

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

#include "vtv_f.h"
#include "../compute_api/reprojections.h"


using namespace coco;

// Construction and destruction
regularizer_vtv_f::regularizer_vtv_f()
{
}

regularizer_vtv_f::~regularizer_vtv_f()
{
}

// Perform dual reprojection
bool regularizer_vtv_f::dual_prox( vector_valued_function_2D *P )
{
  assert( P != NULL );
  assert_valid_dims( NULL,P, NULL );
  assert( _N*2 <= P->N() );

  float lambda = get_parameter( "lambda" );
  reprojection_frobenius_norm( P, lambda, 0, _N );
  return true;
}
