/* -*-c++-*- */
/** \file variational_model.cpp

    Base data structure for variational models,
    consisting of regularizer and data term.

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

#include "variational_model.h"

using namespace coco;


// Construction and destruction
variational_model::variational_model( regularizer *J, data_term *F )
{
  assert( J != NULL );
  assert( F != NULL );
  _J = J;
  _F = F;
  // reshape regularizer,
  // data term must be correctly initialized
  int N = F->N();
  _G = F->grid();
  assert( N>0 );
  _J->resize( _G, N );
}

variational_model::~variational_model()
{
}


regularizer *variational_model::J()
{
  return _J;
}

data_term *variational_model::F()
{
  return _F;
}

compute_grid* variational_model::grid()
{
  return _G;
}

