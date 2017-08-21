/* -*-c++-*- */
/** \file stopping_criterion.cpp

    Base data structure for the stopping criterion of an iterative solve.

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

#include "stopping_criterion.h"

using namespace coco;

stopping_criterion::stopping_criterion( int maxiter )
{
  _maxiter = maxiter;
}

stopping_criterion::~stopping_criterion()
{
}


// Queries for algorithm implementation
bool stopping_criterion::stop( int iteration, vector_valued_function_2D *U ) const
{
  return ( iteration >= _maxiter );
}
