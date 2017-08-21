/** \file stopping_criterion.h

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

#ifndef __CUDA_STOPPING_CRITERION_H
#define __CUDA_STOPPING_CRITERION_H

#include "../compute_api/compute_array.h"

namespace coco {

  /// Base class for stopping criterion
  /***************************************************************
  Default implementation just stops after a fixed number of
  iterations.
  ****************************************************************/
  struct stopping_criterion
  {
    // Construction and destruction
    stopping_criterion( int maxiter = -1 );
    virtual ~stopping_criterion();

    // Queries for algorithm implementation
    virtual bool stop( int iteration, vector_valued_function_2D *U ) const;

  protected:
    // max number of iterations configured
    int _maxiter;
  };

};



#endif
